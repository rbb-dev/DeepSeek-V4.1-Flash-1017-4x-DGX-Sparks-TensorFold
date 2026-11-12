// The device half of DeepSeek-V4.1's one-hop RDMA all-gather (the Python family's cuda/rdma_gather.cu v7). The staging
// kernel copies this rank's fp32 slice into a pinned host ring slot and rings a doorbell; the host's proxy thread
// (rdma.zig) RDMA-writes the slot to every peer with an immediate and publishes a peer's flag once all its parts
// are in; the collecting kernels wait for the flags and copy the slots out in rank order, or straight into the caller's
// views (gather_into). The kernel bodies are the served build's; only their names are the family's.
// v7: a nonzero `pdl` makes each kernel wait for the one before it and let the next one launch (programmatic dependent
// launches, sm_90+), so the pair does not break a chain of early launches.

#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr int STAGE_THREADS = 512;
constexpr int COLLECT_THREADS = 256;

__device__ __forceinline__ uint64_t load_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// A poll turn without the acquire's L1 invalidate (SASS: ld.acquire.sys = LDG.STRONG.SYS + CCTL.IVALL every turn, ld.relaxed.sys = the bare
// LDG.STRONG.SYS): a loop polls relaxed and takes ONE acquire load of the same word after it sees the value, which gives the ordering for the
// data read afterwards (the word is monotone: the acquire load sees the value the relaxed one did, or a later one).
__device__ __forceinline__ uint64_t load_relaxed_sys(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t now_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void store_relaxed_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}


// v7 (TF_RDMA_PDL): wait for the kernel before this one (done, its memory visible), then let the next one launch
__device__ __forceinline__ void pdl_enter(int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}


}  // namespace

// Each block: wait until the slot's last writes are done, copy its share of the partial in; the last block to finish
// (a counter) publishes the size and the doorbell. (One block took ~6 us for a 6-row window's 120 KB and ~35 us for a
// 6-row head's 1.5 MB into the pinned ring on GB10; 8-16 blocks 2.5 and 9.) Every block reads the sequence before it
// counts itself in, and only the last one, after every block has, advances it.
extern "C" __global__ void __launch_bounds__(STAGE_THREADS) dsv41_rdma_stage(const float4* __restrict__ src, int n4,
        char* send_ring, uint64_t* sizes, uint64_t* doorbell, const uint64_t* sent, uint64_t* seq, int slots,
        uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, unsigned* staged, int fence_mode,
        float4* __restrict__ own_dst, int own_row4, long long own_stride4, int pdl) {
    __shared__ uint64_t s_seq;
    __shared__ int s_last;
    pdl_enter(pdl);                              // before anything reads the partial or this gather's sequence
    const uint64_t t0 = now_ns();
    if (threadIdx.x == 0) {
        const uint64_t q = *seq + 1;
        const int slot = (int)(q % slots);
        if (q > (uint64_t)slots)
            for (uint32_t i = 1; load_sys(sent + slot) + slots < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
        s_seq = q;
        // fence_mode 1: the size is written before this block's release below, so the publish needs no second fence
        if (fence_mode && blockIdx.x == 0) store_relaxed_sys(sizes + slot, (uint64_t)n4 * 16);
    }
    __syncthreads();
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    float4* dst = reinterpret_cast<float4*>(send_ring + (size_t)slot * max_bytes);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
    if (fence_mode) {
        // the barrier orders every thread's stores before thread 0's system fence (a release for the whole block, as
        // NCCL publishes its proxy FIFO), then the counter
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
        }
    } else {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *staged = 0;
        *seq = q;
        if (fence_mode) {
            // the fence above acquired every block's release (data and size) and orders them before this strong store
            store_relaxed_sys(doorbell, q);
        } else {
            store_sys(sizes + slot, (uint64_t)n4 * 16);
            store_sys(doorbell, q);
        }
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
    if (own_dst) {
        // TF_RDMA_OWN_EARLY: this rank's own slice of the output, copied after the doorbell (during the RDMA writes and
        // the peer's) instead of by the collecting kernel after the wait; the same bytes from the same source
        if (s_last) __syncthreads();             // the publishing block lets its doorbell out first (s_last is uniform)
        if (own_stride4 == own_row4) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) own_dst[i] = src[i];
        } else {                                 // gather_into: rows of own_row4 float4, own_stride4 apart
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = src[i];
            }
        }
    }
}


// The in-place send's wait: before its producer writes this stream's fixed send slot, the previous send from that slot
// must have completed (the NIC has read the bytes the next producer is about to replace). `send_last` holds the
// sequence the producer last published for the user; `send_done` the sequence the proxy last completed.
extern "C" __global__ void __launch_bounds__(32) dsv41_rdma_await_send(const uint64_t* send_done,
        const uint64_t* send_last, int user, const uint64_t* abort_word, int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // this kernel waits for the one before it but must not let the producer that follows it launch early: pdl_enter
    // would trigger the next kernel before the spin below, so only the wait is here (the producer's wait then waits for
    // this kernel's end)
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (threadIdx.x == 0) {
        const uint64_t want = load_sys(send_last + user);
        for (uint32_t i = 1; load_sys(send_done + user) < want; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    }
}

// The in-place send's publish: the producer has written this stream's fixed send slot, so this kernel copies this
// rank's own slice to its destination (as the staging kernel did; none when own_dst is null) and publishes the slot:
// its size and name at the rotating sequence's slot, the user's last sequence, then the sequence and the doorbell
// (a system release, as the staging kernel's). The send slot's previous use was awaited before the producer wrote it;
// the rotating slot's metadata is held off as the staging kernel held its slot.
extern "C" __global__ void __launch_bounds__(STAGE_THREADS) dsv41_rdma_publish(uint64_t* sizes, uint64_t* doorbell,
        uint64_t* seq, uint64_t* send_q, uint64_t* send_slot, uint64_t* send_last, int slots, int user,
        const uint64_t* sent, const uint64_t* abort_word, const float4* __restrict__ own_src,
        float4* __restrict__ own_dst, int own_row4, long long own_stride4, uint64_t nbytes, uint64_t* probe, int pdl) {
    pdl_enter(pdl);                              // the producer's partial and own slice are written
    const uint64_t t0 = now_ns();
    const uint64_t q = *seq + 1;
    const int slot = (int)(q % slots);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (q > (uint64_t)slots)
            for (uint32_t i = 1; load_sys(sent + slot) + slots < q; ++i)
                if ((i & 1023) == 0 && load_sys(abort_word)) break;
        store_relaxed_sys(sizes + slot, nbytes);
        store_relaxed_sys(send_q + slot, q);
        store_relaxed_sys(send_slot + slot, (uint64_t)user);
        store_relaxed_sys(send_last + user, q);
        *seq = q;
        __threadfence_system();
        store_relaxed_sys(doorbell, q);
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
    if (own_dst) {
        const int n4 = (int)(nbytes / 16);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
            if (own_stride4 == own_row4) {
                own_dst[i] = own_src[i];
            } else {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = own_src[i];
            }
        }
    }
}


// Wait for every peer's flag at this sequence, then copy the slots (and this rank's own source) out in rank order.
extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect(const float4* __restrict__ own,
        float4* __restrict__ dst, int n4, const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int rank,
        int world, int slots, uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, int skip_own, int unroll,
        int pdl) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    if (threadIdx.x < world && threadIdx.x != rank)
        for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    __syncthreads();
    const uint64_t t2 = probe ? now_ns() : 0;
    for (int p = 0; p < world; ++p) {
        float4* out = dst + (size_t)p * n4;
        if (p == rank) {
            if (skip_own) continue;              // the staging kernel wrote it
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) out[i] = own[i];
        } else {
            const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
            if (unroll) {
                // TF_RDMA_COLLECT_UNROLL: up to four loads of the ring in flight a thread before its stores (the ring's
                // first reads after the flag wait out a DRAM queue when a prefetch runs beside)
                const int stride = gridDim.x * blockDim.x;
                for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n4; b += 4 * stride) {
                    float4 v[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (b + u * stride < n4) v[u] = __ldcv(in + b + u * stride);
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (b + u * stride < n4) out[b + u * stride] = v[u];
                }
            } else {
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
                    out[i] = __ldcv(in + i);
            }
        }
    }
    if (probe && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            probe[1] += t2 - probe[4];
            probe[2] += now_ns() - t2;
            probe[3] += 1;
        }
    }
}


constexpr int MAXW = 8;

// gather_into's destinations: rank p's slice is rows x row4[p] float4 (packed in its ring slot), copied to rows
// stride4[p] float4 apart from ptr[p]; ptr[p] null: not copied out.
struct Dsts {
    float4* ptr[MAXW];
    long long stride4[MAXW];
    int row4[MAXW];
};

// Wait for every peer's flag at this sequence, then copy each peer's slot to its destination rows (this rank's own
// slice was written by the staging kernel).
extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect_into(Dsts d, int rows,
        const char* recv_ring, const uint64_t* flags, const uint64_t* seq, int rank, int world, int slots,
        uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, int pdl) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    const uint64_t q = *seq;
    const int slot = (int)(q % slots);
    if (threadIdx.x < world && threadIdx.x != rank)
        for (uint32_t i = 1; load_sys(flags + (size_t)slot * world + threadIdx.x) < q; ++i)
            if ((i & 1023) == 0 && load_sys(abort_word)) break;
    __syncthreads();
    const uint64_t t2 = probe ? now_ns() : 0;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < world; ++p) {
        if (p == rank || d.ptr[p] == nullptr) continue;
        const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
        float4* out = d.ptr[p];
        const int r4 = d.row4[p], n4 = rows * r4;
        const long long s4 = d.stride4[p];
        // up to four ring loads in flight a thread before its stores (as dsv41_rdma_collect's unrolled loop)
        for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n4; b += 4 * stride) {
            float4 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u)
                if (b + u * stride < n4) v[u] = __ldcv(in + b + u * stride);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = b + u * stride;
                if (i < n4) {
                    const int r = i / r4;
                    out[(size_t)r * s4 + (i - r * r4)] = v[u];
                }
            }
        }
    }
    if (probe && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            probe[1] += t2 - probe[4];
            probe[2] += now_ns() - t2;
            probe[3] += 1;
        }
    }
}



// TF_DS_RING_WORD: the arrival signal is the sending lane's 64-bit generation word, written by its NIC after the
// data on the same reliable queue pair. The waiting kernel spins on every lane's word (ld.acquire.sys) until the word
// is at least the sequence, bounded by the deadline and the ring's abort word; the first expired wait writes a failure
// record and poisons the ring, and the kernel copies nothing. MAX_LANES is rdma.zig's max_devices: the words are
// [peer][lane] in the receiver's region, and the sender's lane index is the receiver's (the device pairing is
// symmetric).
constexpr int MAX_LANES = 4;
static_assert(MAX_LANES == 4, "MAX_LANES must equal rdma.zig's max_devices (4): the generation words are [peer][lane]");

__device__ __forceinline__ void roci_record(uint64_t* fault, uint64_t* abort_word, uint64_t q, int rank, int peer,
        int lane, uint64_t seen, uint64_t waited, uint64_t why) {
    // one record a ring: the first expiring wait tags the record, later ones only keep the poison
    unsigned long long* f = reinterpret_cast<unsigned long long*>(fault);
    if (atomicCAS(f, 0ull, 0x524f434900000001ull) == 0ull) {
        fault[1] = q;
        fault[2] = (uint64_t)rank;
        fault[3] = ((uint64_t)(uint32_t)peer << 32) | (uint32_t)lane;
        fault[4] = seen;
        fault[5] = waited;
        fault[6] = now_ns();
        fault[7] = why;                          // 1: the abort word, 2: the deadline
    }
    store_sys(abort_word, 1);
}

// Whether every lane's word reached sequence q: 0 when it did, 1 when the abort word or the deadline ended the wait
// (the record is written and the ring poisoned). Every thread reaches the barrier: a failed lane breaks its own poll
// and the others see the poison within 1,024 spins. A failed lane is every thread's through the barrier's own vote
// (__syncthreads_or), so no thread loads the abort word from host memory after the wait; the deadline's clock is read
// once a lane has had to wait at all.
__device__ __forceinline__ int roci_waited(const uint64_t* gen, const unsigned* lanes, int rank, int world,
        uint64_t q, uint64_t deadline_ns, uint64_t* abort_word, uint64_t* fault) {
    const int total = world * MAX_LANES;
    int failed = 0;
    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        const int p = t / MAX_LANES, k = t % MAX_LANES;
        if (p == rank || k >= (int)lanes[p]) continue;
        uint64_t t0 = 0;
        uint64_t seen = 0;
        for (uint32_t i = 1; (seen = load_relaxed_sys(gen + (size_t)p * MAX_LANES + k)) < q; ++i) {
            if (i == 1) t0 = now_ns();           // the word was not there yet: the wait's start
            if ((i & 1023) == 0) {
                if (load_sys(abort_word) != 0) {
                    roci_record(fault, abort_word, q, rank, p, k, seen, now_ns() - t0, 1);
                    failed = 1;
                    break;
                }
                if (deadline_ns != 0 && now_ns() - t0 >= deadline_ns) {
                    roci_record(fault, abort_word, q, rank, p, k, seen, now_ns() - t0, 2);
                    failed = 1;
                    break;
                }
            }
        }
        if (!failed) seen = load_sys(gen + (size_t)p * MAX_LANES + k);   // the acquire edge, once (the data read after the barrier)
    }
    return __syncthreads_or(failed) != 0;
}


// dsv41_rdma_collect with the generation-word wait: the same bytes from the same slots once every lane's word is in.
extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect_roci(
        const float4* __restrict__ own, float4* __restrict__ dst, int n4, const char* recv_ring,
        const uint64_t* gen, const unsigned* lanes, const uint64_t* seq, int rank, int world, int slots,
        uint64_t max_bytes, uint64_t* probe, uint64_t* abort_word, int skip_own, int unroll, uint64_t deadline_ns,
        uint64_t* fault, int pdl) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    const uint64_t q = *seq;
    if (roci_waited(gen, lanes, rank, world, q, deadline_ns, abort_word, fault)) return;
    const int slot = (int)(q % slots);
    const uint64_t t2 = probe ? now_ns() : 0;
    for (int p = 0; p < world; ++p) {
        float4* out = dst + (size_t)p * n4;
        if (p == rank) {
            if (skip_own) continue;              // the staging kernel wrote it
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) out[i] = own[i];
        } else {
            const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
            if (unroll) {
                const int stride = gridDim.x * blockDim.x;
                for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n4; b += 4 * stride) {
                    float4 v[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (b + u * stride < n4) v[u] = __ldcv(in + b + u * stride);
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (b + u * stride < n4) out[b + u * stride] = v[u];
                }
            } else {
                for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
                    out[i] = __ldcv(in + i);
            }
        }
    }
    if (probe && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            probe[1] += t2 - probe[4];
            probe[2] += now_ns() - t2;
            probe[3] += 1;
        }
    }
}


// dsv41_rdma_collect_into with the generation-word wait.
extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect_into_roci(Dsts d, int rows,
        const char* recv_ring, const uint64_t* gen, const unsigned* lanes, const uint64_t* seq, int rank, int world,
        int slots, uint64_t max_bytes, uint64_t* probe, uint64_t* abort_word, uint64_t deadline_ns, uint64_t* fault,
        int pdl) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    const uint64_t q = *seq;
    if (roci_waited(gen, lanes, rank, world, q, deadline_ns, abort_word, fault)) return;
    const int slot = (int)(q % slots);
    const uint64_t t2 = probe ? now_ns() : 0;
    const int stride = gridDim.x * blockDim.x;
    for (int p = 0; p < world; ++p) {
        if (p == rank || d.ptr[p] == nullptr) continue;
        const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
        float4* out = d.ptr[p];
        const int r4 = d.row4[p], n4 = rows * r4;
        const long long s4 = d.stride4[p];
        for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n4; b += 4 * stride) {
            float4 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u)
                if (b + u * stride < n4) v[u] = __ldcv(in + b + u * stride);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = b + u * stride;
                if (i < n4) {
                    const int r = i / r4;
                    out[(size_t)r * s4 + (i - r * r4)] = v[u];
                }
            }
        }
    }
    if (probe && blockIdx.x == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            probe[1] += t2 - probe[4];
            probe[2] += now_ns() - t2;
            probe[3] += 1;
        }
    }
}


// dsv41_rdma_collect_peers with the generation-word wait: a block row waits on its own peer's lanes.
extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect_peers_roci(Dsts d, int rows,
        const char* recv_ring, const uint64_t* gen, const unsigned* lanes, const uint64_t* seq, int rank, int world,
        int slots, uint64_t max_bytes, uint64_t* abort_word, uint64_t deadline_ns, uint64_t* fault, int pdl,
        const float4* __restrict__ own_src, float4* __restrict__ own_dst, int own_row4, long long own_stride4, int own_n4) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    // This rank's own slice, copied here (the first row of blocks) while the peers' words are still on their way: the publishing kernel
    // does not copy it, so its doorbell is not behind the copy. The same bytes to the same words; the consumer reads them after this kernel.
    if (own_dst != nullptr && blockIdx.y == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < own_n4; i += gridDim.x * blockDim.x) {
            if (own_stride4 == own_row4) {
                own_dst[i] = own_src[i];
            } else {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = own_src[i];
            }
        }
    }
    int p = -1;
    for (int k = 0, y = (int)blockIdx.y; k < world; ++k)
        if (k != rank && d.ptr[k] != nullptr && d.row4[k] != 0 && y-- == 0) {
            p = k;
            break;
        }
    if (p < 0) return;                           // no peer for this row of blocks (a grid of one with none to copy)
    const uint64_t q = *seq;
    // everything that does not depend on the wait is computed before it (the 64-bit modulo and the row division are library routines of ~80
    // and ~40 instructions): the wait then ends in the copy-out's loads
    const int slot = (int)(q % slots);
    const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
    float4* out = d.ptr[p];
    const int r4 = d.row4[p];
    const long long s4 = d.stride4[p];
    const int per_row = (r4 + PEER_UNIT4 - 1) / PEER_UNIT4;
    // a failed wait is the block's: the poison ends it (the vote below)
    int failed = 0;
    const int nl = (int)lanes[p];
    if (threadIdx.x < nl) {                      // the peer's lanes are polled together, one thread a lane (the peer's words are independent)
        const int k = (int)threadIdx.x;
        uint64_t t0 = 0;
        uint64_t seen = 0;
        for (uint32_t i = 1; (seen = load_relaxed_sys(gen + (size_t)p * MAX_LANES + k)) < q; ++i) {
            if (i == 1) t0 = now_ns();               // the word was not there yet: the wait's start
            if ((i & 1023) == 0) {
                if (load_sys(abort_word) != 0) {
                    roci_record(fault, abort_word, q, rank, p, k, seen, now_ns() - t0, 1);
                    failed = 1;
                    break;
                }
                if (deadline_ns != 0 && now_ns() - t0 >= deadline_ns) {
                    roci_record(fault, abort_word, q, rank, p, k, seen, now_ns() - t0, 2);
                    failed = 1;
                    break;
                }
            }
        }
        if (!failed) seen = load_sys(gen + (size_t)p * MAX_LANES + k);   // the acquire edge, once
    }
    if (__syncthreads_or(failed) != 0) return;     // a failed wait is the block's: the vote, not a load of the abort word by every thread
    for (int u = blockIdx.x; u < rows * per_row; u += gridDim.x) {
        const int r = per_row == 1 ? u : u / per_row;     // (a row of up to 1,024 float4 is one unit: every decode exchange; no division then)
        const int c0 = (per_row == 1 ? 0 : (u - r * per_row) * PEER_UNIT4) + threadIdx.x;
        const float4* src = in + (size_t)r * r4;
        float4* dst = out + (size_t)r * s4;
        float4 v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (c0 + k * COLLECT_THREADS < r4) v[k] = __ldcv(src + c0 + k * COLLECT_THREADS);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (c0 + k * COLLECT_THREADS < r4) dst[c0 + k * COLLECT_THREADS] = v[k];
    }
}


// ---- TF_DS_RING_WAIT: the decode rings with no proxy thread (ring_wait.zig) ---------------------------------------------------------
// The card waits on the doorbell word (stored BIG-ENDIAN: it compares the 64-bit word as a big-endian number) and sends the chains the
// engine thread wrote into its queues at launch. The send credit is the card's completion of exchange q's word WRITE, read here from
// the queue pairs' completion queues (host memory): entry q - 1 of every queue pair's queue, present with its lap's owner bit and not an
// error. `sent[]` and `send_done[]` are not used in this mode. The kernels below are the staging, publish and await_send kernels above
// with those two changes and no others.

struct CqRef {                                   // ring_wait.zig CqRef: one queue pair's completion queue as the card writes it
    const unsigned char* base;
    unsigned cnt;                                // entries, a power of two
    unsigned size;                               // an entry's bytes: 64, or 128 with the entry in the second half
};
static_assert(sizeof(CqRef) == 16, "ring_wait.zig's CqRef is 16 bytes");

// ring_wait.zig cqeState: 0 empty (not written by the card, or the last lap's), 1 ready (a successful completion of this lap), -1 failed
__host__ __device__ __forceinline__ int cqe_state(unsigned op_own, uint64_t idx, unsigned cnt) {
    const unsigned opc = op_own >> 4;
    if (opc == 0xf) return 0;
    const unsigned lap = (idx & cnt) ? 1u : 0u;      // the lap bit: bit log2(cnt) of the entry number (cnt is a power of two)
    if ((op_own & 1u) != lap) return 0;
    if (opc == 0xd || opc == 0xe) return -1;
    return 1;
}

constexpr int MAX_CQ = 16;

__device__ __forceinline__ uint64_t bswap64_dev(uint64_t v) {
    const uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
    return ((uint64_t)__byte_perm(lo, 0, 0x0123) << 32) | (uint64_t)__byte_perm(hi, 0, 0x0123);
}

struct CqTab {                                   // ring_wait.zig CqTab: the lanes' queues, passed by value (no load of the table from host memory)
    unsigned n;
    unsigned pad;
    CqRef e[MAX_CQ];
};
static_assert(sizeof(CqTab) == 264, "ring_wait.zig's CqTab is 264 bytes");

// Wait, with warp 0 (all 32 lanes call it together), until entry `idx` of every queue pair's completion queue is a ready completion: lane c reads
// queue pair c's entry, so the lanes' loads are in flight together (one latency, not one a lane). Gives up (returns false) on an error completion
// or when the abort word is set (checked every 1024 rounds, as the proxy path's credit spin). An entry's last 8 bytes carry op_own as the top byte.
__device__ __forceinline__ bool cq_wait_warp(const CqTab& t, uint64_t idx, const uint64_t* abort_word) {
    const int lane = threadIdx.x & 31;
    const int n = t.n > (unsigned)MAX_CQ ? MAX_CQ : (int)t.n;
    for (uint32_t i = 1;; ++i) {
        int s = 1;
        if (lane < n) {
            const CqRef r = t.e[lane];
            const unsigned char* e = r.base + (size_t)(idx & (r.cnt - 1)) * r.size + (r.size == 128 ? 64 : 0);
            const unsigned own = (unsigned)(load_sys(reinterpret_cast<const uint64_t*>(e + 56)) >> 56);
            s = cqe_state(own, idx, r.cnt);
        }
        if (__any_sync(0xffffffffu, s < 0)) return false;
        if (!__any_sync(0xffffffffu, s == 0)) return true;
        const int ab = ((i & 1023) == 0 && load_sys(abort_word) != 0) ? 1 : 0;
        if (__any_sync(0xffffffffu, ab)) return false;
    }
}

extern "C" __global__ void __launch_bounds__(STAGE_THREADS) dsv41_rdma_stage_wait(const float4* __restrict__ src, int n4,
        char* send_ring, uint64_t* sizes, uint64_t* doorbell, const CqTab cq, uint64_t* seq, int slots,
        uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, unsigned* staged, int fence_mode,
        float4* __restrict__ own_dst, int own_row4, long long own_stride4, int pdl) {
    __shared__ uint64_t s_seq;
    __shared__ int s_last;
    pdl_enter(pdl);
    const uint64_t t0 = probe ? now_ns() : 0;     // (the probe is null in a served engine)
    if (threadIdx.x < 32) {                      // warp 0: its lanes read the queue pairs' completions together
        const uint64_t q = *seq + 1;
        if (q > (uint64_t)slots) cq_wait_warp(cq, q - slots - 1, abort_word);
        if (threadIdx.x == 0) s_seq = q;
    }
    __syncthreads();
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    float4* dst = reinterpret_cast<float4*>(send_ring + (size_t)slot * max_bytes);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
    if (fence_mode) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
        }
    } else {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *staged = 0;
        *seq = q;
        if (fence_mode) {
            store_relaxed_sys(doorbell, bswap64_dev(q));
        } else {
            store_sys(doorbell, bswap64_dev(q));
        }
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
    if (own_dst) {
        if (s_last) __syncthreads();
        if (own_stride4 == own_row4) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) own_dst[i] = src[i];
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = src[i];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(STAGE_THREADS) dsv41_rdma_publish_wait(uint64_t* sizes, uint64_t* doorbell,
        uint64_t* seq, uint64_t* send_q, uint64_t* send_slot, uint64_t* send_last, int slots, int user,
        const CqTab cq, const uint64_t* abort_word, const float4* __restrict__ own_src,
        float4* __restrict__ own_dst, int own_row4, long long own_stride4, uint64_t nbytes, uint64_t* probe, int pdl) {
    pdl_enter(pdl);
    const uint64_t t0 = probe ? now_ns() : 0;     // (the probe is null in a served engine)
    const uint64_t q = *seq + 1;
    if (blockIdx.x == 0 && threadIdx.x < 32) {   // warp 0 of block 0
        if (q > (uint64_t)slots) cq_wait_warp(cq, q - slots - 1, abort_word);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // (sizes[], send_q[] and send_slot[] are the proxy path's words: with the queue pairs waiting in the card the chain carries the size
        // and the slot, and nothing reads them; send_last is read by dsv41_rdma_await_send_wait)
        store_relaxed_sys(send_last + user, q);
        *seq = q;
        __threadfence_system();
        store_relaxed_sys(doorbell, bswap64_dev(q));
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
    if (own_dst) {
        const int n4 = (int)(nbytes / 16);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
            if (own_stride4 == own_row4) {
                own_dst[i] = own_src[i];
            } else {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = own_src[i];
            }
        }
    }
}

// The in-place send's wait in this mode: the previous send from the user's fixed slot is complete when its exchange's completion entry is.
extern "C" __global__ void __launch_bounds__(32) dsv41_rdma_await_send_wait(const CqTab cq,
        const uint64_t* send_last, int user, const uint64_t* abort_word, int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    const uint64_t want = load_sys(send_last + user);   // the block is one warp
    if (want > 0) cq_wait_warp(cq, want - 1, abort_word);
}


// ---- TF_DS_ROCI_XLOG: the same three kernels with the exchange log (log only; the host launches these ONLY with TF_DS_ROCI_XLOG=1) ----------------
// Each is its kernel above with the lines that mention xlog added, and nothing else: the unsuffixed kernels carry no logging code at all, so with the log
// off the decode exchange executes the instructions it executed before the log existed (ring_hotpath_test pins both: the unsuffixed bodies hold no
// "xlog", and an _x body without its xlog lines is its original's body).
extern "C" __global__ void __launch_bounds__(STAGE_THREADS) dsv41_rdma_stage_wait_x(const float4* __restrict__ src, int n4,
        char* send_ring, uint64_t* sizes, uint64_t* doorbell, const CqTab cq, uint64_t* seq, int slots,
        uint64_t max_bytes, uint64_t* probe, const uint64_t* abort_word, unsigned* staged, int fence_mode,
        float4* __restrict__ own_dst, int own_row4, long long own_stride4, int pdl) {
    __shared__ uint64_t s_seq;
    __shared__ int s_last;
    pdl_enter(pdl);
    // TF_DS_ROCI_XLOG: the host tags `probe` with bit 0 and points it at the doorbell log (8,192 records of {sequence, this GPU's globaltimer after the
    // doorbell}); no probe then (it is null in a served engine) (xlog)
    uint64_t* xlog_pub = (reinterpret_cast<uintptr_t>(probe) & 1) ? reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(probe) & ~uintptr_t(1)) : nullptr;
    if (xlog_pub) probe = nullptr;
    const uint64_t t0 = probe ? now_ns() : 0;     // (the probe is null in a served engine)
    if (threadIdx.x < 32) {                      // warp 0: its lanes read the queue pairs' completions together
        const uint64_t q = *seq + 1;
        if (q > (uint64_t)slots) cq_wait_warp(cq, q - slots - 1, abort_word);
        if (threadIdx.x == 0) s_seq = q;
    }
    __syncthreads();
    const uint64_t q = s_seq;
    const int slot = (int)(q % slots);
    float4* dst = reinterpret_cast<float4*>(send_ring + (size_t)slot * max_bytes);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
    if (fence_mode) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
        }
    } else {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(staged, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_last) {
        __threadfence_system();
        *staged = 0;
        *seq = q;
        if (fence_mode) {
            store_relaxed_sys(doorbell, bswap64_dev(q));
        } else {
            store_sys(doorbell, bswap64_dev(q));
        }
        if (xlog_pub) {
            xlog_pub[(q & 8191) * 2] = q;
            xlog_pub[(q & 8191) * 2 + 1] = now_ns();
        }  // xlog
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
    if (own_dst) {
        if (s_last) __syncthreads();
        if (own_stride4 == own_row4) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) own_dst[i] = src[i];
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = src[i];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(STAGE_THREADS) dsv41_rdma_publish_wait_x(uint64_t* sizes, uint64_t* doorbell,
        uint64_t* seq, uint64_t* send_q, uint64_t* send_slot, uint64_t* send_last, int slots, int user,
        const CqTab cq, const uint64_t* abort_word, const float4* __restrict__ own_src,
        float4* __restrict__ own_dst, int own_row4, long long own_stride4, uint64_t nbytes, uint64_t* probe, int pdl) {
    pdl_enter(pdl);
    // TF_DS_ROCI_XLOG: the host tags `probe` with bit 0 and points it at the doorbell log (8,192 records of {sequence, this GPU's globaltimer after the
    // doorbell}); no probe then (it is null in a served engine) (xlog)
    uint64_t* xlog_pub = (reinterpret_cast<uintptr_t>(probe) & 1) ? reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(probe) & ~uintptr_t(1)) : nullptr;
    if (xlog_pub) probe = nullptr;
    const uint64_t t0 = probe ? now_ns() : 0;     // (the probe is null in a served engine)
    const uint64_t q = *seq + 1;
    if (blockIdx.x == 0 && threadIdx.x < 32) {   // warp 0 of block 0
        if (q > (uint64_t)slots) cq_wait_warp(cq, q - slots - 1, abort_word);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // (sizes[], send_q[] and send_slot[] are the proxy path's words: with the queue pairs waiting in the card the chain carries the size
        // and the slot, and nothing reads them; send_last is read by dsv41_rdma_await_send_wait)
        store_relaxed_sys(send_last + user, q);
        *seq = q;
        __threadfence_system();
        store_relaxed_sys(doorbell, bswap64_dev(q));
        if (xlog_pub) {
            xlog_pub[(q & 8191) * 2] = q;
            xlog_pub[(q & 8191) * 2 + 1] = now_ns();
        }  // xlog
        if (probe) {
            probe[0] += now_ns() - t0;
            probe[4] = now_ns();
        }
    }
    if (own_dst) {
        const int n4 = (int)(nbytes / 16);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
            if (own_stride4 == own_row4) {
                own_dst[i] = own_src[i];
            } else {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = own_src[i];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(COLLECT_THREADS) dsv41_rdma_collect_peers_roci_x(Dsts d, int rows,
        const char* recv_ring, const uint64_t* gen, const unsigned* lanes, const uint64_t* seq, int rank, int world,
        int slots, uint64_t max_bytes, uint64_t* abort_word, uint64_t deadline_ns, uint64_t* fault, int pdl,
        const float4* __restrict__ own_src, float4* __restrict__ own_dst, int own_row4, long long own_stride4, int own_n4) {
    pdl_enter(pdl);                              // the staging kernel's sequence and own slice are written
    // TF_DS_ROCI_XLOG: the host tags the fault buffer's address with bit 0; the exchange log lives behind the record's 8 words
    const bool xlog = (reinterpret_cast<uintptr_t>(fault) & 1) != 0;
    fault = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(fault) & ~uintptr_t(1));  // xlog
    const uint64_t t_enter = xlog ? now_ns() : 0;   // the kernel's start on this GPU's clock (after the PDL wait)
    // This rank's own slice, copied here (the first row of blocks) while the peers' words are still on their way: the publishing kernel
    // does not copy it, so its doorbell is not behind the copy. The same bytes to the same words; the consumer reads them after this kernel.
    if (own_dst != nullptr && blockIdx.y == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < own_n4; i += gridDim.x * blockDim.x) {
            if (own_stride4 == own_row4) {
                own_dst[i] = own_src[i];
            } else {
                const int r = i / own_row4;
                own_dst[(size_t)r * own_stride4 + (i - r * own_row4)] = own_src[i];
            }
        }
    }
    int p = -1;
    for (int k = 0, y = (int)blockIdx.y; k < world; ++k)
        if (k != rank && d.ptr[k] != nullptr && d.row4[k] != 0 && y-- == 0) {
            p = k;
            break;
        }
    if (p < 0) return;                           // no peer for this row of blocks (a grid of one with none to copy)
    const uint64_t q = *seq;
    // everything that does not depend on the wait is computed before it (the 64-bit modulo and the row division are library routines of ~80
    // and ~40 instructions): the wait then ends in the copy-out's loads
    const int slot = (int)(q % slots);
    const float4* in = reinterpret_cast<const float4*>(recv_ring + ((size_t)slot * world + p) * max_bytes);
    float4* out = d.ptr[p];
    const int r4 = d.row4[p];
    const long long s4 = d.stride4[p];
    const int per_row = (r4 + PEER_UNIT4 - 1) / PEER_UNIT4;
    // a failed wait is the block's: the poison ends it (the vote below)
    int failed = 0;
    const int nl = (int)lanes[p];
    if (threadIdx.x < nl) {                      // the peer's lanes are polled together, one thread a lane (the peer's words are independent)
        const int k = (int)threadIdx.x;
        uint64_t t0 = 0;
        uint64_t seen = 0;
        for (uint32_t i = 1; (seen = load_relaxed_sys(gen + (size_t)p * MAX_LANES + k)) < q; ++i) {
            if (i == 1) t0 = now_ns();               // the word was not there yet: the wait's start
            if ((i & 1023) == 0) {
                if (load_sys(abort_word) != 0) {
                    roci_record(fault, abort_word, q, rank, p, k, seen, now_ns() - t0, 1);
                    failed = 1;
                    break;
                }
                if (deadline_ns != 0 && now_ns() - t0 >= deadline_ns) {
                    roci_record(fault, abort_word, q, rank, p, k, seen, now_ns() - t0, 2);
                    failed = 1;
                    break;
                }
            }
        }
        if (!failed) seen = load_sys(gen + (size_t)p * MAX_LANES + k);   // the acquire edge, once
    }
    if (__syncthreads_or(failed) != 0) return;     // a failed wait is the block's: the vote, not a load of the abort word by every thread
    if (xlog && threadIdx.x == 0) {                // log only: exchange q's words from peer p were all in at this time (this GPU's globaltimer)
        uint64_t* e = fault + 8 + ((q * 8 + (uint64_t)p) & 65535) * 4;   // 4 words a record: key, words in, kernel start, a block's copy-out end  // xlog
        e[0] = (q << 3) | (uint64_t)p;  // xlog
        e[1] = now_ns();  // xlog
        e[2] = t_enter;  // xlog
    }  // xlog
    for (int u = blockIdx.x; u < rows * per_row; u += gridDim.x) {
        const int r = per_row == 1 ? u : u / per_row;     // (a row of up to 1,024 float4 is one unit: every decode exchange; no division then)
        const int c0 = (per_row == 1 ? 0 : (u - r * per_row) * PEER_UNIT4) + threadIdx.x;
        const float4* src = in + (size_t)r * r4;
        float4* dst = out + (size_t)r * s4;
        float4 v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (c0 + k * COLLECT_THREADS < r4) v[k] = __ldcv(src + c0 + k * COLLECT_THREADS);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (c0 + k * COLLECT_THREADS < r4) dst[c0 + k * COLLECT_THREADS] = v[k];
    }
    if (xlog) {                                    // log only (a block-uniform branch): this block's copy-out has ended; the last block to write is the record's end
        __syncthreads();  // xlog
        if (threadIdx.x == 0) fault[8 + ((q * 8 + (uint64_t)p) & 65535) * 4 + 3] = now_ns();  // xlog
    }  // xlog
}
