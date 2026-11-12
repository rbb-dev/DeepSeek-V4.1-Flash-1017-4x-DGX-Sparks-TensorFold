// How long a GPU spin on a host-memory word takes to notice a store, per poll variant (a microbenchmark for ONE GB10, not
// part of the engine). A writer kernel on one stream stores the next sequence into a cudaHostAlloc'd word at a time it
// reads from %globaltimer; a poller kernel on another stream spins on the word and reads %globaltimer when it leaves; both
// timers are the GPU's own, so t_seen - t_write is the detection latency (the store's trip to host memory + the poll's
// load trip + the loop's own cost). The variants:
//   acq      ld.acquire.sys every iteration (what dsv41_rdma_collect* do: LDG.E.64.STRONG.SYS + CCTL.IVALL a turn)
//   rlx_acq  ld.relaxed.sys every iteration, one ld.acquire.sys after the loop leaves (the L1 invalidate once)
//   rlx_fence ld.relaxed.sys every iteration, then fence.acq_rel.sys
//   wide     4 pollers (warps) on the word, each starting a quarter of a load round trip later, the first to see leaves
//   nvcc -O3 -arch=sm_121 -o poll_latency poll_latency.cu && ./poll_latency [rounds]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

__device__ __forceinline__ uint64_t now_ns() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t ld_acq(const uint64_t* p) { uint64_t v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint64_t ld_rlx(const uint64_t* p) { uint64_t v; asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }

// the writer: waits until `go` (device flag the host sets after the poller is spinning), sleeps `delay_ns`, stores q
__global__ void writer(volatile uint64_t* word, volatile int* ready, uint64_t q, uint64_t delay_ns, uint64_t* t_write) {
    const uint64_t tw = now_ns();
    for (uint32_t i = 1; *ready == 0; ++i)
        if ((i & 1023) == 0 && now_ns() - tw > 2000000000ull) { *t_write = 1; return; }  // (never hang)
    const uint64_t t0 = now_ns();
    while (now_ns() - t0 < delay_ns) {}
    *t_write = now_ns();
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(word), "l"(q) : "memory");
}
template <int V>
__global__ void poller(const uint64_t* word, volatile int* ready, uint64_t q, uint64_t* t_seen) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && V != 3) *ready = 1;  // (single poller: it announces itself)
    if (V == 3) {
        // 4 warps' lane 0 on the word; warp w delays its first load by w * 150 ns; the first to see leaves, writing the time
        const int w = threadIdx.x >> 5;
        if ((threadIdx.x & 31) != 0) return;
        if (w == 0) *ready = 1;
        const uint64_t t0 = now_ns();
        while (now_ns() - t0 < (uint64_t)w * 150) {}
        for (uint32_t i = 1; ld_rlx(word) < q; ++i) {
            if (*(volatile uint64_t*)t_seen != 0) return;
            if ((i & 1023) == 0 && now_ns() - t0 > 2000000000ull) return;
        }
        const uint64_t t = now_ns();
        atomicCAS((unsigned long long*)t_seen, 0ull, (unsigned long long)t);
        return;
    }
    uint64_t s = 0;
    const uint64_t tp = now_ns();
    // (the production loops check the clock every 1,024 turns; so do these, to bound the wait)
    if (V == 0) { for (uint32_t i = 1; (s = ld_acq(word)) < q; ++i) if ((i & 1023) == 0 && now_ns() - tp > 2000000000ull) return; }
    if (V == 1) { for (uint32_t i = 1; (s = ld_rlx(word)) < q; ++i) if ((i & 1023) == 0 && now_ns() - tp > 2000000000ull) return; s = ld_acq(word); }
    if (V == 2) { for (uint32_t i = 1; (s = ld_rlx(word)) < q; ++i) if ((i & 1023) == 0 && now_ns() - tp > 2000000000ull) return; asm volatile("fence.acq_rel.sys;" ::: "memory"); }
    t_seen[0] = now_ns() + (s == 0);
}

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 400;
    uint64_t *word, *t_write, *t_seen;
    int* ready;
    CK(cudaHostAlloc((void**)&word, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**)&ready, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**)&t_write, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**)&t_seen, 64, cudaHostAllocMapped));
    cudaStream_t a, b;
    CK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    const char* names[4] = {"acq", "rlx_acq", "rlx_fence", "wide(4)"};
    for (int v = 0; v < 4; v++) {
        std::vector<double> lat;
        for (int r = 1; r <= rounds + 20; r++) {
            *word = 0; *ready = 0; *t_write = 0; *t_seen = 0;
            __sync_synchronize();
            const uint64_t q = r;
            const uint64_t delay = 20000 + (r * 7919) % 13000;  // 20-33 us after the poller is up: not aligned to the poll period
            writer<<<1, 1, 0, b>>>(word, ready, q, delay, t_write);
            if (v == 0) poller<0><<<1, 32, 0, a>>>(word, ready, q, t_seen);
            if (v == 1) poller<1><<<1, 32, 0, a>>>(word, ready, q, t_seen);
            if (v == 2) poller<2><<<1, 32, 0, a>>>(word, ready, q, t_seen);
            if (v == 3) poller<3><<<1, 128, 0, a>>>(word, ready, q, t_seen);
            CK(cudaStreamSynchronize(a));
            CK(cudaStreamSynchronize(b));
            if (*t_seen == 0 || *t_write <= 1) { fprintf(stderr, "%s round %d: no store/poll completed (the kernels did not run together)\n", names[v], r); return 1; }
            if (r > 20) lat.push_back(double(int64_t(*t_seen) - int64_t(*t_write)) / 1000.0);
        }
        std::sort(lat.begin(), lat.end());
        printf("%-10s detection latency us: p10 %.2f  p50 %.2f  p90 %.2f  p99 %.2f  (n=%zu)\n", names[v], lat[lat.size() / 10], lat[lat.size() / 2], lat[lat.size() * 9 / 10], lat[lat.size() * 99 / 100], lat.size());
    }
    return 0;
}
