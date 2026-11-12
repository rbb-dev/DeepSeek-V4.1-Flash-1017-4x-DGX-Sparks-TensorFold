// A prompt chunk's FP8 GEMMs on the tensor core's block-scaled MMA (mma.sync ... kind::mxf8f6f4.block_scale, QMMA.SF;
// TF_DS_MMA_SF, mx.zig useMmaSf): the dense FP8 GEMMs of native.cu's tile kernels and the grouped experts' gate / up and
// down (MXFP4 routed experts as E4M3 x E2M1, the FP8 shared expert as E4M3 x E4M3), each 32-wide K block's sum scaled
// by the activation row's and the weight's UE8M0 scales inside the MMA. Two arithmetics (the kernels' MODE):
//   - chained (TF_DS_MMA_SF=1): a split's K blocks accumulate in the MMA's C fragment, so each block's scaled sum
//     meets the running sum in the tensor core's own addition (aligned to the larger exponent and truncated) instead
//     of an fp32 add: NOT the bits of native.cu's kernels; the splits' partials are added in split order in fp32;
//   - exact: each K block's MMA from zero, its scaled sum then added to the fp32 sum, K blocks in order: native.cu's
//     bits whenever every scaled block sum is a normal float (the MMA scales its sum by a power of two exactly, and
//     native.cu's fmaf adds that same exact product).
// Within a K block the MMA's result does not depend on the order of its 32 products, so the MXFP4 tiles take each
// thread's eight values in the order its packed bytes give them cheaply (even K, then odd), A's bytes alike.
// A row's result never depends on the other rows of a launch, and a GEMM's K split is native.cu's (mx.zig splits), so
// a row has the same bits in every launch of either arithmetic: the split tiles fold the split in the CTA, and at a
// small chunk's rows one CTA a split writes its partial for tf_dsn_reduce, the same adds in the same order.
// Built for sm_121a alone, its own fatbin: native.cu is included for its shapes, staging and epilogues, and every one
// of its kernels keeps its own sm_121 build. The L2 eviction hints stay out of it.
#include "native.cu"

#if PF_HINT
#error "native_sf.cu: the tiles' L2 eviction hints stay out of an sm_121a build"
#endif

namespace sf {

using pf::Cfg;
using pf::kThreads;  // the 8-warp R2 kernels' block (the W16 tiles name theirs: C::THREADS)
using pf::Ops;

constexpr int kChained = 0;  // the switch's arithmetic: a split's K blocks accumulate in the MMA
constexpr int kExact = 1;    // each K block's MMA from zero, then an fp32 add: native.cu's bits

// d (+)= the block-scaled m16n8k32 MMA of A (E4M3) and B (E4M3; FP4: E2M1, the code at bits 5..2 of each byte), each
// operand's K block scaled by its UE8M0 byte, byte KK of the thread's scale words. Thread-id 0 on both sides: row g's
// SF_A comes from lane 4g, row g + 8's from lane 4g + 1, column g's SF_B from lane 4g.
template <int KK, bool FP4>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sa, uint32_t sb) {
    if constexpr (FP4)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%13}, %11, {%12,%13};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "n"(KK), "n"(0));
    else
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%13}, %11, {%12,%13};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb), "n"(KK), "n"(0));
}

// a stage's activation scales of one row (KBS fp32 powers of two, act_quant's: their exponent bits are the UE8M0 byte),
// K block kk in byte kk
template <int KBS>
__device__ __forceinline__ uint32_t act_word(const float* p) {
    if constexpr (KBS == 4) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        return (v.x >> 23) | (v.y >> 23) << 8 | (v.z >> 23) << 16 | (v.w >> 23) << 24;
    } else {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        return (v.x >> 23) | (v.y >> 23) << 8;
    }
}

// a stage's FP8 weight scales of one 32 x 32 column block: KBS bytes of pf::fp8_scales' slab from the stage's first K
// block (the slab's rows hold an even count of K blocks and a stage starts at an even one, so 2-byte loads)
template <int KBS>
__device__ __forceinline__ uint32_t fp8_word(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    if constexpr (KBS == 4) return uint32_t(h[0]) | uint32_t(h[1]) << 16;
    else return h[0];
}

template <int MT, int NT>
__device__ __forceinline__ void zero(float (&x)[MT][NT][4]) {
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) x[i][j][c] = 0.f;
}

// one K block of weight wi's MMAs for a warp's tile: chained, each into part (the MMA's C); exact, an n8 column's MMAs
// from zero, then each added to part
template <class C, int MODE, bool FULL, int KK, bool FP4>
__device__ __forceinline__ void block(float (&part)[C::MT][C::NT][4], const uint32_t (&af)[C::MT][4], const uint32_t (&b)[C::NT][2],
                                      const uint32_t (&saw)[C::MT], const uint32_t (&sbw)[C::NT], int wm, int mvalid) {
#pragma unroll
    for (int j = 0; j < C::NT; j++) {
        if constexpr (MODE == kChained) {
#pragma unroll
            for (int i = 0; i < C::MT; i++) {
                if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                mma<KK, FP4>(part[i][j], af[i], b[j], saw[i], sbw[j]);
            }
        } else {
            float d[C::MT][4];
#pragma unroll
            for (int i = 0; i < C::MT; i++) {
                if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
#pragma unroll
                for (int c = 0; c < 4; c++) d[i][c] = 0.f;
                mma<KK, FP4>(d[i], af[i], b[j], saw[i], sbw[j]);
            }
#pragma unroll
            for (int i = 0; i < C::MT; i++) {
                if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
#pragma unroll
                for (int c = 0; c < 4; c++) part[i][j][c] += d[i][c];
            }
        }
    }
}

// one K block (KK of the stage) of an FP8 tile staged by pf::load_stage: the fragments in native.cu's K order
template <class C, int MODE, bool FULL, int KK>
__device__ __forceinline__ void fp8_block(const char* A, const char* B, const uint32_t (&saw)[C::MT], const uint32_t (&sbw)[C::NWT][C::NT],
                                          int wm, int wn, int g, int t, int mvalid, float (&part)[C::NWT][C::MT][C::NT][4]) {
    uint32_t af[C::MT][4];
    if constexpr (C::LDSM) {
        // ldmatrix hands a thread, from each of four 8 x 16-byte matrices, the 4 bytes at 4t of its row g: A's four
        // registers are (row g, row g + 8) x (piece 2 KK, piece 2 KK + 1), the MMA's natural K order (K 4t .. 4t + 3 and
        // 16 + 4t ..), with no register moves. B takes the same pieces, so each product keeps its partner.
        const int lane = g * 4 + t;
        const int lr = (lane & 7) + ((lane >> 3) & 1) * 8, lc = KK * 2 + (lane >> 4);
#pragma unroll
        for (int i = 0; i < C::MT; i++) {
            const int r = wm * C::WM + i * 16 + lr;
            pf::ldsm4(af[i], A + r * C::SA + C::pa(r, lc) * 16);
        }
    } else {
#pragma unroll
        for (int i = 0; i < C::MT; i++) {
            const int r = wm * C::WM + i * 16 + g;
            const int pc = KK * 2 + (t >> 1), po = (t & 1) * 8;
            const uint2 lo = *reinterpret_cast<const uint2*>(A + r * C::SA + C::pa(r, pc) * 16 + po);
            const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * C::SA + C::pa(r + 8, pc) * 16 + po);
            af[i][0] = lo.x;
            af[i][1] = hi.x;
            af[i][2] = lo.y;
            af[i][3] = hi.y;
        }
    }
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++) {
        uint32_t b[C::NT][2];
        if constexpr (C::LDSM) {
            const int lane = g * 4 + t;
#pragma unroll
            for (int jj = 0; jj < C::NT / 2; jj++) {
                // matrices: n8 tile 2 jj's pieces 2 KK and 2 KK + 1, then tile 2 jj + 1's
                const int br = wi * C::BN + wn * C::WN + (2 * jj + ((lane >> 4) & 1)) * 8 + (lane & 7);
                uint32_t r4[4];
                pf::ldsm4(r4, B + br * C::SB + C::pb(br, KK * 2 + ((lane >> 3) & 1)) * 16);
                b[2 * jj][0] = r4[0];
                b[2 * jj][1] = r4[1];
                b[2 * jj + 1][0] = r4[2];
                b[2 * jj + 1][1] = r4[3];
            }
        } else {
#pragma unroll
            for (int j = 0; j < C::NT; j++) {
                const int br = wi * C::BN + wn * C::WN + j * 8 + g;
                const uint2 v = *reinterpret_cast<const uint2*>(B + br * C::SB + C::pb(br, KK * 2 + (t >> 1)) * 16 + (t & 1) * 8);
                b[j][0] = v.x;
                b[j][1] = v.y;
            }
        }
        block<C, MODE, FULL, KK, false>(part[wi], af, b, saw, sbw[wi], wm, mvalid);
    }
}

// stage st of an FP8 tile's ring (K blocks kb_lo + st * KBS ..): its wait, the load of stage st + STAGES - 1, then its
// K blocks' MMAs into part
template <class C, int MODE, bool FULL, bool TWO = false, bool THREE = false>
__device__ __forceinline__ void fp8_stage(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb, int kb_lo, int nst, int st,
                                          int mvalid, float (&part)[C::NWT][C::MT][C::NT][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    pf::wait_groups<C::STAGES - 2>();
    __syncthreads();
    {
        const int nx = st + C::STAGES - 1;
        if (nx < nst) pf::load_stage<C, TWO, THREE>(smem + (nx % C::STAGES) * C::STAGE, srow, o, kb_lo + nx * C::KBS);
        pf::commit();
    }
    const char* A = smem + (st % C::STAGES) * C::STAGE;
    const char* B = A + C::A_BYTES;
    const float* AS = reinterpret_cast<const float*>(B + C::B_BYTES);
    const int kb0 = kb_lo + st * C::KBS;
    // this thread's scale words: its SF_A row (g of each m16 tile, g + 8 for lanes 4g + 1 and 4g + 3) and its n8
    // tiles' 32-column blocks
    uint32_t saw[C::MT];
#pragma unroll
    for (int i = 0; i < C::MT; i++) saw[i] = act_word<C::KBS>(AS + (wm * C::WM + i * 16 + g + 8 * (t & 1)) * C::KBS);
    uint32_t sbw[C::NWT][C::NT];
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int j = 0; j < C::NT; j++) sbw[wi][j] = fp8_word<C::KBS>(wss + (wi * (C::BN / kBlk) + (wn * C::WN + j * 8) / kBlk) * nkb + kb0);
    fp8_block<C, MODE, FULL, 0>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
    fp8_block<C, MODE, FULL, 1>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
    if constexpr (C::KBS == 4) {
        fp8_block<C, MODE, FULL, 2>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
        fp8_block<C, MODE, FULL, 3>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
    }
}

// The K loop of a tile of FP8 weights (the dense GEMMs, the FP8 shared expert) staged by pf::load_stage: acc as
// pf::run_tile1's, over K blocks [kb_lo, kb_hi) (multiples of KBS), MODE's arithmetic. SPLIT: in splits of `fold` K
// blocks from kb_lo (a multiple of KBS), each split's partial from zero, then acc = the first split's partial, then
// acc + each later one (tf_dsn_reduce's order); else the range is one split, summed into acc. mvalid: the tile's rows
// from mvalid on are padding (their m16 tiles skipped). wss: pf::fp8_scales' slab of the tile's weight scales, nkb K
// blocks a row.
template <class C, int MODE, bool SPLIT, bool FULL, bool TWO = false, bool THREE = false>
__device__ __forceinline__ void tile_fp8_1(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb, int kb_lo, int kb_hi,
                                           int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    static_assert(C::FMT == 0 && (C::KBS == 2 || C::KBS == 4), "FP8 weights, 2 or 4 K blocks a stage");
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++) zero(acc[wi]);
    const int nst = (kb_hi - kb_lo) / C::KBS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) pf::load_stage<C, TWO, THREE>(smem + s * C::STAGE, srow, o, kb_lo + s * C::KBS);
        pf::commit();
    }
    if constexpr (!SPLIT) {
        for (int st = 0; st < nst; st++) fp8_stage<C, MODE, FULL, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, nst, st, mvalid, acc);
    } else {
        const int sps = fold / C::KBS;  // stages a split
        int st = 0;
        for (int split = 0; st < nst; split++) {
            float part[C::NWT][C::MT][C::NT][4];
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++) zero(part[wi]);
            const int st_end = min(nst, st + sps);
            for (; st < st_end; st++) fp8_stage<C, MODE, FULL, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, nst, st, mvalid, part);
            // the split's end: its partial into the sum (tf_dsn_reduce: z[0], then + z[1], ...)
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                for (int i = 0; i < C::MT; i++)
#pragma unroll
                    for (int j = 0; j < C::NT; j++)
#pragma unroll
                        for (int c = 0; c < 4; c++) acc[wi][i][j][c] = split == 0 ? part[wi][i][j][c] : acc[wi][i][j][c] + part[wi][i][j][c];
        }
    }
    pf::wait_groups<0>();
}

template <class C, int MODE, bool SPLIT, bool TWO = false, bool THREE = false>
__device__ __forceinline__ void tile_fp8(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb, int kb_lo, int kb_hi, int fold,
                                         int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    if (mvalid >= C::BM) tile_fp8_1<C, MODE, SPLIT, true, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, kb_hi, fold, mvalid, acc);
    else tile_fp8_1<C, MODE, SPLIT, false, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, kb_hi, fold, mvalid, acc);
}

// one K block (KK of the stage) of an MXFP4 tile staged by pf::W2 (the pair slot's half `half`): A's eight bytes of a
// row and B's eight codes of a column taken even K first, then odd (the MMA's sum does not depend on the order)
template <class C, int MODE, bool FULL, int KK>
__device__ __forceinline__ void fp4_block(const char* A, const char* B, int half, const uint32_t (&saw)[C::MT], const uint32_t (&sbw)[C::NWT][C::NT],
                                          int wm, int wn, int g, int t, int mvalid, float (&part)[C::NWT][C::MT][C::NT][4]) {
    using P = pf::W2<C>;
    uint32_t af[C::MT][4];
#pragma unroll
    for (int i = 0; i < C::MT; i++) {
        const int r = wm * C::WM + i * 16 + g;
        const int pc = KK * 2 + (t >> 1), po = (t & 1) * 8;
        const uint2 lo = *reinterpret_cast<const uint2*>(A + r * C::SA + C::pa(r, pc) * 16 + po);
        const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * C::SA + C::pa(r + 8, pc) * 16 + po);
        af[i][0] = __byte_perm(lo.x, lo.y, 0x6420);  // row g, K 8t, 8t + 2, 8t + 4, 8t + 6
        af[i][1] = __byte_perm(hi.x, hi.y, 0x6420);  // row g + 8
        af[i][2] = __byte_perm(lo.x, lo.y, 0x7531);  // row g, K 8t + 1, 8t + 3, 8t + 5, 8t + 7
        af[i][3] = __byte_perm(hi.x, hi.y, 0x7531);
    }
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++) {
        uint32_t b[C::NT][2];
#pragma unroll
        for (int j = 0; j < C::NT; j++) {
            const int br = wi * C::BN + wn * C::WN + j * 8 + g;
            // eight codes, K 8t .. 8t + 7, two a byte (low nibble first): the low nibbles (even K) and the high ones
            // (odd K) each into the E2M1 containers' bits 5..2
            const uint32_t v = *reinterpret_cast<const uint32_t*>(B + br * P::ROW + P::piece(br, half * 4 + KK) * 16 + t * 4);
            b[j][0] = (v << 2) & 0x3c3c3c3cu;
            b[j][1] = (v >> 2) & 0x3c3c3c3cu;
        }
        block<C, MODE, FULL, KK, true>(part[wi], af, b, saw, sbw[wi], wm, mvalid);
    }
}

// The K loop of a tile of MXFP4 weights staged by pf::W2 (each weight row 128 bytes a slot, pf::run_tile_w128_1's
// ring and loads): acc as pf::run_tile1's, the K blocks in order (MODE's arithmetic).
template <class C, int MODE, bool FULL, bool THREE = false>
__device__ __forceinline__ void tile_fp4_1(char* smem, const int* srow, const Ops& o, int nkb, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    using P = pf::W2<C>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    char* aring = smem;
    char* bring = smem + C::STAGES * P::A_SLOT;
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++) zero(acc[wi]);
    const int nst = nkb / C::KBS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) {
            pf::load_a<C, THREE>(aring + s * P::A_SLOT, srow, o, s * C::KBS);
            if ((s & 1) == 0) pf::load_b2<C>(bring + ((s >> 1) & 1) * P::SLOT, o, s * C::KBS, nkb);
        }
        pf::commit();
    }
    for (int st = 0; st < nst; st++) {
        pf::wait_groups<C::STAGES - 2>();
        __syncthreads();
        {
            const int nx = st + C::STAGES - 1;
            if (nx < nst) {
                pf::load_a<C, THREE>(aring + (nx % C::STAGES) * P::A_SLOT, srow, o, nx * C::KBS);
                if ((nx & 1) == 0) pf::load_b2<C>(bring + ((nx >> 1) & 1) * P::SLOT, o, nx * C::KBS, nkb);
            }
            pf::commit();
        }
        const char* A = aring + (st % C::STAGES) * P::A_SLOT;
        const float* AS = reinterpret_cast<const float*>(A + C::A_BYTES);
        const char* B = bring + ((st >> 1) & 1) * P::SLOT;
        const uint32_t* WS = reinterpret_cast<const uint32_t*>(B + P::B_BYTES);
        const int half = st & 1;
        uint32_t saw[C::MT];
#pragma unroll
        for (int i = 0; i < C::MT; i++) saw[i] = act_word<C::KBS>(AS + (wm * C::WM + i * 16 + g + 8 * (t & 1)) * C::KBS);
        // column g's word of this stage's scales (the B fragment's column, not the C fragment's)
        uint32_t sbw[C::NWT][C::NT];
#pragma unroll
        for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
            for (int j = 0; j < C::NT; j++) sbw[wi][j] = WS[(wi * C::BN + wn * C::WN + j * 8 + g) * 2 + half];
        fp4_block<C, MODE, FULL, 0>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
        fp4_block<C, MODE, FULL, 1>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
        fp4_block<C, MODE, FULL, 2>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
        fp4_block<C, MODE, FULL, 3>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
    }
    pf::wait_groups<0>();
}

template <class C, int MODE, bool THREE = false>
__device__ __forceinline__ void tile_fp4(char* smem, const int* srow, const Ops& o, int nkb, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    if (mvalid >= C::BM) tile_fp4_1<C, MODE, true, THREE>(smem, srow, o, nkb, mvalid, acc);
    else tile_fp4_1<C, MODE, false, THREE>(smem, srow, o, nkb, mvalid, acc);
}

// TF_DS_GEMM_R2's activation staging: pf::load_a / pf::load_stage with a padding row's copies left out. A stored
// output row's sum reads only its own A row and its own scale word, and the epilogues store only rows below the tile's
// valid count, so a padding row's slot feeds no stored output and its copy (which pf::load_a would zero-fill) is
// skipped. The weight staging, the fragment loads and every MMA are the baseline.
template <class C>
__device__ __forceinline__ void load_a_skip(char* st, const int* srow, const Ops& o, int kb0) {
    const int tid = threadIdx.x;
    char* A = st;
    char* AS = A + C::A_BYTES;
    constexpr int CPA = C::KBS * 2;
    for (int c = tid; c < C::BM * CPA; c += kThreads) {
        const int r = c / CPA, cc = c % CPA;
        const int row = srow[r];
        if (row < 0) continue;
        pf::cp16(A + r * C::SA + C::pa(r, cc) * 16, o.q + size_t(row) * o.ldq + kb0 * 32 + cc * 16, true);
    }
    for (int r = tid; r < C::BM; r += kThreads) {
        const int row = srow[r];
        if (row < 0) continue;
        const float* src = o.s + size_t(row) * o.lds + kb0;
        if (C::KBS == 4) pf::cp16(AS + r * 16, src, true);
        else pf::cp8(AS + r * 8, src, true);
    }
}

// pf::load_stage with the activation rows and scales staged as there, a padding row's copies left out
template <class C>
__device__ __forceinline__ void load_stage_skip(char* st, const int* srow, const Ops& o, int kb0) {
    const int tid = threadIdx.x;
    char* A = st;
    char* B = A + C::A_BYTES;
    char* AS = B + C::B_BYTES;
    constexpr int CPA = C::KBS * 2;
    for (int c = tid; c < C::BM * CPA; c += kThreads) {
        const int r = c / CPA, cc = c % CPA;
        const int row = srow[r];
        if (row < 0) continue;
        pf::cp16(A + r * C::SA + C::pa(r, cc) * 16, o.q + size_t(row) * o.ldq + kb0 * 32 + cc * 16, true);
    }
    for (int r = tid; r < C::BM; r += kThreads) {
        const int row = srow[r];
        if (row < 0) continue;
        const float* src = o.s + size_t(row) * o.lds + kb0;
        if (C::KBS == 4) pf::cp16(AS + r * 16, src, true);
        else pf::cp8(AS + r * 8, src, true);
    }
    constexpr int CPB = C::KBS * 2;
    for (int c = tid; c < C::NWT * C::BN * CPB; c += kThreads) {
        const int r = c / CPB, cc = c % CPB;
        const int wi = r / C::BN, col = o.n0 + r % C::BN;
        pf::cp16(B + r * C::SB + C::pb(r, cc) * 16, o.w(wi) + size_t(col) * o.ldw + kb0 * 32 + cc * 16, true);
    }
}

// fp8_stage with load_stage_skip
template <class C, int MODE, bool FULL>
__device__ __forceinline__ void fp8_stage_skip(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb, int kb_lo, int nst, int st,
                                               int mvalid, float (&part)[C::NWT][C::MT][C::NT][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    pf::wait_groups<C::STAGES - 2>();
    __syncthreads();
    {
        const int nx = st + C::STAGES - 1;
        if (nx < nst) load_stage_skip<C>(smem + (nx % C::STAGES) * C::STAGE, srow, o, kb_lo + nx * C::KBS);
        pf::commit();
    }
    const char* A = smem + (st % C::STAGES) * C::STAGE;
    const char* B = A + C::A_BYTES;
    const float* AS = reinterpret_cast<const float*>(B + C::B_BYTES);
    const int kb0 = kb_lo + st * C::KBS;
    uint32_t saw[C::MT];
#pragma unroll
    for (int i = 0; i < C::MT; i++) saw[i] = act_word<C::KBS>(AS + (wm * C::WM + i * 16 + g + 8 * (t & 1)) * C::KBS);
    uint32_t sbw[C::NWT][C::NT];
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int j = 0; j < C::NT; j++) sbw[wi][j] = fp8_word<C::KBS>(wss + (wi * (C::BN / kBlk) + (wn * C::WN + j * 8) / kBlk) * nkb + kb0);
    fp8_block<C, MODE, FULL, 0>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
    fp8_block<C, MODE, FULL, 1>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
    if constexpr (C::KBS == 4) {
        fp8_block<C, MODE, FULL, 2>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
        fp8_block<C, MODE, FULL, 3>(A, B, saw, sbw, wm, wn, g, t, mvalid, part);
    }
}

template <class C, int MODE, bool SPLIT, bool FULL>
__device__ __forceinline__ void tile_fp8_skip_1(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb, int kb_lo, int kb_hi,
                                                int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    static_assert(C::FMT == 0 && (C::KBS == 2 || C::KBS == 4), "FP8 weights, 2 or 4 K blocks a stage");
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++) zero(acc[wi]);
    const int nst = (kb_hi - kb_lo) / C::KBS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) load_stage_skip<C>(smem + s * C::STAGE, srow, o, kb_lo + s * C::KBS);
        pf::commit();
    }
    if constexpr (!SPLIT) {
        for (int st = 0; st < nst; st++) fp8_stage_skip<C, MODE, FULL>(smem, srow, wss, o, nkb, kb_lo, nst, st, mvalid, acc);
    } else {
        const int sps = fold / C::KBS;  // stages a split
        int st = 0;
        for (int split = 0; st < nst; split++) {
            float part[C::NWT][C::MT][C::NT][4];
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++) zero(part[wi]);
            const int st_end = min(nst, st + sps);
            for (; st < st_end; st++) fp8_stage_skip<C, MODE, FULL>(smem, srow, wss, o, nkb, kb_lo, nst, st, mvalid, part);
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                for (int i = 0; i < C::MT; i++)
#pragma unroll
                    for (int j = 0; j < C::NT; j++)
#pragma unroll
                        for (int c = 0; c < 4; c++) acc[wi][i][j][c] = split == 0 ? part[wi][i][j][c] : acc[wi][i][j][c] + part[wi][i][j][c];
        }
    }
    pf::wait_groups<0>();
}

template <class C, int MODE, bool SPLIT>
__device__ __forceinline__ void tile_fp8_skip(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb, int kb_lo, int kb_hi,
                                              int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    if (mvalid >= C::BM) tile_fp8_skip_1<C, MODE, SPLIT, true>(smem, srow, wss, o, nkb, kb_lo, kb_hi, fold, mvalid, acc);
    else tile_fp8_skip_1<C, MODE, SPLIT, false>(smem, srow, wss, o, nkb, kb_lo, kb_hi, fold, mvalid, acc);
}

// tile_fp4_1 with load_a_skip: the weight ring, the fragment loads and the arithmetic as there
template <class C, int MODE, bool FULL>
__device__ __forceinline__ void tile_fp4_skip_1(char* smem, const int* srow, const Ops& o, int nkb, int mvalid,
                                                float (&acc)[C::NWT][C::MT][C::NT][4]) {
    static_assert(C::FMT == 1 && C::KBS == 4, "MXFP4 tiles of W2's 4 K blocks a stage");
    using P = pf::W2<C>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    char* aring = smem;
    char* bring = smem + C::STAGES * P::A_SLOT;
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++) zero(acc[wi]);
    const int nst = nkb / C::KBS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) {
            load_a_skip<C>(aring + s * P::A_SLOT, srow, o, s * C::KBS);
            if ((s & 1) == 0) pf::load_b2<C>(bring + ((s >> 1) & 1) * P::SLOT, o, s * C::KBS, nkb);
        }
        pf::commit();
    }
    for (int st = 0; st < nst; st++) {
        pf::wait_groups<C::STAGES - 2>();
        __syncthreads();
        {
            const int nx = st + C::STAGES - 1;
            if (nx < nst) {
                load_a_skip<C>(aring + (nx % C::STAGES) * P::A_SLOT, srow, o, nx * C::KBS);
                if ((nx & 1) == 0) pf::load_b2<C>(bring + ((nx >> 1) & 1) * P::SLOT, o, nx * C::KBS, nkb);
            }
            pf::commit();
        }
        const char* A = aring + (st % C::STAGES) * P::A_SLOT;
        const float* AS = reinterpret_cast<const float*>(A + C::A_BYTES);
        const char* B = bring + ((st >> 1) & 1) * P::SLOT;
        const uint32_t* WS = reinterpret_cast<const uint32_t*>(B + P::B_BYTES);
        const int half = st & 1;
        uint32_t saw[C::MT];
#pragma unroll
        for (int i = 0; i < C::MT; i++) saw[i] = act_word<C::KBS>(AS + (wm * C::WM + i * 16 + g + 8 * (t & 1)) * C::KBS);
        uint32_t sbw[C::NWT][C::NT];
#pragma unroll
        for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
            for (int j = 0; j < C::NT; j++) sbw[wi][j] = WS[(wi * C::BN + wn * C::WN + j * 8 + g) * 2 + half];
        fp4_block<C, MODE, FULL, 0>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
        fp4_block<C, MODE, FULL, 1>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
        fp4_block<C, MODE, FULL, 2>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
        fp4_block<C, MODE, FULL, 3>(A, B, half, saw, sbw, wm, wn, g, t, mvalid, acc);
    }
    pf::wait_groups<0>();
}

template <class C, int MODE>
__device__ __forceinline__ void tile_fp4_skip(char* smem, const int* srow, const Ops& o, int nkb, int mvalid,
                                              float (&acc)[C::NWT][C::MT][C::NT][4]) {
    if (mvalid >= C::BM) tile_fp4_skip_1<C, MODE, true>(smem, srow, o, nkb, mvalid, acc);
    else tile_fp4_skip_1<C, MODE, false>(smem, srow, o, nkb, mvalid, acc);
}

// gate / up of the grouped experts on R2's tiles: the MXFP4 tile's activation rows with padding skipped, the FP8 shared
// expert on GuCfg8's sub-tiles the same way
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    const int c0 = blockIdx.x * C4::BN;
    const int main = (a.n_tiles - a.rest_tiles) * 32;
    if (a.owner && c0 >= main && a.owner[e] == 0) return;
    __shared__ int srow[C4::BM];
    __shared__ __align__(16) uint8_t wss[pf::kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < C4::BM; r += kThreads) srow[r] = m0 + r < cnt ? a.erow[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = reinterpret_cast<const uint8_t*>(a.w1[e]);
    o.w1 = reinterpret_cast<const uint8_t*>(a.w3[e]);
    o.ws0 = reinterpret_cast<const uint8_t*>(a.s1[e]);
    o.ws1 = reinterpret_cast<const uint8_t*>(a.s3[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = 0;
    const int nkb = a.D / kBlk;
    static_assert(C4::BM == pf::GuCfg8::BM, "the shared expert's sub-tiles: the tile's rows");
    if (fp8) {
        using C = pf::GuCfg8;
        for (int half = 0; half < C4::BN / C::BN; half++) {
            o.n0 = c0 + half * C::BN;
            if (half) __syncthreads();
            pf::fp8_scales<C>(wss, o, nkb);
            __syncthreads();
            float acc[2][C::MT][C::NT][4];
            tile_fp8_skip<C, MODE, false>(pf::smem, srow, wss, o, nkb, 0, nkb, nkb, cnt - m0, acc);
            pf::gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
        }
    } else {
        using C = C4;
        o.n0 = c0;
        __syncthreads();
        float acc[2][C::MT][C::NT][4];
        tile_fp4_skip<C, MODE>(pf::smem, srow, o, nkb, cnt - m0, acc);
    }
}

// down of the grouped experts on R2's 128-row MXFP4 tile (the 64-row tile list's entries at a multiple of 128 take a
// tile, the others return at once) and the FP8 shared expert's 64-row sub-tiles, two at a time
template <int MODE, class C4>
__device__ __forceinline__ void dn_r2(const DnArgs& a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    if (m0 % C4::BM != 0) return;  // C4::BM rows a tile: the list's other entries lie inside one
    __shared__ int srow[C4::BM];
    __shared__ __align__(16) uint8_t wss[pf::kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < C4::BM; r += kThreads) srow[r] = m0 + r < cnt ? a.ent[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = o.w1 = reinterpret_cast<const uint8_t*>(a.w2[e]);
    o.ws0 = o.ws1 = reinterpret_cast<const uint8_t*>(a.s2[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = 0;
    o.n0 = blockIdx.x * C4::BN;
    const int nkb = a.I / kBlk;
    if (fp8) {
        using C = pf::DnCfg8;
        for (int rh = 0; rh < C4::BM / C::BM; rh++) {
            const int s0 = m0 + rh * C::BM;
            if (s0 >= cnt) break;  // the tile's rows are done: the rest is padding
            if (rh) __syncthreads();
            pf::fp8_scales<C>(wss, o, nkb);
            __syncthreads();
            float acc[1][C::MT][C::NT][4];
            const int mv = cnt > s0 ? cnt - s0 : 0;
            tile_fp8_skip<C, MODE, false>(pf::smem, srow + rh * C::BM, wss, o, nkb, 0, nkb, nkb, mv, acc);
            pf::dn_tile_store<C>(a, acc, o.n0, base, s0, cnt);
        }
    } else {
        using C = C4;
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        tile_fp4_skip<C, MODE>(pf::smem, srow, o, nkb, cnt - m0, acc);
        pf::dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}


using GemmCfg = Cfg<128, 128, 2, 4, 1, 0, 2, 3, 2>;
using Split64Cfg = Cfg<64, 128, 2, 4, 1, 0, 2, 6, 2>;
using Split128Cfg = Cfg<128, 128, 2, 4, 1, 0, 2, 5, 2>;
// R2's expert tiles: down at 128 rows (a warp 4 m16 tiles by 4 n8), so each weight fragment feeds more MMAs; gate / up
// keeps the baseline 64-row tile, whose two weights' accumulators already fill the registers a wider tile would need. The
// activation rows are the baseline layout, a padding row's copies left out.
using GuR2Cfg = Cfg<64, 128, 2, 4, 2, 1, 4, 3>;
using DnR2Cfg = Cfg<128, 128, 2, 4, 1, 1, 4, 3>;
static_assert(pf::W2<GuR2Cfg>::SMEM == 97280 && pf::W2<DnR2Cfg>::SMEM == 90112, "the experts' shared memory");
// their dynamic shared memory (mx.zig sf_gemm_smem, sf_split64_smem, sf_split128_smem)
static_assert(GemmCfg::SMEM == 52224 && Split64Cfg::SMEM == 76800 && Split128Cfg::SMEM == 87040, "the dense tiles' shared memory");
// the same two split tiles on 16 warps a block (TF_DS_GEMM_W16): split64 doubles its column groups (a warp's tile
// 2 x 4 -> 2 x 2 m16 x n8) and split128 its row groups (4 x 4 -> 8 x 2), halving the C fragments a warp holds; the
// tile, its K blocks a stage, its stages and its shared memory are the 8-warp ones'. The grids are those whose
// fragments, scale words and addresses fit 128 registers a thread (a 512-thread block's limit), which the 512-thread
// build's spill count decides.
using Split64CfgW16 = Cfg<64, 128, 2, 8, 1, 0, 2, 6, 2>;
using Split128CfgW16 = Cfg<128, 128, 8, 2, 1, 0, 2, 5, 2>;
static_assert(Split64CfgW16::SMEM == Split64Cfg::SMEM && Split128CfgW16::SMEM == Split128Cfg::SMEM, "the 16-warp tiles' shared memory");
// The experts: pf::pf_gu's / pf_dn's 128-byte-run tiles (pf::GuR128Cfg, DnR128Cfg) and the FP8 shared expert's
// (pf::GuCfg8, DnCfg8: every one of their stages starts at an even K block)
static_assert(pf::GuCfg8::KBS % 2 == 0 && pf::DnCfg8::KBS % 2 == 0, "the shared expert's stages");
// the dynamic shared memory of the expert kernels (mx.zig sf_gu_smem / sf_dn_smem): their twins'
static_assert(pf::kGuR128Smem == 97280 && pf::kDnR128Smem == 62464, "the expert tiles' shared memory");

// ROWMAP: the order of a launch's CTAs by row quarter. A CTA's place in the launch, order = blockIdx.x + blockIdx.y * gridDim.x,
// gives its row tile and column tile: the launch is two phases (the CTAs before `span`, then the others) of `1 << per_log2` row
// tiles each, every column of a phase's row tiles before the next phase's, row tiles fastest within a phase; the row tile at
// position pos of the launch's row order is tile (pos & (tpq - 1)) of the quarter named by byte (pos >> tpq_log2) of `quarters`
// (tpq = 1 << tpq_log2 row tiles a quarter). The grid is the grid of the unordered launch, (ceil(M / BM), N / 128). No table is
// read and nothing is divided: a CTA's prologue is a few shifts. Without ROWMAP a CTA runs the tile (blockIdx.x, blockIdx.y).
struct RowOrder {
    unsigned quarters;
    unsigned tpq_log2;
    unsigned per_log2;
    unsigned span;
};
template <bool ROWMAP>
__device__ __forceinline__ unsigned tile_row(const RowOrder& ro) {
    if constexpr (ROWMAP) {
        const unsigned order = blockIdx.x + blockIdx.y * gridDim.x;
        const unsigned phase = order >= ro.span;
        const unsigned w = order - (phase ? ro.span : 0u);
        const unsigned pos = (phase << ro.per_log2) + (w & ((1u << ro.per_log2) - 1u));
        const unsigned q = (ro.quarters >> (8u * (pos >> ro.tpq_log2))) & 0xffu;
        return (q << ro.tpq_log2) | (pos & ((1u << ro.tpq_log2) - 1u));
    } else {
        return blockIdx.x;
    }
}
template <bool ROWMAP>
__device__ __forceinline__ unsigned tile_col(const RowOrder& ro) {
    if constexpr (ROWMAP) {
        const unsigned order = blockIdx.x + blockIdx.y * gridDim.x;
        const unsigned w = order - (order >= ro.span ? ro.span : 0u);
        return w >> ro.per_log2;
    } else {
        return blockIdx.y;
    }
}

// The dense FP8 GEMM of a prompt chunk (MxArgs as tf_dsn_pf_gemm: FP8 weights, N a multiple of 128, K of 64 up to 8192):
// grid (ceil(M / BM), N / 128), 256 threads, C::SMEM dynamic bytes. SPLIT: the K split (kblocks a split, even) folded in
// the CTA; ZSPLIT: one split a CTA (blockIdx.z), its partial into z [splits, M, N] fp32 for tf_dsn_reduce; neither: K
// whole.
// TWO: the activation rows' K blocks from `kb_split` on are read from a second base (q1, s1: the strides of the first) -- wo_b's input as the
// two pairs' parts where they lie, no assembled copy.
struct Part2 {
    const uint8_t* q1;
    const float* s1;
    int kb_split;
};

template <class C, int MODE, bool SPLIT, bool ZSPLIT, bool STAGE = false, bool ROWMAP = false, bool TWO = false>
__device__ __forceinline__ void gemm(const MxArgs& a, const RowOrder& ro = RowOrder{}, const Part2& p2 = Part2{}) {
    __shared__ int srow[C::BM];
    __shared__ __align__(16) uint8_t wss[pf::kWss];  // fp8_word reads it 2 bytes at a time
    // row tiles fastest: a wave's CTAs share their weight tiles in L2; without ROWMAP, a.raster (pf_tile_of) groups them
    int tr, tc;
    if constexpr (ROWMAP) {
        tr = int(tile_row<true>(ro));
        tc = int(tile_col<true>(ro));
    } else {
        pf_tile_of(a.raster, tr, tc);
    }
    const int m0 = tr * C::BM;
    for (int r = threadIdx.x; r < C::BM; r += C::THREADS) {
        const int m = m0 + r;
        srow[r] = m < a.M ? (a.rows ? a.rows[m] : m) : -1;
    }
    Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    if constexpr (TWO) {
        o.q1 = p2.q1;
        o.s1 = p2.s1;
        o.kb_split = p2.kb_split;
    }
    o.w0 = o.w1 = a.w; o.ws0 = o.ws1 = a.ws; o.ldw = a.ldw; o.ldws = a.ldws;
    o.n0 = tc * C::BN;
    o.stream = 0;
    const int nkb = a.K / kBlk;
    pf::fp8_scales<C>(wss, o, nkb);
    __syncthreads();
    const int lo = ZSPLIT ? int(blockIdx.z) * a.kblocks : 0;
    const int hi = ZSPLIT ? min(lo + a.kblocks, nkb) : nkb;
    float acc[1][C::MT][C::NT][4];
    tile_fp8<C, MODE, SPLIT, TWO>(pf::smem, srow, wss, o, nkb, lo, hi, a.kblocks, a.M - m0, acc);
    if constexpr (STAGE) return ::stage_store<C>(a, m0, o.n0, acc);  // y in write-combined host memory
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int i = 0; i < C::MT; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * C::WM + i * 16 + g + h * 8;
                if (m >= a.M) continue;
                const int n = o.n0 + wn * C::WN + j * 8 + 2 * t;
                const float v0 = acc[0][i][j][2 * h], v1 = acc[0][i][j][2 * h + 1];
                if (ZSPLIT) {
                    *reinterpret_cast<float2*>(a.z + (size_t(blockIdx.z) * a.M + m) * a.N + n) = make_float2(v0, v1);
                } else if (a.accumulate) {
                    float2* yp = reinterpret_cast<float2*>(reinterpret_cast<float*>(a.y) + size_t(m) * a.ldy + n);
                    float2 cur = *yp;
                    cur.x += v0;
                    cur.y += v1;
                    *yp = cur;
                } else {
                    store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v0);
                    store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n + 1, v1);
                }
            }
}

// The FP8 shared expert's column sub-tiles of sf::gu, a function of their own when `apart` (a 16-warp block's 128
// registers a thread do not hold this path's scale words and fragments beside the MXFP4 path's): each sub-tile's
// weight scales, K loop and epilogue in sf::gu's order.
template <class C4, class C, int MODE>
__device__ __noinline__ void gu_fp8_tiles(const GuArgs& a, const Ops& o0, const int* srow, uint8_t* wss, int c0, int base, int m0,
                                          int cnt, int nkb) {
    Ops o = o0;
    for (int half = 0; half < C4::BN / C::BN; half++) {
        o.n0 = c0 + half * C::BN;
        if (half) __syncthreads();  // the stages' buffers and the scales are the next sub-tile's
        fp8_scales<C>(wss, o, nkb);
        __syncthreads();
        float acc[2][C::MT][C::NT][4];
        tile_fp8<C, MODE, false>(pf::smem, srow, wss, o, nkb, 0, nkb, nkb, cnt - m0, acc);
        pf::gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

// Gate / up of the grouped experts for a prompt chunk (tf_dsn_pf_gu_r128's arguments, grid, tiles and epilogue): the
// MXFP4 routed experts on C4's 128-byte-run tiles, the FP8 shared expert on C8's column sub-tiles (the defaults the
// 8-warp kernels' configurations; a 16-warp block passes its own, `APART` its shared expert in gu_fp8_tiles).
    using C4 = C4_;
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    const int c0 = blockIdx.x * C4::BN;
    const int main = (a.n_tiles - a.rest_tiles) * 32;
    if (a.owner && c0 >= main && a.owner[e] == 0) return;
    __shared__ int srow[pf::kBm];
    __shared__ __align__(16) uint8_t wss[pf::kWss];  // fp8_word reads it 2 bytes at a time
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < pf::kBm; r += C4::THREADS) srow[r] = m0 + r < cnt ? a.erow[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = reinterpret_cast<const uint8_t*>(a.w1[e]);
    o.w1 = reinterpret_cast<const uint8_t*>(a.w3[e]);
    o.ws0 = reinterpret_cast<const uint8_t*>(a.s1[e]);
    o.ws1 = reinterpret_cast<const uint8_t*>(a.s3[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = 0;
    const int nkb = a.D / kBlk;
    if (fp8) {
        using C = C8_;
        if constexpr (APART) {
            gu_fp8_tiles<C4, C, MODE>(a, o, srow, wss, c0, base, m0, cnt, nkb);
        } else {
            for (int half = 0; half < C4::BN / C::BN; half++) {
                o.n0 = c0 + half * C::BN;
                if (half) __syncthreads();  // the stages' buffers and the scales are the next sub-tile's
                pf::fp8_scales<C>(wss, o, nkb);
                __syncthreads();
                float acc[2][C::MT][C::NT][4];
                tile_fp8<C, MODE, false>(pf::smem, srow, wss, o, nkb, 0, nkb, nkb, cnt - m0, acc);
                pf::gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
            }
        }
    } else {
        using C = C4;
        o.n0 = c0;
        __syncthreads();
        float acc[2][C::MT][C::NT][4];
        tile_fp4<C, MODE>(pf::smem, srow, o, nkb, cnt - m0, acc);
    }
}

// Down of the grouped experts for a prompt chunk (tf_dsn_pf_dn_r128's arguments, grid, tiles and epilogue): the MXFP4
// routed experts on C4's 128-byte-run tiles, the FP8 shared expert on C8's tile (the defaults the 8-warp kernels'
// configurations; a 16-warp block passes its own).
// THREE: down's activation rows (the pack, hq / hs) as the two pairs' pack parts where they lie, not assembled. A part is its main block
// [rows, ig] of codes (row stride a.ldq) then its rest block [rows, ir] (ldq_rest), the scales the same (a.lds, lds_rest). K blocks
// [0, kb_main) come from the first part's main block (a.q, a.s), [kb_main, 2 kb_main) from the second's (q1, s1), and the rest block from
// the part of the pair that computed the tile's expert's rest (owner[e], 0: the first), at (qr0, sr0) / (qr1, sr1), the pointers at the
// first row of the rest block.
struct Part3 {
    const uint8_t* q1;
    const float* s1;
    const uint8_t* qr0;
    const float* sr0;
    const uint8_t* qr1;
    const float* sr1;
    const int* owner;
    int64_t ldq_rest, lds_rest;  // the rest block's row strides (bytes, floats); a.ldq / a.lds are the main blocks'
    int kb_main;
};

template <int MODE, bool THREE = false, class C4_ = pf::DnR128Cfg, class C8_ = pf::DnCfg8>
__device__ __forceinline__ void dn(const DnArgs& a, const int2* __restrict__ tiles, const int* __restrict__ ntiles, const Part3& p3 = Part3{}) {
    using C4 = C4_;
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    __shared__ int srow[pf::kBm];
    __shared__ __align__(16) uint8_t wss[pf::kWss];  // fp8_word reads it 2 bytes at a time
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < pf::kBm; r += C4::THREADS) srow[r] = m0 + r < cnt ? a.ent[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    if constexpr (THREE) {
        const bool second = p3.owner[e] != 0;
        o.q1 = p3.q1;
        o.s1 = p3.s1;
        o.kb_split = p3.kb_main;
        o.q2 = second ? p3.qr1 : p3.qr0;
        o.s2 = second ? p3.sr1 : p3.sr0;
        o.kb_split2 = 2 * p3.kb_main;
        o.ldq2 = p3.ldq_rest;
        o.lds2 = p3.lds_rest;
    }
    o.w0 = o.w1 = reinterpret_cast<const uint8_t*>(a.w2[e]);
    o.ws0 = o.ws1 = reinterpret_cast<const uint8_t*>(a.s2[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = 0;
    o.n0 = blockIdx.x * C4::BN;
    const int nkb = a.I / kBlk;
    if (fp8) {
        using C = C8_;
        pf::fp8_scales<C>(wss, o, nkb);
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        tile_fp8<C, MODE, false, false, THREE>(pf::smem, srow, wss, o, nkb, 0, nkb, nkb, cnt - m0, acc);
        pf::dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    } else {
        using C = C4;
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        tile_fp4<C, MODE, THREE>(pf::smem, srow, o, nkb, cnt - m0, acc);
        pf::dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

}  // namespace sf

// The smallest and the largest byte of each of n device ranges (TF_DS_MMA_SF_EXACT's guard over the weights' UE8M0
// scales, mx.zig scaleRanges): CTA i reads [ptrs[i], ptrs[i] + lens[i]), 16 bytes a load while the range is 16-byte
// aligned and its tail a byte at a time, and writes mm[2i] (the smallest) and mm[2i + 1] (the largest); an empty range
// gives 255 and 0. 256 threads, grid (n).
extern "C" __global__ void __launch_bounds__(256) tf_dsn_sf_ranges_kernel(const uint64_t* __restrict__ ptrs, const uint64_t* __restrict__ lens,
                                                                          int n, unsigned* __restrict__ mm) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(ptrs[i]);
    const uint64_t len = lens[i];
    unsigned lo = 0xffffffffu, hi = 0u;  // four bytes' minima and maxima side by side
    const uint64_t nv = (reinterpret_cast<uintptr_t>(p) % 16 == 0) ? len / 16 : 0;
    const uint4* v = reinterpret_cast<const uint4*>(p);
    for (uint64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const uint4 w = v[j];
        lo = __vminu4(__vminu4(lo, w.x), __vminu4(__vminu4(w.y, w.z), w.w));
        hi = __vmaxu4(__vmaxu4(hi, w.x), __vmaxu4(__vmaxu4(w.y, w.z), w.w));
    }
    for (uint64_t j = nv * 16 + threadIdx.x; j < len; j += blockDim.x) {
        const unsigned b = unsigned(p[j]) * 0x01010101u;
        lo = __vminu4(lo, b);
        hi = __vmaxu4(hi, b);
    }
    unsigned l1 = min(min(lo & 255u, (lo >> 8) & 255u), min((lo >> 16) & 255u, lo >> 24));
    unsigned h1 = max(max(hi & 255u, (hi >> 8) & 255u), max((hi >> 16) & 255u, hi >> 24));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        l1 = min(l1, __shfl_xor_sync(0xffffffffu, l1, o));
        h1 = max(h1, __shfl_xor_sync(0xffffffffu, h1, o));
    }
    __shared__ unsigned slo[8], shi[8];
    if ((threadIdx.x & 31) == 0) {
        slo[threadIdx.x >> 5] = l1;
        shi[threadIdx.x >> 5] = h1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < int(blockDim.x >> 5); w++) {
            l1 = min(l1, slo[w]);
            h1 = max(h1, shi[w]);
        }
        mm[2 * i] = l1;
        mm[2 * i + 1] = h1;
    }
}

// The dense FP8 GEMM of a prompt chunk without a K split (MxArgs as tf_dsn_pf_gemm, splits 1): grid (ceil(M / 128),
// N / 128), 256 threads, sf::GemmCfg::SMEM dynamic bytes; _exact: the exact arithmetic (tf_dsn_pf_gemm's bits).
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kChained, false, false>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_exact_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kExact, false, false>(a); }
// The dense FP8 GEMMs with a K split (MxArgs as tf_dsn_pf_gemm_fold: splits, kblocks even), the split folded in the
// CTA: grid (ceil(M / 64), N / 128) and sf::Split64Cfg::SMEM bytes, or (ceil(M / 128), N / 128) and
// sf::Split128Cfg::SMEM; _exact: the exact arithmetic (tf_dsn_pf_gemm_fold's bits).
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kChained, true, false>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_exact_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kExact, true, false>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kChained, true, false>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kExact, true, false>(a); }
// tf_dsn_sf_split64's split a CTA (blockIdx.z of `splits`), its partial into z [splits, M, N] fp32: grid (ceil(M / 64),
// N / 128, splits), sf::Split64Cfg::SMEM bytes, then tf_dsn_reduce: the folded kernels' bits at a small chunk's rows.
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_zsplit_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kChained, false, true>(a); }
// The same kernels storing their fp32 tile as whole lines (stage_store), for a y in write-combined host memory.
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_st_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kChained, false, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_exact_st_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kExact, false, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_st_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kChained, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_exact_st_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kExact, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_st_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kChained, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kExact, true, false, true>(a); }
// the same on the activation rows' two bases (sf::Part2): wo_b's input as the node's part and the column partner's part where they lie
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_two_kernel(MxArgs a, sf::Part2 p) { sf::gemm<sf::Split128Cfg, sf::kExact, true, false, true, false, true>(a, sf::RowOrder{}, p); }
// the chained arithmetic's (TF_DS_MMA_SF) twin
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_st_two_kernel(MxArgs a, sf::Part2 p) { sf::gemm<sf::Split128Cfg, sf::kChained, true, false, true, false, true>(a, sf::RowOrder{}, p); }
// tf_dsn_sf_split128_exact_st with the CTAs in `ro`'s order (sf::gemm's ROWMAP): same grid, same arithmetic of every output element.
// The timed twin also writes each CTA's %globaltimer at its begin and end to ts[2 * order] and ts[2 * order + 1] (order: its place
// in the launch, blockIdx.x + blockIdx.y * gridDim.x).
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_first_kernel(MxArgs a, sf::RowOrder ro) {
    sf::gemm<sf::Split128Cfg, sf::kExact, true, false, true, true>(a, ro);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_first_t_kernel(MxArgs a, sf::RowOrder ro, unsigned long long* __restrict__ ts) {
    if (threadIdx.x == 0) ts[2 * (blockIdx.x + blockIdx.y * gridDim.x)] = gtimer();
    sf::gemm<sf::Split128Cfg, sf::kExact, true, false, true, true>(a, ro);
    __syncthreads();
    if (threadIdx.x == 0) ts[2 * (blockIdx.x + blockIdx.y * gridDim.x) + 1] = gtimer();
}
// The same in `ro`'s order with the doorbell rung from inside: once the first ro.span CTAs (the sent quarters') have stored their
// tiles (Door: native.cu). EARLY: the arrival is counted before the CTA's stores (a hazard: the control that must be seen); TIMED:
// the begin and end stamps as above and the ring's in ts[2 * gridDim.x * gridDim.y].
template <bool EARLY, bool TIMED, int BLOCK, int MODE = sf::kExact>
__device__ __forceinline__ void gemm_door(const MxArgs& a, const sf::RowOrder& ro, const Door& d, unsigned long long* ts) {
    const unsigned order = blockIdx.x + blockIdx.y * gridDim.x;
    const bool sent = order < ro.span;
    if constexpr (TIMED) {
        if (threadIdx.x == 0) ts[2 * order] = gtimer();
    }
    if constexpr (EARLY) {
        if (sent && threadIdx.x == 0) door_arrive(d, ro.span);
    }
    sf::gemm<sf::Split128Cfg, MODE, true, false, true, true>(a, ro);
    if constexpr (!EARLY) {
        if (sent) {
            if (door_arrive_cta<BLOCK>(d, ro.span)) {
                if constexpr (TIMED) ts[2 * gridDim.x * gridDim.y] = gtimer();
            }
        }
    }
    if constexpr (TIMED) {
        __syncthreads();
        if (threadIdx.x == 0) ts[2 * order + 1] = gtimer();
    }
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_door_kernel(MxArgs a, sf::RowOrder ro, Door d) {
    gemm_door<false, false, 0>(a, ro, d, nullptr);
}
// the chained arithmetic's (TF_DS_MMA_SF) twin of the kernel above, the same order and door
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_st_door_kernel(MxArgs a, sf::RowOrder ro, Door d) {
    gemm_door<false, false, 0, sf::kChained>(a, ro, d, nullptr);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_door_early_kernel(MxArgs a, sf::RowOrder ro, Door d) {
    gemm_door<true, false, 0>(a, ro, d, nullptr);
}
// _b: the block-level release (door_arrive_cta<true>), _b_t its timed twin
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_door_b_kernel(MxArgs a, sf::RowOrder ro, Door d) {
    gemm_door<false, false, 1>(a, ro, d, nullptr);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_door_b_t_kernel(MxArgs a, sf::RowOrder ro, Door d, unsigned long long* __restrict__ ts) {
    gemm_door<false, true, 1>(a, ro, d, ts);
}
// _g: the release at GPU scope with the system fence by the last arrival alone (door_arrive_cta<2>), _g_t its timed twin
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_door_g_kernel(MxArgs a, sf::RowOrder ro, Door d) {
    gemm_door<false, false, 2>(a, ro, d, nullptr);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_door_g_t_kernel(MxArgs a, sf::RowOrder ro, Door d, unsigned long long* __restrict__ ts) {
    gemm_door<false, true, 2>(a, ro, d, ts);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_door_t_kernel(MxArgs a, sf::RowOrder ro, Door d, unsigned long long* __restrict__ ts) {
    gemm_door<false, true, 0>(a, ro, d, ts);
}
// Gate / up and down of the grouped experts for a prompt chunk (tf_dsn_pf_gu_r128's / tf_dsn_pf_dn_r128's arguments,
// grid and tiles): pf::kGuR128Smem / pf::kDnR128Smem dynamic bytes; _exact: the exact arithmetic (the twins' bits).
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gu_kernel(GuArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::gu<sf::kChained>(a, tiles, ntiles);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gu_exact_kernel(GuArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::gu<sf::kExact>(a, tiles, ntiles);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_dn_kernel(DnArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::dn<sf::kChained>(a, tiles, ntiles);
}
// tf_dsn_sf_dn_exact_kernel on the pack's two parts and the rest block's owner part where they lie (sf::Part3)
// the chained arithmetic's (TF_DS_MMA_SF) down on the pack's parts
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_dn_three_kernel(DnArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles, sf::Part3 p) {
    sf::dn<sf::kChained, true>(a, tiles, ntiles, p);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_dn_exact_three_kernel(DnArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles, sf::Part3 p) {
    sf::dn<sf::kExact, true>(a, tiles, ntiles, p);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_dn_exact_kernel(DnArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::dn<sf::kExact>(a, tiles, ntiles);
}

// The exact arithmetic's split tiles and experts on 16 warps a block (TF_DS_GEMM_W16). The tile, its K blocks a
// stage, its stages and its loads are the 8-warp kernels'; only the warp grid moves, so every output element is still
// produced by one accumulator over the same K blocks in the same order, and the bytes are theirs. The dynamic shared
// memory is theirs too (mx.zig sf_split64_smem, sf_split128_smem, pf_gu_r128_smem, pf_dn_r128_smem): one block an SM.
extern "C" __global__ void __launch_bounds__(512, 1) tf_dsn_sf_split64_exact_w16_kernel(MxArgs a) { sf::gemm<sf::Split64CfgW16, sf::kExact, true, false>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) tf_dsn_sf_split128_exact_w16_kernel(MxArgs a) { sf::gemm<sf::Split128CfgW16, sf::kExact, true, false>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) tf_dsn_sf_gu_exact_w16_kernel(GuArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::gu<sf::kExact, pf::GuR128CfgW16, pf::GuCfg8W16, true>(a, tiles, ntiles);
}
extern "C" __global__ void __launch_bounds__(512, 1) tf_dsn_sf_dn_exact_w16_kernel(DnArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::dn<sf::kExact, false, pf::DnR128CfgW16, pf::DnCfg8W16>(a, tiles, ntiles);
}

// TF_DS_GEMM_R2: the exact arithmetic on the same split fold (down on 128-row tiles), its activation rows staged so an
// m16 tile's A fragment of a K block is one 16-byte load and padding rows are not copied. Each output's MMAs, K order
// and fp32 adds are the kernels above's, so their bytes.
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gu_exact_r2_kernel(GuArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::gu_r2<sf::kExact, sf::GuR2Cfg>(a, tiles, ntiles);
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_dn_exact_r2_kernel(DnArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    sf::dn_r2<sf::kExact, sf::DnR2Cfg>(a, tiles, ntiles);
}

#ifdef TF_SF_LOAD_ONLY
// ---------------------------------------------------------------------------------------------------------------------
// The scale spreads of the indexer's FP4 dots on real prompts (a test-instruments build; the host's idx_spread.zig launches it beside a chunk's
// index score). A dot of one query head with one key is the sum over the four 32-value blocks b of an exact block sum times 2^(eq[h][b] +
// ek[b]) (eq: the E8M0 exponent byte _fp4qd_p2 gives the head's block of the query, ek: the key's stored scale byte), so the dot is
// exact in the tensor core only while the spread of those four exponents is small. For a sample of rows (every row_step-th) and keys (every
// key_step-th, visible ones only) it counts, over all 32 heads of the row: the pair's largest head spread S (hist [0, 32)), the key's own
// spread (hist [32, 64)), the row's largest head spread of eq (hist [64, 96)), the pairs seen (ctr[96]) and those a sufficient test (the row's
// eq spread + the key's spread <= 7) passes (ctr[97]).
__device__ __forceinline__ uint32_t spread_qd_scale(float amax) {
    const float sv = __fdiv_rn(fmaxf(amax, 6.0f * 0x1p-126f), 6.0f);
    const uint32_t bits = __float_as_uint(sv);
    return ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u);
}
extern "C" __global__ void __launch_bounds__(128) tf_dsn_idx_spread_kernel(const uint16_t* __restrict__ iq, const uint8_t* __restrict__ ks,
                                                                           const int64_t* __restrict__ vis, int rows, int row_step, int n,
                                                                           int key_step, unsigned* __restrict__ ctr) {
    __shared__ int eq[32][4];
    __shared__ int qrow;
    const int r = blockIdx.x * row_step;
    if (r >= rows) return;
    const int t = threadIdx.x, h0 = t >> 2, b0 = t & 3;
    {
        const uint4* src = reinterpret_cast<const uint4*>(iq + (size_t(r) * 32 + h0) * 128 + b0 * 32);
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint4 u = src[i];
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                amax = fmaxf(amax, fabsf(__uint_as_float((w[j] & 0xffffu) << 16)));
                amax = fmaxf(amax, fabsf(__uint_as_float(w[j] & 0xffff0000u)));
            }
        }
        eq[h0][b0] = int(spread_qd_scale(amax));
    }
    __syncthreads();
    if (t == 0) {
        int m = 0;
        for (int h = 0; h < 32; h++) {
            int lo = 255, hi = 0;
            for (int b = 0; b < 4; b++) { lo = min(lo, eq[h][b]); hi = max(hi, eq[h][b]); }
            m = max(m, hi - lo);
        }
        qrow = m;
        atomicAdd(&ctr[64 + min(m, 31)], 1u);
    }
    __syncthreads();
    const int64_t lim = min(int64_t(n), vis[r]);
    for (int kk = t; int64_t(kk) * key_step < lim; kk += 128) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(ks + size_t(kk) * key_step * 4);
        const int e0 = w & 0xff, e1 = (w >> 8) & 0xff, e2 = (w >> 16) & 0xff, e3 = w >> 24;
        const int ksp = max(max(e0, e1), max(e2, e3)) - min(min(e0, e1), min(e2, e3));
        int smax = 0;
        for (int h = 0; h < 32; h++) {
            const int a0 = eq[h][0] + e0, a1 = eq[h][1] + e1, a2 = eq[h][2] + e2, a3 = eq[h][3] + e3;
            smax = max(smax, max(max(a0, a1), max(a2, a3)) - min(min(a0, a1), min(a2, a3)));
        }
        atomicAdd(&ctr[min(smax, 31)], 1u);
        atomicAdd(&ctr[32 + min(ksp, 31)], 1u);
        atomicAdd(&ctr[96], 1u);
        if (qrow + ksp <= 7) atomicAdd(&ctr[97], 1u);
    }
}
#endif
