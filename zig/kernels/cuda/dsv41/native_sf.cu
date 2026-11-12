template <class C, int MODE, bool FULL, bool TWO = false, bool THREE = false>
        if (nx < nst) pf::load_stage<C, TWO, THREE>(smem + (nx % C::STAGES) * C::STAGE, srow, o, kb_lo + nx * C::KBS);
template <class C, int MODE, bool SPLIT, bool FULL, bool TWO = false, bool THREE = false>
        if (s < nst) pf::load_stage<C, TWO, THREE>(smem + s * C::STAGE, srow, o, kb_lo + s * C::KBS);
        for (int st = 0; st < nst; st++) fp8_stage<C, MODE, FULL, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, nst, st, mvalid, acc);
            for (; st < st_end; st++) fp8_stage<C, MODE, FULL, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, nst, st, mvalid, part);
template <class C, int MODE, bool SPLIT, bool TWO = false, bool THREE = false>
    if (mvalid >= C::BM) tile_fp8_1<C, MODE, SPLIT, true, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, kb_hi, fold, mvalid, acc);
    else tile_fp8_1<C, MODE, SPLIT, false, TWO, THREE>(smem, srow, wss, o, nkb, kb_lo, kb_hi, fold, mvalid, acc);
template <class C, int MODE, bool FULL, bool THREE = false>
            pf::load_a<C, THREE>(aring + s * P::A_SLOT, srow, o, s * C::KBS);
                pf::load_a<C, THREE>(aring + (nx % C::STAGES) * P::A_SLOT, srow, o, nx * C::KBS);
template <class C, int MODE, bool THREE = false>
    if (mvalid >= C::BM) tile_fp4_1<C, MODE, true, THREE>(smem, srow, o, nkb, mvalid, acc);
    else tile_fp4_1<C, MODE, false, THREE>(smem, srow, o, nkb, mvalid, acc);
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

// TWO: the activation rows' K blocks from `kb_split` on are read from a second base (q1, s1: the strides of the first) -- wo_b's input as the
// two pairs' parts where they lie, no assembled copy.
struct Part2 {
    const uint8_t* q1;
    const float* s1;
    int kb_split;
};

template <class C, int MODE, bool SPLIT, bool ZSPLIT, bool STAGE = false, bool ROWMAP = false, bool TWO = false>
__device__ __forceinline__ void gemm(const MxArgs& a, const RowOrder& ro = RowOrder{}, const Part2& p2 = Part2{}) {
    if constexpr (TWO) {
        o.q1 = p2.q1;
        o.s1 = p2.s1;
        o.kb_split = p2.kb_split;
    }
    tile_fp8<C, MODE, SPLIT, TWO>(pf::smem, srow, wss, o, nkb, lo, hi, a.kblocks, a.M - m0, acc);
    if constexpr (STAGE) return ::stage_store<C>(a, m0, o.n0, acc);  // y in write-combined host memory
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

__device__ __forceinline__ void dn(const DnArgs& a, const int2* __restrict__ tiles, const int* __restrict__ ntiles, const Part3& p3 = Part3{}) {
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
        tile_fp8<C, MODE, false, false, THREE>(pf::smem, srow, wss, o, nkb, 0, nkb, nkb, cnt - m0, acc);
        tile_fp4<C, MODE, THREE>(pf::smem, srow, o, nkb, cnt - m0, acc);
// The same kernels storing their fp32 tile as whole lines (stage_store), for a y in write-combined host memory.
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_st_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kChained, false, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_gemm_exact_st_kernel(MxArgs a) { sf::gemm<sf::GemmCfg, sf::kExact, false, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_st_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kChained, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split64_exact_st_kernel(MxArgs a) { sf::gemm<sf::Split64Cfg, sf::kExact, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_st_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kChained, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_kernel(MxArgs a) { sf::gemm<sf::Split128Cfg, sf::kExact, true, false, true>(a); }
// the same on the activation rows' two bases (sf::Part2): wo_b's input as the node's part and the column partner's part where they lie
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_split128_exact_st_two_kernel(MxArgs a, sf::Part2 p) { sf::gemm<sf::Split128Cfg, sf::kExact, true, false, true, false, true>(a, sf::RowOrder{}, p); }
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
__device__ __forceinline__ void gemm_door(const MxArgs& a, const sf::RowOrder& ro, const Door& d, unsigned long long* ts) {
    const unsigned order = blockIdx.x + blockIdx.y * gridDim.x;
    const bool sent = order < ro.span;
    if constexpr (TIMED) {
        if (threadIdx.x == 0) ts[2 * order] = gtimer();
    }
    if constexpr (EARLY) {
        if (sent && threadIdx.x == 0) door_arrive(d, ro.span);
    }
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
// tf_dsn_sf_dn_exact_kernel on the pack's two parts and the rest block's owner part where they lie (sf::Part3)
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_sf_dn_exact_three_kernel(DnArgs a, const int2* __restrict__ tiles, const int* __restrict__ ntiles, sf::Part3 p) {
    sf::dn<sf::kExact, true>(a, tiles, ntiles, p);
}