// DeepSeek-V4.1-Flash's own checkpoint format on GB10 (sm_121): the linears and experts of
// deepseek-ai/DeepSeek-V4.1-Flash-1017 as its reference inference code computes them (inference/model.py `linear`,
// kernel.py `act_quant`, `fp8_gemm`, `fp4_gemm`; MIT, Copyright (c) 2023 DeepSeek). Written for this port from that
// arithmetic, no code copied:
//   - an activation is rounded to bf16, then quantized to FP8 E4M3 with one power-of-two (UE8M0) scale per 32 values
//     (amax floored at 1e-4, scale 2^ceil(log2(amax / 448)));
//   - FP8 weights carry one UE8M0 scale per 32 x 32 block, MXFP4 weights (E2M1 pairs, low nibble first) one per 32
//     values of a row; every E2M1 value is exact in E4M3, so both run on the same e4m3 tensor-core MMA;
//   - each 32-wide K block's MMA sum is scaled by both scales and added to an fp32 accumulator, K blocks in order.
// A row's result never depends on the other rows of a launch (one row's MMA lanes never mix with another's) and every
// layer's K split is fixed by its shape, so drafted, concurrent and resumed rows get the bits of a row alone.
// Raw device pointers, the caller's stream, extern "C" names (tf_dsn_*).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stddef.h>
#include <type_traits>

namespace {

constexpr int kBlk = 32;           // the scale block (fp8 and fp4, weights and activations)
constexpr float kFp8Max = 448.0f;  // E4M3's largest finite value

// dtype codes of the engine's EXL3 calls (exl3_linear.DType): f16, bf16, f32
__device__ __forceinline__ float load_x(const void* x, int dtype, size_t i) {
    if (dtype == 1) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[i]);
    if (dtype == 0) return __half2float(reinterpret_cast<const __half*>(x)[i]);
    return reinterpret_cast<const float*>(x)[i];
}

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// kernel.py fast_round_scale: 2^ceil(log2(amax * (1/448))) from the fp32 bits (exponent, plus one if any mantissa bit)
__device__ __forceinline__ float round_scale(float amax) {
    const float v = amax * (1.0f / kFp8Max);
    const uint32_t b = __float_as_uint(v);
    const int e = int((b >> 23) & 0xff) - 127 + ((b & 0x7fffff) ? 1 : 0);
    return __uint_as_float(uint32_t(e + 127) << 23);
}

__device__ __forceinline__ float ue8m0(uint8_t e) { return __uint_as_float(uint32_t(e) << 23); }

__device__ __forceinline__ uint8_t to_e4m3(float v) {
    return uint8_t(__nv_cvt_float_to_fp8(fminf(fmaxf(v, -kFp8Max), kFp8Max), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float from_e4m3(uint8_t q) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    return __half2float(__half(h));
}

// E2M1 code -> E4M3 byte: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (convert.py FP4_TABLE), all exact in E4M3
__constant__ uint8_t kE2m1ToE4m3[16] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c,
                                        0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc};

// two packed bytes (four E2M1 values, low nibble first) -> four E4M3 bytes, first value in the low byte
__device__ __forceinline__ uint32_t fp4x4_to_e4m3(uint16_t p) {
    return uint32_t(kE2m1ToE4m3[p & 0xf]) | uint32_t(kE2m1ToE4m3[(p >> 4) & 0xf]) << 8 |
           uint32_t(kE2m1ToE4m3[(p >> 8) & 0xf]) << 16 | uint32_t(kE2m1ToE4m3[(p >> 12) & 0xf]) << 24;
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void store_y(void* y, int dtype, size_t i, float v) {
    if (dtype == 1) reinterpret_cast<__nv_bfloat16*>(y)[i] = __float2bfloat16_rn(v);
    else if (dtype == 0) reinterpret_cast<__half*>(y)[i] = __float2half_rn(v);
    else reinterpret_cast<float*>(y)[i] = v;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// act_quant: x [M, K] (row stride ldx, dtype f16/bf16/f32) -> q [M, K] E4M3 (row stride ldq) and s [M, K/32] fp32
// powers of two (row stride lds); kernel.py act_quant(x, 32, "ue8m0") of x.to(bf16). With `fake` (non-null) the
// dequantized values go to fake [M, K] bf16 instead (act_quant(..., inplace=True), wo_a's input). A warp a 32-group.
extern "C" __global__ void tf_dsn_act_quant_kernel(const void* __restrict__ x, int x_dtype, int64_t ldx, int M, int K,
                                                   uint8_t* __restrict__ q, int64_t ldq, float* __restrict__ s,
                                                   int64_t lds, __nv_bfloat16* __restrict__ fake, int64_t ldf) {
    const int groups = K / kBlk;
    const int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= int64_t(M) * groups) return;
    const int m = int(warp / groups), g = int(warp % groups);
    const int k = g * kBlk + lane;
    const float v = bf16r(load_x(x, x_dtype, size_t(m) * ldx + k));
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = round_scale(fmaxf(amax, 1e-4f));
    const uint8_t c = to_e4m3(v / sc);
    if (fake) {
        fake[size_t(m) * ldf + k] = __float2bfloat16_rn(from_e4m3(c) * sc);
    } else {
        q[size_t(m) * ldq + k] = c;
        if (lane == 0) s[size_t(m) * lds + g] = sc;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// act_quant on four lanes a 32-group (TF_DS_AQ_VEC, on unless 0; mx.zig aqVecFits says when a call may take it): the bytes of tf_dsn_act_quant_kernel above,
// from 16-byte loads instead of 2-byte ones. A lane takes 8 consecutive bf16 values (one uint4 load), the group's amax is the max over the lane's eight |v|
// and two xor shuffles (fmaxf ignores NaN, so any order gives base's value: all-NaN gives NaN as base), the scale round_scale(max(amax, 1e-4)), the codes
// to_e4m3(v * (1 / sc)) written as one 8-byte store, lane 0 of the group the scale. The reciprocal is built from the scale's bits (0x7f000000 - bits) only when
// sc is a normal power of two whose reciprocal is normal (exponent field 1 .. 253): then v * (1 / sc) and v / sc are the same real number correctly rounded
// (a power-of-two scale), so the same float; any other scale (zero, subnormal, 2^127, inf) takes base's division. `fake` (non-null) gets the dequantized
// rows as 16-byte stores, from_e4m3(c) * sc rounded to bf16 as base. Calls: bf16 x only, K a multiple of 32, ldx % 8 == 0 and x 16-byte aligned, ldq % 8 == 0
// and q 8-byte aligned (fake: ldf % 8 == 0 and 16-byte aligned): the host (mx.zig aqVecFits) sends anything else to the kernel above. Launch: 4 threads a
// group, 256 threads a block. The kernels replay's act_quant section compares every output byte with the kernel above's.
extern "C" __global__ void __launch_bounds__(256) tf_dsn_act_quant_vec_kernel(const void* __restrict__ x, int x_dtype, int64_t ldx, int M, int K,
                                                                              uint8_t* __restrict__ q, int64_t ldq, float* __restrict__ s,
                                                                              int64_t lds, __nv_bfloat16* __restrict__ fake, int64_t ldf) {
    (void)x_dtype;  // (bf16: aqVecFits)
    const int groups = K / kBlk;
    const int64_t unit = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t grp = unit >> 2;
    const int sub = int(unit & 3);
    const bool live = grp < int64_t(M) * groups;  // (no early return before the shuffles: they take the whole warp)
    const int m = live ? int(grp / groups) : 0, g = live ? int(grp % groups) : 0;
    float f[8];
    if (live) {
        const uint4 u = *reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(x) + int64_t(m) * ldx + int64_t(g) * kBlk + sub * 8);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 p = __bfloat1622float2(h[j]);
            f[2 * j] = bf16r(p.x);
            f[2 * j + 1] = bf16r(p.y);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; j++) f[j] = 0.f;
    }
    float amax = fabsf(f[0]);
#pragma unroll
    for (int j = 1; j < 8; j++) amax = fmaxf(amax, fabsf(f[j]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (!live) return;
    const float sc = round_scale(fmaxf(amax, 1e-4f));
    const uint32_t sb = __float_as_uint(sc);
    const uint32_t se = (sb >> 23) & 0xffu;
    const bool rcp_ok = (sb & 0x7fffffu) == 0 && (sb >> 31) == 0 && se >= 1 && se <= 253;
    const float inv = rcp_ok ? __uint_as_float(0x7f000000u - sb) : 0.f;
    uint8_t c[8];
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = to_e4m3(rcp_ok ? f[j] * inv : f[j] / sc);
    if (fake) {
        uint4 o;
        __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
        for (int j = 0; j < 4; j++) oh[j] = __floats2bfloat162_rn(from_e4m3(c[2 * j]) * sc, from_e4m3(c[2 * j + 1]) * sc);
        *reinterpret_cast<uint4*>(fake + int64_t(m) * ldf + int64_t(g) * kBlk + sub * 8) = o;
    } else {
        uint2 o;
        o.x = uint32_t(c[0]) | uint32_t(c[1]) << 8 | uint32_t(c[2]) << 16 | uint32_t(c[3]) << 24;
        o.y = uint32_t(c[4]) | uint32_t(c[5]) << 8 | uint32_t(c[6]) << 16 | uint32_t(c[7]) << 24;
        *reinterpret_cast<uint2*>(q + int64_t(m) * ldq + int64_t(g) * kBlk + sub * 8) = o;
        if (sub == 0) s[int64_t(m) * lds + g] = sc;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// A quality gate's body canary (TF_DS_QGATE_CANARY 2 / 3, never a serving setting): x [n] fp32 perturbed in place on
// purpose, n whole 32-value groups. mode 2: one more FP8 step, act_quant's quant-dequant of each group (the value rounded
// to bf16, its E4M3 code at the group's power-of-two scale, back: exact in fp32, what fake writes); mode 3: each
// nonzero finite value one ulp away from zero (its bits plus one), the largest finite values, zeros, infinities and NaN
// kept. A warp a 32-group, as act_quant.
extern "C" __global__ void tf_dsn_qgate_body_kernel(float* __restrict__ x, int64_t n, int mode) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;  // (n whole groups: a warp's lanes are all in or all out)
    const float v = x[i];
    if (mode == 2) {
        const float b = bf16r(v);
        float amax = fabsf(b);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float sc = round_scale(fmaxf(amax, 1e-4f));
        x[i] = bf16r(from_e4m3(to_e4m3(b / sc)) * sc);
    } else if (mode == 3) {
        const uint32_t u = __float_as_uint(v);
        const uint32_t mag = u & 0x7fffffffu;
        if (mag != 0u && mag < 0x7f7fffffu) x[i] = __uint_as_float(u + 1u);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The attention epilogue (TF_DS_ATTN_EPILOGUE): rope_heads' inverse RoPE of each head's last RD columns, in place,
// and act_quant's codes and scales of the roped bf16 values (or its fake rows), in one pass. A thread takes eight
// consecutive values of a row (a 32-value group is four lanes), so a RoPE pair never leaves its thread and a lane has
// 16 bytes of o in flight; the RoPE is rope_heads' compiled contraction on the same bf16 inputs (re = fma(xe, cs,
// -(xo sn)), im = fma(xe, sn, xo cs), sn negated for INV) and the same bf16 rounding. RD and HD - RD are multiples of
// the scale block, so a group is wholly inside or outside the roped columns. `store_o` 0 leaves the roped columns out
// of o (its readers are the codes or fake rows). Grid (M, ceil(H * HD / 8 / 256)), 256 threads. The group's scale is a
// power of two, so v / sc is v times its exact reciprocal (the same correctly rounded value); a scale outside the
// range where that reciprocal is a normal float (an infinite or NaN group) divides.
extern "C" __global__ void __launch_bounds__(256) tf_dsn_attn_epilogue_kernel(
    __nv_bfloat16* __restrict__ o, const float* __restrict__ cos, const float* __restrict__ sin,
    const int64_t* __restrict__ pos, int M, int H, int HD, int RD, uint8_t* __restrict__ q, int64_t ldq,
    float* __restrict__ s, int64_t lds, __nv_bfloat16* __restrict__ fake, int64_t ldf, int store_o) {
    const int K = H * HD;
    const int m = blockIdx.x;
    const int k0 = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    const bool live = m < M && k0 < K;  // K is a multiple of 32: a group is all live or all not
    float v[8];
    uint4 raw = make_uint4(0, 0, 0, 0);
    if (live) raw = *reinterpret_cast<const uint4*>(o + size_t(m) * K + k0);
    const uint32_t rw[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[2 * i] = __uint_as_float(rw[i] << 16);
        v[2 * i + 1] = __uint_as_float(rw[i] & 0xffff0000u);
    }
    const int col = live ? k0 % HD : 0;
    if (live && col >= HD - RD) {
        const int j = (col - (HD - RD)) >> 1;
        const int64_t p = pos[m];
        const float4 c4 = *reinterpret_cast<const float4*>(cos + p * (RD / 2) + j);
        const float4 s4 = *reinterpret_cast<const float4*>(sin + p * (RD / 2) + j);
        const float cs[4] = {c4.x, c4.y, c4.z, c4.w};
        const float sv[4] = {s4.x, s4.y, s4.z, s4.w};
        uint32_t ow[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float xe = v[2 * i], xo = v[2 * i + 1], c = cs[i], sn = -sv[i];
            const __nv_bfloat16 re = __float2bfloat16_rn(__fmaf_rn(xe, c, -(xo * sn)));
            const __nv_bfloat16 im = __float2bfloat16_rn(__fmaf_rn(xe, sn, xo * c));
            v[2 * i] = __bfloat162float(re);
            v[2 * i + 1] = __bfloat162float(im);
            ow[i] = uint32_t(__bfloat16_as_ushort(re)) | uint32_t(__bfloat16_as_ushort(im)) << 16;
        }
        if (store_o) *reinterpret_cast<uint4*>(o + size_t(m) * K + k0) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
    float amax = fabsf(v[0]);
#pragma unroll
    for (int i = 1; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (!live) return;
    const float sc = round_scale(fmaxf(amax, 1e-4f));
    const uint32_t sb = __float_as_uint(sc);
    const bool recip = sb - 0x00800000u < 0x7f000000u - 0x00800000u;  // sc = 2^e, e in [-126, 126]
    const float rc = __uint_as_float(0x7f000000u - sb);
    uint8_t c8[8];
    if (recip) {
#pragma unroll
        for (int i = 0; i < 8; i++) c8[i] = to_e4m3(v[i] * rc);
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) c8[i] = to_e4m3(v[i] / sc);
    }
    if (fake) {
        uint32_t fw[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t a = __bfloat16_as_ushort(__float2bfloat16_rn(from_e4m3(c8[2 * i]) * sc));
            const uint32_t b = __bfloat16_as_ushort(__float2bfloat16_rn(from_e4m3(c8[2 * i + 1]) * sc));
            fw[i] = a | b << 16;
        }
        *reinterpret_cast<uint4*>(fake + size_t(m) * ldf + k0) = make_uint4(fw[0], fw[1], fw[2], fw[3]);
    } else {
        uint32_t qw[2];
#pragma unroll
        for (int i = 0; i < 2; i++) qw[i] = uint32_t(c8[4 * i]) | uint32_t(c8[4 * i + 1]) << 8 | uint32_t(c8[4 * i + 2]) << 16 | uint32_t(c8[4 * i + 3]) << 24;
        *reinterpret_cast<uint2*>(q + size_t(m) * ldq + k0) = make_uint2(qw[0], qw[1]);
        if ((threadIdx.x & 3) == 0) s[size_t(m) * lds + (k0 >> 5)] = sc;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The MX GEMM: y [M, N] = sum over K blocks kb of (q[m, kb] . w[n, kb]) * s[m, kb] * ws(n, kb).
// A CTA is 4 warps; a warp owns 32 output columns (four m16n8 tiles) and MT m16 row tiles, over its K split's blocks.
// Within a 32-wide block the K order is permuted the same way for both operands (each thread reads 8 contiguous bytes
// of a row: its 4 low values feed the MMA's k = 4t.., its 4 high ones k = 16 + 4t..), a fixed choice, so a dot product
// is the same sum for every launch.
// Weights: fmt 0, FP8 [N, K] bytes (row stride ldw) with UE8M0 scales [N/32, K/32] (row stride ldws); fmt 1, MXFP4
// [N, K/2] packed (row stride ldw bytes) with UE8M0 scales [N, K/32] (row stride ldws).
// Rows: q [M, K] E4M3 (row stride ldq) with s [M, K/32] fp32 (row stride lds); `rows` (optional) maps the launch's row
// i to the activation row rows[i] (an expert's gathered rows).
// Split K: blockIdx.y is the split (of `splits`, each `kblocks` K blocks); with splits > 1 the partial goes to
// z [splits, M, N] fp32 and tf_dsn_reduce sums it in split order; else the epilogue writes y directly.
struct MxArgs {
    const uint8_t* q;
    const float* s;
    const int* rows;
    int64_t ldq, lds;
    const uint8_t* w;
    const uint8_t* ws;
    int64_t ldw, ldws;
    int fmt;
    int M, N, K;
    int splits, kblocks;
    float* z;
    void* y;
    int64_t ldy;
    int y_dtype;
    int accumulate;  // y += result (fp32 y only)
    int raster;      // the prefill tile GEMMs' grouped tile order (pf_tile_of): row tiles a group, 0: row tiles fastest
};

// The (row tile, column tile) a CTA of a prefill GEMM's grid (gridDim.x row tiles, gridDim.y column tiles) works on.
// raster 0: blockIdx.x, blockIdx.y (row tiles fastest). raster G: the grid is walked in groups of G row tiles, each
// group's row tiles fastest across all the column tiles, then the next group; the last group takes the remaining row
// tiles. Every (row, column) tile is visited once and a tile's content does not depend on when it runs, so the output
// bytes are the default order's. Out of line: inlined, it moves a few kernels' registers past their cap (spills).
__device__ __noinline__ int2 pf_tile_pos(int raster) {
    if (raster <= 0) return make_int2(blockIdx.x, blockIdx.y);
    const int nrow = gridDim.x, ncol = gridDim.y;
    const int lin = blockIdx.x + nrow * blockIdx.y;
    const int gsize = raster * ncol;
    const int gid = lin / gsize;
    const int r0 = gid * raster;
    const int rows = min(raster, nrow - r0);
    const int w = lin - gid * gsize;
    return make_int2(r0 + w % rows, w / rows);
}
__device__ __forceinline__ void pf_tile_of(int raster, int& row, int& col) {
    const int2 t = pf_tile_pos(raster);
    row = t.x;
    col = t.y;
}

template <int MT>
__device__ __forceinline__ void mx_tile(const MxArgs& a, int n0, int m0, int split, float (&acc)[MT][4][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kb0 = split * a.kblocks;
    const int kb1 = min(kb0 + a.kblocks, a.K / kBlk);
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
    // this thread's activation rows (g and g + 8 of each m16 tile), clamped (rows past M read row 0 and are not stored)
    int arow[MT][2];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int r = m0 + i * 16 + g + h * 8;
            r = r < a.M ? r : 0;
            arow[i][h] = a.rows ? a.rows[r] : r;
        }
#pragma unroll 2
    for (int kb = kb0; kb < kb1; kb++) {
        const int k = kb * kBlk + t * 8;
        // B: the four n8 tiles' rows n0 + 8j + g, 8 values from k (FP8: 8 bytes; FP4: 4 packed bytes)
        uint32_t b[4][2];
        float wsc[4][2];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + j * 8 + g;
            if (a.fmt == 0) {
                const uint2 v = *reinterpret_cast<const uint2*>(a.w + size_t(n) * a.ldw + k);
                b[j][0] = v.x;
                b[j][1] = v.y;
            } else {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(a.w + size_t(n) * a.ldw + k / 2);
                b[j][0] = fp4x4_to_e4m3(uint16_t(v & 0xffff));
                b[j][1] = fp4x4_to_e4m3(uint16_t(v >> 16));
            }
            // the C fragment's columns n0 + 8j + 2t, + 1: their weight scales for this block
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int nc = n0 + j * 8 + 2 * t + c;
                wsc[j][c] = a.fmt == 0 ? ue8m0(a.ws[size_t(nc / kBlk) * a.ldws + kb]) : ue8m0(a.ws[size_t(nc) * a.ldws + kb]);
            }
        }
#pragma unroll
        for (int i = 0; i < MT; i++) {
            uint32_t af[4];
            const uint2 lo = *reinterpret_cast<const uint2*>(a.q + size_t(arow[i][0]) * a.ldq + k);
            const uint2 hi = *reinterpret_cast<const uint2*>(a.q + size_t(arow[i][1]) * a.ldq + k);
            af[0] = lo.x;  // row g, k 4t..
            af[1] = hi.x;  // row g + 8
            af[2] = lo.y;  // row g, k 16 + 4t..
            af[3] = hi.y;
            const float sa0 = a.s[size_t(arow[i][0]) * a.lds + kb];
            const float sa1 = a.s[size_t(arow[i][1]) * a.lds + kb];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float d[4];
                mma_e4m3(d, af, b[j]);
                // kernel.py: accum += C_local * scale_a * scale_b (powers of two: exact products)
                acc[i][j][0] += d[0] * sa0 * wsc[j][0];
                acc[i][j][1] += d[1] * sa0 * wsc[j][1];
                acc[i][j][2] += d[2] * sa1 * wsc[j][0];
                acc[i][j][3] += d[3] * sa1 * wsc[j][1];
            }
        }
    }
}

template <int MT>
__device__ __forceinline__ void mx_gemm_kernel(MxArgs a) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 32;
    if (n0 >= a.N) return;
    const int split = blockIdx.y;
    for (int m0 = blockIdx.z * MT * 16; m0 < a.M; m0 += gridDim.z * MT * 16) {
        float acc[MT][4][4];
        mx_tile<MT>(a, n0, m0, split, acc);
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int m = m0 + i * 16 + g + (c >> 1) * 8;
                    const int n = n0 + j * 8 + 2 * t + (c & 1);
                    if (m >= a.M) continue;
                    const float v = acc[i][j][c];
                    if (a.splits > 1) {
                        a.z[(size_t(split) * a.M + m) * a.N + n] = v;
                    } else if (a.accumulate) {
                        float* yp = reinterpret_cast<float*>(a.y) + size_t(m) * a.ldy + n;
                        *yp += v;
                    } else {
                        store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v);
                    }
                }
    }
}

// the split partials z [splits, M, N] summed in split order into y (dtype, row stride ldy), or added to fp32 y
extern "C" __global__ void tf_dsn_reduce_kernel(const float* __restrict__ z, int splits, int M, int N, void* y,
                                                int64_t ldy, int y_dtype, int accumulate) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(M) * N) return;
    float v = z[i];
    for (int s = 1; s < splits; s++) v += z[size_t(s) * M * N + i];
    const int m = int(i / N), n = int(i % N);
    if (accumulate) reinterpret_cast<float*>(y)[size_t(m) * ldy + n] += v;
    else store_y(y, y_dtype, size_t(m) * ldy + n, v);
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm1_kernel(MxArgs a) { mx_gemm_kernel<1>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm2_kernel(MxArgs a) { mx_gemm_kernel<2>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm4_kernel(MxArgs a) { mx_gemm_kernel<4>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm8_kernel(MxArgs a) { mx_gemm_kernel<8>(a); }

// ---------------------------------------------------------------------------------------------------------------------
// Grouped experts (MoE). The routed picks pick [R, S] (expert ids; >= E or < 0: not this node's) with weights
// wts [R, S] fp32 are grouped by expert: count, offsets, and each expert's (row, slot) entries.
//   tf_dsn_group_count: cnt[E] (zeroed by the caller) += picks
//   tf_dsn_group_scan:  off[E + 1] exclusive prefix, act[] the experts with a pick in id order, nact[0] their count
//   tf_dsn_group_place: ent[off[e] + i] = r * S + s and erow[off[e] + i] = r (in no particular order within an expert:
//                       a row's arithmetic is its own, so the order changes no bits)
extern "C" __global__ void tf_dsn_group_count_kernel(const int* __restrict__ pick, int RS, int E, int* __restrict__ cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RS) return;
    const int e = pick[i];
    if (e >= 0 && e < E) atomicAdd(cnt + e, 1);
}

extern "C" __global__ void tf_dsn_group_scan_kernel(const int* __restrict__ cnt, int E, int* __restrict__ off,
                                                    int* __restrict__ fill, int* __restrict__ act, int* __restrict__ nact) {
    // one thread: E is a few hundred
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int o = 0, n = 0;
    for (int e = 0; e < E; e++) {
        off[e] = o;
        fill[e] = 0;
        if (cnt[e]) act[n++] = e;
        o += cnt[e];
    }
    off[E] = o;
    nact[0] = n;
}

// tf_dsn_group_scan's outputs and tf_dsn_pf_tiles' tile list in one launch (TF_DS_GROUP_TILES): off[E + 1] the counts'
// exclusive prefix, fill[E] zeros, act[] the experts with a pick in id order and nact[0] their count, then each active
// expert's ceil(cnt / bm) tiles (expert, first row) at its exclusive prefix of tile counts (id order: act order) and
// ntiles[0] their count: tiles_of's list, entry for entry. One CTA; warp 0 takes the experts in steps of 32 (two
// warp prefixes a step), then every thread writes its experts' tiles. Dynamic shared memory: 4 E bytes.
extern "C" __global__ void __launch_bounds__(1024) tf_dsn_group_tiles_kernel(const int* __restrict__ cnt, int E, int bm,
                                                                             int* __restrict__ off, int* __restrict__ fill,
                                                                             int* __restrict__ act, int* __restrict__ nact,
                                                                             int2* __restrict__ tiles,
                                                                             int* __restrict__ ntiles) {
    extern __shared__ int first[];  // [E]: each expert's first tile
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        int base = 0, nb = 0, tb = 0;
        for (int e0 = 0; e0 < E; e0 += 32) {
            const int e = e0 + lane;
            const int c = e < E ? cnt[e] : 0;
            const int nt = (c + bm - 1) / bm;
            int x = c, y = nt;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int xu = __shfl_up_sync(0xffffffffu, x, o);
                const int yu = __shfl_up_sync(0xffffffffu, y, o);
                if (lane >= o) {
                    x += xu;
                    y += yu;
                }
            }
            if (e < E) {
                off[e] = base + x - c;
                fill[e] = 0;
                first[e] = tb + y - nt;
            }
            const unsigned m = __ballot_sync(0xffffffffu, c != 0);
            if (c != 0) act[nb + __popc(m & ((1u << lane) - 1))] = e;
            nb += __popc(m);
            base += __shfl_sync(0xffffffffu, x, 31);
            tb += __shfl_sync(0xffffffffu, y, 31);
        }
        if (lane == 0) {
            off[E] = base;
            nact[0] = nb;
            ntiles[0] = tb;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const int c = cnt[e];
        for (int j = 0; j * bm < c; j++) tiles[first[e] + j] = make_int2(e, j * bm);
    }
}

extern "C" __global__ void tf_dsn_group_place_kernel(const int* __restrict__ pick, int RS, int S, int E,
                                                     const int* __restrict__ off, int* __restrict__ fill,
                                                     int* __restrict__ ent, int* __restrict__ erow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RS) return;
    const int e = pick[i];
    if (e < 0 || e >= E) return;
    const int at = off[e] + atomicAdd(fill + e, 1);
    ent[at] = i;
    erow[at] = i / S;
}

// Gate / up of the grouped experts with the SwiGLU epilogue (model.py Expert.forward): for expert act[blockIdx.y]'s
// entries and a 32-column tile of the node's pack of its intermediate (its main blocks, then its rest block: tile t is
// the pack's rows and the output columns 32 t ..), g = bf16(x . w1), u = bf16(x . w3), then
// h = bf16(silu(min(g, limit)) * clamp(u, -limit, limit) * wts[entry]) (wts null: the shared expert, no weight) into
// h [entry, out column] (row stride ldh). x is quantized already (q, s: one row per routed row, erow[] of an entry).
// w1 / w3 / their scales: an expert e's tensors at w1[e] etc. (pointer tables), so experts may sit anywhere.
struct GuArgs {
    const uint8_t* q;
    const float* s;
    int64_t ldq, lds;
    const int* ent;     // [entries]: r * S + s
    const int* erow;    // [entries]: r, the entry's activation row
    const int* off;     // [E + 1]
    const int* act;     // [E]: active experts
    const int* nact;    // [1]
    const uint64_t* w1; // [E] pointers: MXFP4 [I, D/2] (fmt 1) or FP8 [I, D] (fmt 0)
    const uint64_t* w3;
    const uint64_t* s1; // [E] pointers: their UE8M0 scales
    const uint64_t* s3;
    int64_t ldw, ldws;  // the routed experts' strides (fmt)
    int fmt;
    int D;
    int shared;         // the FP8 expert's id (the shared expert), -1: none
    int64_t sldw, sldws; // its strides
    const int* owner;   // optional [E]: the last rest_tiles tiles (the rest block) only for experts with owner[e] != 0
    int n_tiles, rest_tiles;
    const float* wts;   // [R * S] or null
    float limit;
    __nv_bfloat16* h;   // the pack: main columns [P, ldh] (P entries), then with rest tiles their columns [P, rest]
    int64_t ldh;
    int P;              // the entries (r * S), where the rest columns start (P * ldh elements in)
    // split K (decode): splits > 1 writes each split's g and u partials to z [splits, 2, P, n_tiles * 32] fp32 and
    // tf_dsn_gu_finish sums them in split order and applies the epilogue; splits 1: the epilogue here
    int splits, kblocks;
    float* z;
    int early_exit;     // Track C (TF_DS_EXPERT_EXIT): the decode kernel's rows past the active experts leave at once
};

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + expf(-v)); }

// gate / up's epilogue for one (entry, column): g and u (fp32 sums) into the pack
__device__ __forceinline__ void gu_store(const GuArgs& a, int entry, int col, float g_acc, float u_acc) {
    // fp4_gemm / fp8_gemm write bf16; Expert.forward takes .float() of them
    float gv = bf16r(g_acc), uv = bf16r(u_acc);
    if (a.limit > 0.f) {
        uv = fminf(fmaxf(uv, -a.limit), a.limit);
        gv = fminf(gv, a.limit);
    }
    float hv = silu_f(gv) * uv;
    if (a.wts) hv = hv * a.wts[entry];
    const int main = (a.n_tiles - a.rest_tiles) * 32;
    // the pack as the 2D exchange carries it: every entry's main columns, then every entry's rest
    const size_t at = col < main ? size_t(entry) * a.ldh + col
                                 : size_t(a.P) * a.ldh + size_t(entry) * (a.rest_tiles * 32) + (col - main);
    a.h[at] = __float2bfloat16_rn(hv);
}

template <int MT>
__device__ __forceinline__ void gu_kernel(GuArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    if (tile >= a.n_tiles) return;
    if (a.owner && tile >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const int c0 = tile * 32;
    const bool fp8 = e == a.shared;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += MT * 16) {
        float ag[MT][4][4], au[MT][4][4];
        MxArgs m{};
        m.q = a.q; m.s = a.s; m.ldq = a.ldq; m.lds = a.lds;
        m.ldw = fp8 ? a.sldw : a.ldw; m.ldws = fp8 ? a.sldws : a.ldws; m.fmt = fp8 ? 0 : a.fmt;
        const int split = blockIdx.z;
        m.M = cnt; m.K = a.D; m.splits = a.splits; m.kblocks = a.splits > 1 ? a.kblocks : a.D / kBlk;
        m.rows = a.erow + base;  // the entries' activation rows
        m.w = reinterpret_cast<const uint8_t*>(a.w1[e]);
        m.ws = reinterpret_cast<const uint8_t*>(a.s1[e]);
        mx_tile<MT>(m, c0, m0, split, ag);
        m.w = reinterpret_cast<const uint8_t*>(a.w3[e]);
        m.ws = reinterpret_cast<const uint8_t*>(a.s3[e]);
        mx_tile<MT>(m, c0, m0, split, au);
        const int ncols = a.n_tiles * 32;
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int mi = m0 + i * 16 + g + (c >> 1) * 8;
                    if (mi >= cnt) continue;
                    const int entry = a.ent[base + mi];
                    const int col = c0 + j * 8 + 2 * t + (c & 1);
                    if (a.splits > 1) {
                        float* zs = a.z + size_t(split) * 2 * a.P * ncols;
                        zs[size_t(entry) * ncols + col] = ag[i][j][c];
                        zs[size_t(a.P + entry) * ncols + col] = au[i][j][c];
                    } else {
                        gu_store(a, entry, col, ag[i][j][c], au[i][j][c]);
                    }
                }
    }
}

// gate / up's split partials summed in split order (the same order for every row count), then the epilogue: one
// thread a column of expert act[blockIdx.y], over its entries (grid: x over the columns, y as tf_dsn_gu)
extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu_finish_kernel(GuArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int ncols = a.n_tiles * 32;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;
    if (a.owner && col / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int mi = 0; mi < cnt; mi++) {
        const int entry = a.ent[base + mi];
        float gs = 0.f, us = 0.f;
        for (int sp = 0; sp < a.splits; sp++) {
            const float* zs = a.z + size_t(sp) * 2 * a.P * ncols;
            gs += zs[size_t(entry) * ncols + col];
            us += zs[size_t(a.P + entry) * ncols + col];
        }
        gu_store(a, entry, col, gs, us);
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu1_kernel(GuArgs a) { gu_kernel<1>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu2_kernel(GuArgs a) { gu_kernel<2>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu4_kernel(GuArgs a) { gu_kernel<4>(a); }

// Down of the grouped experts: for expert act[blockIdx.y]'s entries, y_e = h_q . w2 over the node's output columns
// (`n` of them, 32-column tiles), into out [entry, n] fp32; with round_bf16 the bf16 value widened (Expert.forward
// returns bf16), which only a rank holding the whole intermediate can do (a TP rank's is a partial sum).
// q / s: the entries' quantized h (tf_dsn_act_quant of h [entries, I], one row an entry).
struct DnArgs {
    const uint8_t* q;
    const float* s;
    int64_t ldq, lds;
    const int* ent;
    const int* off;
    const int* act;
    const int* nact;
    const uint64_t* w2; // [E]: MXFP4 [n, I/2] or FP8 [n, I] (this node's output columns)
    const uint64_t* s2;
    int64_t ldw, ldws;
    int fmt;
    int shared;         // the FP8 expert's id, -1: none
    int64_t sldw, sldws;
    int I, N;
    int round_bf16;     // 1: each expert's output rounded to bf16 (Expert.forward's dtype; a whole intermediate only)
    float* out;
    int64_t ldo;
    // split K (decode): splits > 1 writes each split's partial to z [splits, entries, N] fp32 and tf_dsn_dn_finish sums
    // them in split order into out
    int splits, kblocks;
    float* z;
    int P;              // the entries (z's row count)
    int early_exit;     // Track C (TF_DS_EXPERT_EXIT): as GuArgs'
};

template <int MT>
__device__ __forceinline__ void dn_kernel(DnArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 32;
    if (n0 >= a.N) return;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += MT * 16) {
        float acc[MT][4][4];
        MxArgs m{};
        m.q = a.q; m.s = a.s; m.ldq = a.ldq; m.lds = a.lds;
        m.rows = a.ent + base;
        m.w = reinterpret_cast<const uint8_t*>(a.w2[e]);
        m.ws = reinterpret_cast<const uint8_t*>(a.s2[e]);
        const bool fp8 = e == a.shared;
        m.ldw = fp8 ? a.sldw : a.ldw; m.ldws = fp8 ? a.sldws : a.ldws; m.fmt = fp8 ? 0 : a.fmt;
        const int split = blockIdx.z;
        m.M = cnt; m.K = a.I; m.splits = a.splits; m.kblocks = a.splits > 1 ? a.kblocks : a.I / kBlk;
        mx_tile<MT>(m, n0, m0, split, acc);
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int mi = m0 + i * 16 + g + (c >> 1) * 8;
                    if (mi >= cnt) continue;
                    const int entry = a.ent[base + mi];
                    const float v = acc[i][j][c];
                    const int n = n0 + j * 8 + 2 * t + (c & 1);
                    if (a.splits > 1) a.z[(size_t(split) * a.P + entry) * a.N + n] = v;
                    else a.out[size_t(entry) * a.ldo + n] = a.round_bf16 ? bf16r(v) : v;
                }
    }
}

// down's split partials summed in split order into out: one thread a column of expert act[blockIdx.y], over its
// entries
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn_finish_kernel(DnArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= a.N) return;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int mi = 0; mi < cnt; mi++) {
        const int entry = a.ent[base + mi];
        float v = 0.f;
        for (int sp = 0; sp < a.splits; sp++) v += a.z[(size_t(sp) * a.P + entry) * a.N + n];
        a.out[size_t(entry) * a.ldo + n] = a.round_bf16 ? bf16r(v) : v;
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn1_kernel(DnArgs a) { dn_kernel<1>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn2_kernel(DnArgs a) { dn_kernel<2>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn4_kernel(DnArgs a) { dn_kernel<4>(a); }

// The routed sum (MoE.forward: y[idx] += expert(x[idx]) in expert id order, fp32): y [R, N] fp32 (row stride ldy) =
// (accumulate ? y : 0) + the row's picked entries' out [r * S + s, N] in ascending expert id.
extern "C" __global__ void tf_dsn_combine_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick,
                                                 int R, int S, int E, int N, float* __restrict__ y, int64_t ldy,
                                                 int accumulate) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(R) * N) return;
    const int r = int(i / N), n = int(i % N);
        int order[16];
        int k = 0;
        for (int s = 0; s < S && s < 16; s++) {
            const int e = pick[r * S + s];
            if (e < 0 || e >= E) continue;
            int j = k++;
            while (j > 0 && pick[r * S + order[j - 1]] > e) { order[j] = order[j - 1]; j--; }
            order[j] = s;
        }
        for (int j = 0; j < k; j++) v += out[size_t(r * S + order[j]) * ldo + n];
    y[size_t(r) * ldy + n] = v;
}

// A producer kernel that rings the transport's doorbell itself, once a set of its CTAs ("the sent CTAs", the first `span` in the
// launch) have stored their outputs: every thread fences its stores to the system, the CTA's threads meet, and the CTA's arrival
// is counted in `cnt` (device memory, zero between launches); the last arrival fences again, zeroes the count and stores `ticket`
// (the value a stream write would have stored) into the doorbell line `door` (host-mapped memory the proxy naps on).
struct Door {
    unsigned* cnt;
    unsigned long long* door;
    unsigned long long ticket;
};

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// The CTA's arrival at the count (thread 0), the last one ringing. Returns whether this CTA rang.
__device__ __forceinline__ bool door_arrive(const Door& d, unsigned span) {
    if (atomicAdd(d.cnt, 1u) != span - 1) return false;
    *d.cnt = 0;
    __threadfence_system();
    *reinterpret_cast<volatile unsigned long long*>(d.door) = d.ticket;
    return true;
}

// The CTA's arrival. MODE 0: every thread fences its own stores before the barrier (the sequence that has been shown to hold; each
// thread's fence waits for its stores to drain). MODE 1, the block-level release: the CTA's threads meet (their stores are ordered before
// thread 0 by the barrier), thread 0 fences them to the system once and counts the CTA in. MODE 2: thread 0 counts the CTA in with a
// release at GPU scope, which does not wait for the stores to reach the host; the last arrival, having acquired every earlier
// arrival, fences to the system before it rings, which orders everything it has acquired before the ring. In every mode the last
// arrival zeroes the count and rings as door_arrive does.
template <int MODE>
__device__ __forceinline__ bool door_arrive_cta(const Door& d, unsigned span) {
    if constexpr (MODE == 1) {
        __syncthreads();
        if (threadIdx.x != 0) return false;
        __threadfence_system();
    } else if constexpr (MODE == 0) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x != 0) return false;
    } else {
        __syncthreads();
        if (threadIdx.x != 0) return false;
        unsigned old;
        asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(d.cnt) : "memory");
        if (old != span - 1) return false;
        *d.cnt = 0;
        __threadfence_system();
        *reinterpret_cast<volatile unsigned long long*>(d.door) = d.ticket;
        return true;
    }
    return door_arrive(d, span);
}

// The order of a launch's CTAs by quarter: the quarter of position i (8 bits each in `quarters`), `unit` CTAs a quarter,
// the first `span` CTAs the sent ones.
struct QuarterOrder {
    unsigned quarters;
    unsigned unit;
    unsigned span;
};

// tf_dsn_combine over a launch whose CTAs run the row quarters in `qo`'s order (a CTA is 256 consecutive elements of one row
// block: its place in the launch gives the quarter by position, the offset within it unchanged), and the first qo.span CTAs ring
// the doorbell (Door). R * N a multiple of 256. EARLY: the arrival is counted before the CTA's stores (a hazard: the control
// that must be seen); TIMED: also each CTA's %globaltimer at its begin and end in ts[2 * b], ts[2 * b + 1] and the ring's in
// ts[2 * gridDim.x].
template <bool EARLY, bool TIMED, int BLOCK>
__device__ __forceinline__ void combine_door(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R, int S,
                                             int E, int N, float* __restrict__ y, int64_t ldy, int accumulate, const QuarterOrder& qo,
                                             const Door& d, unsigned long long* ts) {
    const unsigned b = blockIdx.x;
    const bool sent = b < qo.span;
    if constexpr (TIMED) {
        if (threadIdx.x == 0) ts[2 * b] = gtimer();
    }
    if constexpr (EARLY) {
        if (sent && threadIdx.x == 0) door_arrive(d, qo.span);
    }
    if constexpr (EARLY) {
        // the control must be able to see the hazard: a sent CTA stores 300 us after it has been counted, so the CTAs counted last have
        // stored nothing yet when the doorbell shows
        if (sent) {
            if (threadIdx.x == 0) {
                const unsigned long long t0 = gtimer();
                while (gtimer() - t0 < 300000ull) {}
            }
            __syncthreads();
        }
    }
    const unsigned pos = b / qo.unit;
    const unsigned pb = ((qo.quarters >> (8 * pos)) & 0xffu) * qo.unit + (b - pos * qo.unit);
    const int64_t i = int64_t(pb) * blockDim.x + threadIdx.x;
    const int r = int(i / N), n = int(i % N);
        int order[16];
        int k = 0;
        for (int s = 0; s < S && s < 16; s++) {
            const int e = pick[r * S + s];
            if (e < 0 || e >= E) continue;
            int j = k++;
            while (j > 0 && pick[r * S + order[j - 1]] > e) { order[j] = order[j - 1]; j--; }
            order[j] = s;
        }
        for (int j = 0; j < k; j++) v += out[size_t(r * S + order[j]) * ldo + n];
    y[size_t(r) * ldy + n] = v;
    if constexpr (!EARLY) {
        if (sent) {
            if (door_arrive_cta<BLOCK>(d, qo.span)) {
                if constexpr (TIMED) ts[2 * gridDim.x] = gtimer();
            }
        }
    }
    if constexpr (TIMED) {
        __syncthreads();
        if (threadIdx.x == 0) ts[2 * b + 1] = gtimer();
    }
}

// The block-level release (BLOCK) is the engine's; _pt fences every thread's stores; _early is the control; _t stamps.
extern "C" __global__ void tf_dsn_combine_door_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R,
                                                      int S, int E, int N, float* __restrict__ y, int64_t ldy, int accumulate,
                                                      QuarterOrder qo, Door d) {
    combine_door<false, false, 1>(out, ldo, pick, R, S, E, N, y, ldy, accumulate, qo, d, nullptr);
}
extern "C" __global__ void tf_dsn_combine_door_g_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R,
                                                      int S, int E, int N, float* __restrict__ y, int64_t ldy, int accumulate,
                                                      QuarterOrder qo, Door d) {
    combine_door<false, false, 2>(out, ldo, pick, R, S, E, N, y, ldy, accumulate, qo, d, nullptr);
}
extern "C" __global__ void tf_dsn_combine_door_g_early_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R,
                                                            int S, int E, int N, float* __restrict__ y, int64_t ldy, int accumulate,
                                                            QuarterOrder qo, Door d) {
    combine_door<true, false, 2>(out, ldo, pick, R, S, E, N, y, ldy, accumulate, qo, d, nullptr);
}
extern "C" __global__ void tf_dsn_combine_door_pt_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R,
                                                      int S, int E, int N, float* __restrict__ y, int64_t ldy, int accumulate,
                                                      QuarterOrder qo, Door d) {
    combine_door<false, false, 0>(out, ldo, pick, R, S, E, N, y, ldy, accumulate, qo, d, nullptr);
}
extern "C" __global__ void tf_dsn_combine_door_early_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R,
                                                      int S, int E, int N, float* __restrict__ y, int64_t ldy, int accumulate,
                                                      QuarterOrder qo, Door d) {
    combine_door<true, false, 1>(out, ldo, pick, R, S, E, N, y, ldy, accumulate, qo, d, nullptr);
}
extern "C" __global__ void tf_dsn_combine_door_t_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R,
                                                      int S, int E, int N, float* __restrict__ y, int64_t ldy, int accumulate,
                                                      QuarterOrder qo, Door d, unsigned long long* __restrict__ ts) {
    combine_door<false, true, 1>(out, ldo, pick, R, S, E, N, y, ldy, accumulate, qo, d, ts);
}
extern "C" __global__ void tf_dsn_combine_door_pt_t_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick, int R,
                                                      int S, int E, int N, float* __restrict__ y, int64_t ldy, int accumulate,
                                                      QuarterOrder qo, Door d, unsigned long long* __restrict__ ts) {
    combine_door<false, true, 0>(out, ldo, pick, R, S, E, N, y, ldy, accumulate, qo, d, ts);
}

// The replay's observer of a doorbell: one CTA waits for the doorbell line to show `want` (at most timeout_ns), and then reads two
// word ranges (got0 / got1 of `words` each, bypassing the caches) against their references, counting the words that differ in
// out[0]; out[1] says whether the doorbell showed. Launched on a stream beside a producer that rings the doorbell, it sees the
// producer's output exactly as complete as it is when the doorbell shows.
extern "C" __global__ void __launch_bounds__(1024) tf_dsn_door_watch_kernel(const unsigned long long* door, unsigned long long want,
                                                                           const unsigned* ref0, const unsigned* got0,
                                                                           const unsigned* ref1, const unsigned* got1,
                                                                           unsigned long long words, unsigned* out,
                                                                           unsigned long long timeout_ns) {
    __shared__ int seen;
    if (threadIdx.x == 0) {
        const unsigned long long t0 = gtimer();
        int ok = 0;
        while (gtimer() - t0 < timeout_ns) {
            if (*reinterpret_cast<const volatile unsigned long long*>(door) == want) {
                ok = 1;
                break;
            }
        }
        seen = ok;
    }
    __syncthreads();
    unsigned bad = 0;
    if (seen) {
        // from the last word of the last range backwards: the CTAs that ring last are the ones that store last
        for (int w = 1; w >= 0; w--) {
            const unsigned* ref = w ? ref1 : ref0;
            const unsigned* got = w ? got1 : got0;
            for (unsigned long long i = threadIdx.x; i < words; i += blockDim.x) {
                const unsigned long long j = words - 1 - i;
                unsigned g;
                asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(g) : "l"(got + j));
                bad += g != ref[j];
            }
        }
    }
    if (bad) atomicAdd(&out[0], bad);
    if (threadIdx.x == 0) out[1] = seen;
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode GEMV of the experts (a decode round's few rows): a warp per output column, its lanes over the column's 32-wide
// K blocks (one 16-byte load an MXFP4 block, two an FP8 block, and the block's scale), each block decoded once into
// registers and applied to every one of the expert's rows: block sum (32 exact products, in order) * row scale * weight
// scale into the row's fp32 accumulator; the lanes' accumulators summed by a fixed shuffle tree. A row's arithmetic is
// its own, so its bits are the same at every row count. Grid: x the columns / 8 (8 warps a CTA), y the active experts.

constexpr int kGvRows = 8;  // the rows a pass over a column's weights takes (more rows: more passes)

// one weight block of row n at block kb as 32 floats (times nothing: the scale is apart)
__device__ __forceinline__ void gv_load_block(const uint8_t* w, int64_t ldw, int fmt, int n, int kb, float (&v)[32],
                                              const float* lut) {
    if (fmt == 1) {
        const uint4 b = *reinterpret_cast<const uint4*>(w + size_t(n) * ldw + size_t(kb) * 16);
        const uint32_t ws[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) v[i * 8 + j] = lut[(ws[i] >> (4 * j)) & 0xf];
    } else {
        const uint4* p = reinterpret_cast<const uint4*>(w + size_t(n) * ldw + size_t(kb) * 32);
        const uint4 b0 = p[0], b1 = p[1];
        const uint32_t ws[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) v[i * 4 + j] = from_e4m3(uint8_t(ws[i] >> (8 * j)));
    }
}

// the weight scale of row n, block kb (fp8: one per 32 x 32 block; fp4: one per 32 values of a row)
__device__ __forceinline__ float gv_wscale(const uint8_t* ws, int64_t ldws, int fmt, int n, int kb) {
    return fmt == 1 ? ue8m0(ws[size_t(n) * ldws + kb]) : ue8m0(ws[size_t(n / kBlk) * ldws + kb]);
}

// a block's dot product with an activation row's block (q codes at qrow, 32 bytes), in order
__device__ __forceinline__ float gv_dot(const float (&v)[32], const uint8_t* qrow) {
    const uint4* p = reinterpret_cast<const uint4*>(qrow);
    const uint4 a0 = p[0], a1 = p[1];
    const uint32_t as[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) d = fmaf(v[i * 4 + j], from_e4m3(uint8_t(as[i] >> (8 * j))), d);
    return d;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void gv_lut(float* lut) {
    // E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (convert.py FP4_TABLE)
    if (threadIdx.x < 16) {
        const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        lut[threadIdx.x] = threadIdx.x < 8 ? t[threadIdx.x] : -t[threadIdx.x - 8];
    }
    __syncthreads();
}

// a dense FP8 / FP4 linear as tf_dsn_mx_gemm (splits unused): y [M, N] of a decode launch's rows
extern "C" __global__ void __launch_bounds__(256) tf_dsn_mx_gemv_kernel(MxArgs a) {
    __shared__ float lut[16];
    gv_lut(lut);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= a.N) return;
    const int nkb = a.K / kBlk;
    for (int r0 = 0; r0 < a.M; r0 += kGvRows) {
        const int nr = min(kGvRows, a.M - r0);
        float acc[kGvRows];
#pragma unroll
        for (int i = 0; i < kGvRows; i++) acc[i] = 0.f;
        for (int kb = lane; kb < nkb; kb += 32) {
            float v[32];
            gv_load_block(a.w, a.ldw, a.fmt, n, kb, v, lut);
            const float sw = gv_wscale(a.ws, a.ldws, a.fmt, n, kb);
#pragma unroll
            for (int i = 0; i < kGvRows; i++) {
                if (i >= nr) break;
                const int row = a.rows ? a.rows[r0 + i] : r0 + i;
                acc[i] += gv_dot(v, a.q + size_t(row) * a.ldq + size_t(kb) * kBlk) * a.s[size_t(row) * a.lds + kb] * sw;
            }
        }
#pragma unroll
        for (int i = 0; i < kGvRows; i++) {
            if (i >= nr) break;
            const float v = warp_sum(acc[i]);
            if (lane != 0) continue;
            const int m = r0 + i;
            if (a.accumulate) reinterpret_cast<float*>(a.y)[size_t(m) * a.ldy + n] += v;
            else store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v);
        }
    }
}

// gate / up as tf_dsn_gu (the same arguments; splits unused): h of every entry of expert act[blockIdx.y]
extern "C" __global__ void __launch_bounds__(256) tf_dsn_gu_gemv_kernel(GuArgs a) {
    __shared__ float lut[16];
    gv_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.x * 8 + warp;
    const int ncols = a.n_tiles * 32;
    if (col >= ncols) return;
    if (a.owner && col / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const uint8_t* w1 = reinterpret_cast<const uint8_t*>(a.w1[e]);
    const uint8_t* w3 = reinterpret_cast<const uint8_t*>(a.w3[e]);
    const uint8_t* s1 = reinterpret_cast<const uint8_t*>(a.s1[e]);
    const uint8_t* s3 = reinterpret_cast<const uint8_t*>(a.s3[e]);
    const int nkb = a.D / kBlk;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r0 = 0; r0 < cnt; r0 += kGvRows) {
        const int nr = min(kGvRows, cnt - r0);
        float ag[kGvRows], au[kGvRows];
#pragma unroll
        for (int i = 0; i < kGvRows; i++) ag[i] = au[i] = 0.f;
        for (int kb = lane; kb < nkb; kb += 32) {
            float vg[32], vu[32];
            gv_load_block(w1, ldw, fmt, col, kb, vg, lut);
            gv_load_block(w3, ldw, fmt, col, kb, vu, lut);
            const float sg = gv_wscale(s1, ldws, fmt, col, kb), su = gv_wscale(s3, ldws, fmt, col, kb);
#pragma unroll
            for (int i = 0; i < kGvRows; i++) {
                if (i >= nr) break;
                const int row = a.erow[base + r0 + i];
                const uint8_t* qr = a.q + size_t(row) * a.ldq + size_t(kb) * kBlk;
                const float sa = a.s[size_t(row) * a.lds + kb];
                ag[i] += gv_dot(vg, qr) * sa * sg;
                au[i] += gv_dot(vu, qr) * sa * su;
            }
        }
#pragma unroll
        for (int i = 0; i < kGvRows; i++) {
            if (i >= nr) break;
            const float g = warp_sum(ag[i]), u = warp_sum(au[i]);
            if (lane == 0) gu_store(a, a.ent[base + r0 + i], col, g, u);
        }
    }
}

// down as tf_dsn_dn (the same arguments; splits unused): out of every entry of expert act[blockIdx.y]
extern "C" __global__ void __launch_bounds__(256) tf_dsn_dn_gemv_kernel(DnArgs a) {
    __shared__ float lut[16];
    gv_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= a.N) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const uint8_t* w2 = reinterpret_cast<const uint8_t*>(a.w2[e]);
    const uint8_t* s2 = reinterpret_cast<const uint8_t*>(a.s2[e]);
    const int nkb = a.I / kBlk;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r0 = 0; r0 < cnt; r0 += kGvRows) {
        const int nr = min(kGvRows, cnt - r0);
        float acc[kGvRows];
#pragma unroll
        for (int i = 0; i < kGvRows; i++) acc[i] = 0.f;
        for (int kb = lane; kb < nkb; kb += 32) {
            float v[32];
            gv_load_block(w2, ldw, fmt, n, kb, v, lut);
            const float sw = gv_wscale(s2, ldws, fmt, n, kb);
#pragma unroll
            for (int i = 0; i < kGvRows; i++) {
                if (i >= nr) break;
                const int row = a.ent[base + r0 + i];
                acc[i] += gv_dot(v, a.q + size_t(row) * a.ldq + size_t(kb) * kBlk) * a.s[size_t(row) * a.lds + kb] * sw;
            }
        }
#pragma unroll
        for (int i = 0; i < kGvRows; i++) {
            if (i >= nr) break;
            const float v = warp_sum(acc[i]);
            if (lane == 0) a.out[size_t(a.ent[base + r0 + i]) * a.ldo + n] = a.round_bf16 ? bf16r(v) : v;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode MMA with the operands swapped ("swap AB"): the tensor core's 16-wide side is 16 weight rows (output columns),
// its 8-wide side up to 8 activation rows, so a decode round's rows share one m16n8k32 MMA a K block and each weight
// row's block is read whole (32 bytes FP8, 16 bytes MXFP4). Both operands take the tile kernel's K order in a block
// (a thread's 8 contiguous bytes: low 4 to k = 4t.., high 4 to k = 16 + 4t..). Every (column, row) of the result is its
// own MMA output, so a row's bits never depend on the other rows. A warp: 16 columns; K blocks in order.

constexpr int kSabRows = 8;

__device__ __forceinline__ void sab_lut(uint8_t* lut) {
    if (threadIdx.x < 16) lut[threadIdx.x] = kE2m1ToE4m3[threadIdx.x];
    __syncthreads();
}

__device__ __forceinline__ uint32_t sab_fp4(const uint8_t* lut, uint16_t p) {
    return uint32_t(lut[p & 0xf]) | uint32_t(lut[(p >> 4) & 0xf]) << 8 | uint32_t(lut[(p >> 8) & 0xf]) << 16 |
           uint32_t(lut[(p >> 12) & 0xf]) << 24;
}

// weight rows n0 + g and n0 + g + 8 at block kb: the A fragment
__device__ __forceinline__ void sab_a(const uint8_t* w, int64_t ldw, int fmt, int n0, int kb, const uint8_t* lut,
                                      uint32_t (&a)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const size_t row = size_t(n0 + g + 8 * h);
        if (fmt == 1) {
            const uint32_t v = *reinterpret_cast<const uint32_t*>(w + row * ldw + size_t(kb) * 16 + t * 4);
            a[h] = sab_fp4(lut, uint16_t(v & 0xffff));
            a[2 + h] = sab_fp4(lut, uint16_t(v >> 16));
        } else {
            const uint2 v = *reinterpret_cast<const uint2*>(w + row * ldw + size_t(kb) * 32 + t * 8);
            a[h] = v.x;
            a[2 + h] = v.y;
        }
    }
}

// one 16-column, up-to-8-row block step: acc (C fragment: columns g, g + 8 by rows 2t, 2t + 1) += sum * sa * sw
__device__ __forceinline__ void sab_step(const uint32_t (&a)[4], const uint8_t* qrow, float sa0, float sa1, float sw0,
                                         float sw1, float (&acc)[4]) {
    const int t = threadIdx.x & 3;
    const uint2 bv = *reinterpret_cast<const uint2*>(qrow + t * 8);
    const uint32_t b[2] = {bv.x, bv.y};
    float d[4];
    mma_e4m3(d, a, b);
    acc[0] += d[0] * sa0 * sw0;
    acc[1] += d[1] * sa1 * sw0;
    acc[2] += d[2] * sa0 * sw1;
    acc[3] += d[3] * sa1 * sw1;
}

// columns n0 .. n0 + 15 of a weight (w, ws, fmt), K blocks kb0, kb0 + kstep, ... below nkb, for up to 8 rows: row i
// (< nr) is activation row rows_of(i); a thread's B row is g (its lane group), its C rows 2t, 2t + 1
template <typename RowOf>
__device__ __forceinline__ void sab_cols(const uint8_t* w, const uint8_t* ws, int64_t ldw, int64_t ldws, int fmt,
                                         int n0, int kb0, int kstep, int nkb, const uint8_t* q, int64_t ldq,
                                         const float* s, int64_t lds, int nr, RowOf rows_of, const uint8_t* lut,
                                         float (&acc)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    const int rb = rows_of(g < nr ? g : 0);
    const int r0 = rows_of(2 * t < nr ? 2 * t : 0), r1 = rows_of(2 * t + 1 < nr ? 2 * t + 1 : 0);
    const uint8_t* qb = q + size_t(rb) * ldq;
    const int c0 = n0 + g, c1 = n0 + g + 8;
#pragma unroll 4
    for (int kb = kb0; kb < nkb; kb += kstep) {
        uint32_t a[4];
        sab_a(w, ldw, fmt, n0, kb, lut, a);
        const float sw0 = fmt == 1 ? ue8m0(ws[size_t(c0) * ldws + kb]) : ue8m0(ws[size_t(c0 / kBlk) * ldws + kb]);
        const float sw1 = fmt == 1 ? ue8m0(ws[size_t(c1) * ldws + kb]) : ue8m0(ws[size_t(c1 / kBlk) * ldws + kb]);
        sab_step(a, qb + size_t(kb) * kBlk, s[size_t(r0) * lds + kb], s[size_t(r1) * lds + kb], sw0, sw1, acc);
    }
}

constexpr int kSabWarps = 8;  // a CTA's warps: one 16-column group, the K blocks dealt round them

// the CTA's warps' partials summed in warp order (a fixed order): warp 0's lanes end with the sums
__device__ __forceinline__ void sab_reduce(float (&acc)[4], float (*red)[32][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int c = 0; c < 4; c++) red[warp][lane][c] = acc[c];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = 0.f;
            for (int w2 = 0; w2 < kSabWarps; w2++) v += red[w2][lane][c];
            acc[c] = v;
        }
    }
    __syncthreads();
}

// a dense FP8 / FP4 linear (MxArgs; splits unused): y [M, N], a CTA 16 columns, rows 8 at a time
extern "C" __global__ void __launch_bounds__(256) tf_dsn_mx_sab_kernel(MxArgs a) {
    __shared__ uint8_t lut[16];
    __shared__ float red[kSabWarps][32][4];
    sab_lut(lut);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    if (n0 >= a.N) return;
    for (int m0 = 0; m0 < a.M; m0 += kSabRows) {
        const int nr = min(kSabRows, a.M - m0);
        float acc[4];
        sab_cols(a.w, a.ws, a.ldw, a.ldws, a.fmt, n0, warp, kSabWarps, a.K / kBlk, a.q, a.ldq, a.s, a.lds, nr,
                 [&](int i) { return a.rows ? a.rows[m0 + i] : m0 + i; }, lut, acc);
        sab_reduce(acc, red);
        if (warp != 0) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int i = 2 * t + (c & 1);
            if (i >= nr) continue;
            const int m = m0 + i, n = n0 + g + 8 * (c >> 1);
            if (a.accumulate) reinterpret_cast<float*>(a.y)[size_t(m) * a.ldy + n] += acc[c];
            else store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, acc[c]);
        }
    }
}

// gate / up (GuArgs; splits unused): a CTA 16 columns of the pack of expert act[blockIdx.y], its entries 8 at a time
extern "C" __global__ void __launch_bounds__(256) tf_dsn_gu_sab_kernel(GuArgs a) {
    __shared__ uint8_t lut[16];
    __shared__ float red[kSabWarps][32][4];
    sab_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    if (n0 >= a.n_tiles * 32) return;
    if (a.owner && n0 / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += kSabRows) {
        const int nr = min(kSabRows, cnt - m0);
        auto rows_of = [&](int i) { return a.erow[base + m0 + i]; };
        float ag[4], au[4];
        sab_cols(reinterpret_cast<const uint8_t*>(a.w1[e]), reinterpret_cast<const uint8_t*>(a.s1[e]), ldw, ldws, fmt,
                 n0, warp, kSabWarps, a.D / kBlk, a.q, a.ldq, a.s, a.lds, nr, rows_of, lut, ag);
        sab_cols(reinterpret_cast<const uint8_t*>(a.w3[e]), reinterpret_cast<const uint8_t*>(a.s3[e]), ldw, ldws, fmt,
                 n0, warp, kSabWarps, a.D / kBlk, a.q, a.ldq, a.s, a.lds, nr, rows_of, lut, au);
        sab_reduce(ag, red);
        sab_reduce(au, red);
        if (warp != 0) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int i = 2 * t + (c & 1);
            if (i >= nr) continue;
            gu_store(a, a.ent[base + m0 + i], n0 + g + 8 * (c >> 1), ag[c], au[c]);
        }
    }
}

// down (DnArgs; splits unused): a CTA 16 output columns of expert act[blockIdx.y], its entries 8 at a time
extern "C" __global__ void __launch_bounds__(256) tf_dsn_dn_sab_kernel(DnArgs a) {
    __shared__ uint8_t lut[16];
    __shared__ float red[kSabWarps][32][4];
    sab_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    if (n0 >= a.N) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += kSabRows) {
        const int nr = min(kSabRows, cnt - m0);
        float acc[4];
        sab_cols(reinterpret_cast<const uint8_t*>(a.w2[e]), reinterpret_cast<const uint8_t*>(a.s2[e]), ldw, ldws, fmt,
                 n0, warp, kSabWarps, a.I / kBlk, a.q, a.ldq, a.s, a.lds, nr,
                 [&](int i) { return a.ent[base + m0 + i]; }, lut, acc);
        sab_reduce(acc, red);
        if (warp != 0) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int i = 2 * t + (c & 1);
            if (i >= nr) continue;
            const float v = acc[c];
            a.out[size_t(a.ent[base + m0 + i]) * a.ldo + n0 + g + 8 * (c >> 1)] = a.round_bf16 ? bf16r(v) : v;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// bf16 linears with fp32 accumulation (F.linear of bf16 values: the head's fp32 GEMM of bf16 values, the compressor's,
// the indexer's wk and weights_proj, wo_a's einsum per group): y [M, N] = x [M, K] . w [N, K]^T, x bf16 (row stride
// ldx; group g reads x + g * gx and w + g * gw, y + g * gy: wo_a's block-diagonal groups as blockIdx.z), K in
// 16-column MMA steps in order, one warp 32 columns, 16 rows a tile.
struct BfArgs {
    const __nv_bfloat16* x;
    int64_t ldx, gx;
    const __nv_bfloat16* w;
    int64_t ldw, gw;
    int M, N, K;
    void* y;
    int64_t ldy, gy;
    int y_dtype;
    float scale;  // y = acc * scale (1: none); weights_proj's softmax_scale * n_heads^-0.5 after its bf16 rounding
    int round_first; // 1: y = bf16(bf16(acc) * scale) (weights_proj: the linear's bf16 output, then the scale)
    // split K (decode, one group): splits > 1 writes each split's partial to z [splits, M, N] fp32 (blockIdx.z the
    // split) and tf_dsn_bf_reduce sums them in split order, then rounds / scales as below; ksteps: 16-column steps a split
    int splits, ksteps;
    float* z;
    int raster;  // tf_dsn_pf_bf16_kernel's grouped tile order (pf_tile_of), 0: row tiles fastest
};

// the bf16 GEMM's epilogue for one value
__device__ __forceinline__ void bf_store(const BfArgs& a, void* y, int m, int n, float v) {
    if (a.round_first) v = bf16r(v);
    if (a.scale != 1.f) v = v * a.scale;
    store_y(y, a.y_dtype, size_t(m) * a.ldy + n, v);
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_bf16_gemm_kernel(BfArgs a) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 32;
    if (n0 >= a.N) return;
    const int grp = a.splits > 1 ? 0 : blockIdx.z;
    const int split = a.splits > 1 ? blockIdx.z : 0;
    const int k0 = split * a.ksteps * 16;
    const int k1 = a.splits > 1 ? min(a.K, k0 + a.ksteps * 16) : a.K;
    const __nv_bfloat16* x = a.x + size_t(grp) * a.gx;
    const __nv_bfloat16* w = a.w + size_t(grp) * a.gw;
    for (int m0 = blockIdx.y * 16; m0 < a.M; m0 += gridDim.y * 16) {
        float acc[4][4] = {};
        const int r0 = min(m0 + g, a.M - 1), r1 = min(m0 + g + 8, a.M - 1);
#pragma unroll 4
        for (int k = k0; k < k1; k += 16) {
            uint32_t af[4];
            af[0] = *reinterpret_cast<const uint32_t*>(x + size_t(r0) * a.ldx + k + 2 * t);
            af[1] = *reinterpret_cast<const uint32_t*>(x + size_t(r1) * a.ldx + k + 2 * t);
            af[2] = *reinterpret_cast<const uint32_t*>(x + size_t(r0) * a.ldx + k + 8 + 2 * t);
            af[3] = *reinterpret_cast<const uint32_t*>(x + size_t(r1) * a.ldx + k + 8 + 2 * t);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = n0 + j * 8 + g;
                uint32_t b[2];
                b[0] = *reinterpret_cast<const uint32_t*>(w + size_t(n) * a.ldw + k + 2 * t);
                b[1] = *reinterpret_cast<const uint32_t*>(w + size_t(n) * a.ldw + k + 8 + 2 * t);
                mma_bf16(acc[j], af, b);
            }
        }
        char* yb = reinterpret_cast<char*>(a.y);
        const int esz = a.y_dtype == 2 ? 4 : 2;
        void* y = yb + size_t(grp) * a.gy * esz;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + g + (c >> 1) * 8;
                if (m >= a.M) continue;
                const int n = n0 + j * 8 + 2 * t + (c & 1);
                if (a.splits > 1) a.z[(size_t(split) * a.M + m) * a.N + n] = acc[j][c];
                else bf_store(a, y, m, n, acc[j][c]);
            }
    }
}

// the bf16 GEMM's split partials summed in split order, then its epilogue
extern "C" __global__ void tf_dsn_bf_reduce_kernel(BfArgs a) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(a.M) * a.N) return;
    float v = 0.f;
    for (int sp = 0; sp < a.splits; sp++) v += a.z[size_t(sp) * a.M * a.N + i];
    bf_store(a, a.y, int(i / a.N), int(i % a.N), v);
}

// bf16 -> fp16 (round to nearest; the checkpoint's bf16 tensors that a kernel of the EXL3 lane reads as fp16, the
// drafter's Markov head)
extern "C" __global__ void tf_dsn_bf16_to_f16_kernel(const __nv_bfloat16* __restrict__ in, __half* __restrict__ out,
                                                     int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(__bfloat162float(in[i]));
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill tile GEMMs (prompt chunks). The arithmetic of tf_dsn_mx_gemm, tf_dsn_gu and tf_dsn_dn exactly: each 32-wide
// K block's m16n8k32 sum from zero (both operands in mx_tile's K order within the block), times s[m, kb] * ws(n, kb),
// added to the fp32 accumulator, K blocks in order; with a shape's K split (mx.zig splits()) each split's partial is
// summed from zero and the partials added in split order as tf_dsn_reduce adds them. So every output has the tile
// kernels' bits, a row's bits never depend on the other rows, and the engine may pick either kernel for any launch.
// What differs is the schedule: a CTA of 8 warps computes a BM x BN tile (NWT weights at once: gate and up share the
// activation rows), its operands staged through shared memory by cp.async in a STAGES-deep pipeline of KBS K blocks a
// stage (activation rows gathered by a row table: an expert's entries), MXFP4 decoded to E4M3 by byte permutes at the
// fragment load. Grouped experts run over a tile list (expert, first row) built after the grouping, so a skewed expert
// takes as many CTAs as its rows need. (A multi-stage cp.async pipeline in the manner of CUTLASS's sm80 GEMMs; written
// for this kernel, no code copied.)
namespace pf {

__device__ __forceinline__ uint32_t sptr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 8 : 0));
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 4 : 0));
}
// with an L2 eviction policy (createpolicy): streamed weights evict first, reused activation rows last
__device__ __forceinline__ void cp16h(void* dst, const void* src, bool ok, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(sptr(dst)), "l"(src),
                 "r"(ok ? 16 : 0), "l"(pol));
}
__device__ __forceinline__ uint64_t evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_groups() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// four E2M1 codes (16 bits, low nibble first) -> four E4M3 bytes: kE2m1ToE4m3's values (the magnitudes 0, 0.5 .. 6 by
// one byte permute of the eight-byte table, the sign bit moved to bit 7)
__device__ __forceinline__ uint32_t e2m1x4(uint32_t p) {
    const uint32_t mag = __byte_perm(0x3c383000u, 0x4c484440u, p & 0x7777u);
    const uint32_t q = __byte_perm(p, 0u, 0x1100u);  // bytes b0 b0 b1 b1: value i's sign at bit 3 or 7 of byte i
    return mag | ((q << 4) & 0x00800080u) | (q & 0x80008000u);
}

constexpr int kThreads = 256;  // 8 warps a CTA

// A tile's shapes and shared memory: BM rows by BN columns of NWT weights, warps WGM (rows) x WGN (columns), FMT 1
// MXFP4 / 0 FP8 weights, KBS K blocks a stage, STAGES stages. Row strides are padded so a warp's fragment loads hit
// distinct banks (activation / FP8 rows: 8-byte loads; MXFP4 rows: 4-byte loads). THREADS is the block the tile's
// kernels launch: one thread a lane of the WGM x WGN warps, so the staging loops spread a stage over the whole block.
template <int BM_, int BN_, int WGM_, int WGN_, int NWT_, int FMT_, int KBS_, int STAGES_, int SWZ_ = 1>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, WGM = WGM_, WGN = WGN_, NWT = NWT_, FMT = FMT_, KBS = KBS_, STAGES = STAGES_;
    static constexpr bool SWZ = SWZ_ != 0;  // rows unpadded, their 16-byte pieces XOR-swizzled (else padded rows)
    // SWZ_ 2: the 64-byte FP8 rows' four pieces swizzled by the row pair (r >> 1) & 3, so ldmatrix's eight rows of one
    // piece land on eight distinct 16-byte bank groups (SWZ_ 1's swizzle puts rows r and r + 4 on the same one)
    static constexpr bool LDSM = SWZ_ == 2;
    static constexpr int WM = BM / WGM, WN = BN / WGN, MT = WM / 16, NT = WN / 8;
    static constexpr int THREADS = WGM * WGN * 32;
    static constexpr int SA = KBS * 32 + (SWZ ? 0 : 32);
    static constexpr int SB = FMT ? KBS * 16 + (SWZ ? 0 : 16) : KBS * 32 + (SWZ ? 0 : 32);
    static constexpr int A_BYTES = BM * SA;
    static constexpr int B_BYTES = NWT * BN * SB;
    static constexpr int AS_BYTES = BM * KBS * 4;
    static constexpr int WS_BYTES = FMT ? NWT * BN * 4 : 0;
    static constexpr int STAGE = (A_BYTES + B_BYTES + AS_BYTES + WS_BYTES + 127) / 128 * 128;
    static constexpr int SMEM = STAGES * STAGE;
    static_assert(THREADS <= 1024, "a block's warps");
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile");
    static_assert(FMT == 0 || KBS == 4, "MXFP4 scales stage as one 4-byte word a column");
    static_assert(KBS == 2 || KBS == 4, "stage");
    static_assert(!LDSM || (KBS == 2 && FMT == 0 && (WN / 8) % 2 == 0), "the ldmatrix layout: FP8 weights, 2 K blocks a stage, n8 tiles in pairs");
    // a byte row's 16-byte piece c of row r: where it sits. 8-byte fragment rows (activations, FP8 weights: a half
    // warp reads rows r..r+3, two pieces each) and 4-byte ones (MXFP4: a warp reads rows r..r+7, one piece each) land
    // on distinct banks.
    __device__ static __forceinline__ int pa(int r, int c) {
        if (!SWZ) return c;
        if (LDSM) return c ^ ((r >> 1) & 3);
        return KBS == 4 ? c ^ ((r & 3) << 1) : c ^ (((r >> 1) & 1) << 1);
    }
    __device__ static __forceinline__ int pb(int r, int c) {
        if (!SWZ) return c;
        if (FMT) return c ^ ((r >> 1) & 3);
        return pa(r, c);
    }
};

// The operands of one tile: activation codes q / scales s (row strides ldq, lds) of the rows srow[] names (shared
// memory, -1: a padding row), NWT weights w[] / ws[] (row strides ldw, ldws) from column n0.
struct Ops {
    const uint8_t* q;
    const float* s;
    int64_t ldq, lds;
    const uint8_t* w0;  // weight 0 (gate, or the one weight) and 1 (up): scalars, so no local copy is indexed
    const uint8_t* w1;
    const uint8_t* ws0;
    const uint8_t* ws1;
    int64_t ldw, ldws;
    int n0;
    int stream;  // 1: the weights stream through once (grouped experts): they evict first, the activation rows last
    // a second base of the activation rows from K block kb_split on (load_stage<C, true>): the K blocks from kb_split read q1 / s1 at
    // block kb - kb_split, with the strides ldq / lds of the first
    const uint8_t* q1;
    const float* s1;
    int kb_split;
    // a third base from K block kb_split2 on (load_stage / load_a with THREE): block kb - kb_split2 of q2 / s2
    const uint8_t* q2;
    const float* s2;
    int kb_split2;
    int64_t ldq2, lds2;  // the third base's row strides (the first two bases use ldq / lds)
    __device__ __forceinline__ const uint8_t* w(int i) const { return i ? w1 : w0; }
    __device__ __forceinline__ const uint8_t* ws(int i) const { return i ? ws1 : ws0; }
};

// the activation rows' base for K block kb0: the first, from kb_split on the second (TWO), from kb_split2 on the third (THREE)
struct ABase {
    const uint8_t* q;
    const float* s;
    int kb;
    int64_t ldq, lds;
};
template <bool TWO, bool THREE>
__device__ __forceinline__ ABase abase(const Ops& o, int kb0) {
    ABase b{o.q, o.s, kb0, o.ldq, o.lds};
    if constexpr (TWO || THREE) {
        if (kb0 >= o.kb_split) b = ABase{o.q1, o.s1, kb0 - o.kb_split, o.ldq, o.lds};
    }
    if constexpr (THREE) {
        if (kb0 >= o.kb_split2) b = ABase{o.q2, o.s2, kb0 - o.kb_split2, o.ldq2, o.lds2};
    }
    return b;
}

// f(c) for c = tid, tid + T, ... below N (a thread's share of N staging copies): unrolled when the block divides N,
// so the copies' addresses are the thread's first plus constants
template <int N, int T, class F>
__device__ __forceinline__ void for_copies(int tid, F f) {
    if constexpr (N % T == 0) {
#pragma unroll
        for (int k = 0; k < N / T; k++) f(tid + k * T);
    } else {
        for (int c = tid; c < N; c += T) f(c);
    }
}

template <class C, bool TWO = false, bool THREE = false>
__device__ __forceinline__ void load_stage(char* st, const int* srow, const Ops& o, int kb0) {
    const int tid = threadIdx.x;
    const ABase ab = abase<TWO, THREE>(o, kb0);
    const uint8_t* qa = ab.q;
    const float* sa = ab.s;
    const int kba = ab.kb;
    char* A = st;
    char* B = A + C::A_BYTES;
    char* AS = B + C::B_BYTES;
    char* WS = AS + C::AS_BYTES;
    const uint64_t pa = o.stream ? evict_last() : 0, pw = o.stream ? evict_first() : 0;
    constexpr int CPA = C::KBS * 2;  // 16-byte pieces of an activation row's stage
    for_copies<C::BM * CPA, C::THREADS>(tid, [&](int c) {
        const int r = c / CPA, cc = c % CPA;
        const int row = srow[r];
        const bool ok = row >= 0;
        void* dst = A + r * C::SA + C::pa(r, cc) * 16;
        const uint8_t* src = qa + size_t(ok ? row : 0) * ab.ldq + kba * 32 + cc * 16;
        if (o.stream) cp16h(dst, src, ok, pa);
        else cp16(dst, src, ok);
    });
    for_copies<C::BM, C::THREADS>(tid, [&](int r) {
        const int row = srow[r];
        const bool ok = row >= 0;
        const float* src = sa + size_t(ok ? row : 0) * ab.lds + kba;
        if (C::KBS == 4) cp16(AS + r * 16, src, ok);
        else cp8(AS + r * 8, src, ok);
    });
    constexpr int CPB = C::FMT ? C::KBS : C::KBS * 2;
    for_copies<C::NWT * C::BN * CPB, C::THREADS>(tid, [&](int c) {
        const int r = c / CPB, cc = c % CPB;
        const int wi = r / C::BN, col = o.n0 + r % C::BN;
        void* dst = B + r * C::SB + C::pb(r, cc) * 16;
        const uint8_t* src = o.w(wi) + size_t(col) * o.ldw + kb0 * (C::FMT ? 16 : 32) + cc * 16;
        if (o.stream) cp16h(dst, src, true, pw);
        else cp16(dst, src, true);
    });
    if (C::FMT)
        for_copies<C::NWT * C::BN, C::THREADS>(tid, [&](int r) {
            const int wi = r / C::BN, col = o.n0 + r % C::BN;
            cp4(WS + r * 4, o.ws(wi) + size_t(col) * o.ldws + kb0, true);
        });
}

// The tile's K loop over nkb K blocks: acc[wi][i][j][c] (mx_tile's C fragments) for weight wi, this warp's m16 tile i
// and n8 tile j. fold > 0: a K split every `fold` blocks (the partials summed from zero, added in split order).
// mvalid: the tile's rows from mvalid on are padding (their m16 tiles skipped).
// FP8 weights: wss holds the tile's weight scales [NWT][BN / 32][nkb] (fp8_scales filled it; any barrier after).
template <class C>
__device__ __forceinline__ void fp8_scales(uint8_t* wss, const Ops& o, int nkb) {
    constexpr int nb = C::BN / kBlk;
    for (int i = threadIdx.x; i < C::NWT * nb * nkb; i += C::THREADS) {
        const int wi = i / (nb * nkb), r = i % (nb * nkb);
        const int blk = r / nkb, kb = r % nkb;
        wss[i] = o.ws(wi)[size_t(o.n0 / kBlk + blk) * o.ldws + kb];
    }
}
constexpr int kWss = 2048;  // the slab's bytes (2 weights x 4 blocks x 256 K blocks)

template <class C, bool FOLD, bool FULL, bool FAST>
__device__ __forceinline__ void run_tile1(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb,
                                          int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    float part[C::NWT][C::MT][C::NT][4];
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    part[wi][i][j][c] = 0.f;
                    if (FOLD) acc[wi][i][j][c] = 0.f;
                }
    bool first = true;
    const int nst = nkb / C::KBS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) load_stage<C>(smem + s * C::STAGE, srow, o, s * C::KBS);
        commit();
    }
    for (int st = 0; st < nst; st++) {
        wait_groups<C::STAGES - 2>();
        __syncthreads();
        {
            const int nx = st + C::STAGES - 1;
            if (nx < nst) load_stage<C>(smem + (nx % C::STAGES) * C::STAGE, srow, o, nx * C::KBS);
            commit();
        }
        const char* A = smem + (st % C::STAGES) * C::STAGE;
        const char* B = A + C::A_BYTES;
        const float* AS = reinterpret_cast<const float*>(B + C::B_BYTES);
        const uint32_t* WS = reinterpret_cast<const uint32_t*>(B + C::B_BYTES + C::AS_BYTES);
        const int kb0 = st * C::KBS;
        // this thread's activation rows' scales (rows g, g + 8 of each m16 tile) and, MXFP4, its C columns' weight
        // scales (a word of KBS bytes a column)
        float sa[C::MT][2][C::KBS];
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = wm * C::WM + i * 16 + g + h * 8;
#pragma unroll
                for (int kk = 0; kk < C::KBS; kk++) sa[i][h][kk] = AS[r * C::KBS + kk];
            }
        uint32_t wsw[C::NWT][C::NT][2];
        if (C::FMT) {
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int c = 0; c < 2; c++) wsw[wi][j][c] = WS[wi * C::BN + wn * C::WN + j * 8 + 2 * t + c];
        }
#pragma unroll
        for (int kk = 0; kk < C::KBS; kk++) {
            const int kb = kb0 + kk;
            uint32_t af[C::MT][4];
#pragma unroll
            for (int i = 0; i < C::MT; i++) {
                const int r = wm * C::WM + i * 16 + g;
                const int pc = kk * 2 + (t >> 1), po = (t & 1) * 8;
                const uint2 lo = *reinterpret_cast<const uint2*>(A + r * C::SA + C::pa(r, pc) * 16 + po);
                const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * C::SA + C::pa(r + 8, pc) * 16 + po);
                af[i][0] = lo.x;
                af[i][1] = hi.x;
                af[i][2] = lo.y;
                af[i][3] = hi.y;
            }
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++) {
                // every MMA of the K block first (independent, so their latencies overlap), then the scaled sums
                float d[C::MT][C::NT][4];
                float wsc[C::NT][2];
#pragma unroll
                for (int j = 0; j < C::NT; j++) {
                    const int br = wi * C::BN + wn * C::WN + j * 8 + g;
                    uint32_t b[2];
                    if (C::FMT) {
                        const uint32_t v = *reinterpret_cast<const uint32_t*>(B + br * C::SB + C::pb(br, kk) * 16 + t * 4);
                        b[0] = e2m1x4(v & 0xffffu);
                        b[1] = e2m1x4(v >> 16);
                        wsc[j][0] = ue8m0(uint8_t(wsw[wi][j][0] >> (8 * kk)));
                        wsc[j][1] = ue8m0(uint8_t(wsw[wi][j][1] >> (8 * kk)));
                    } else {
                        const uint2 v = *reinterpret_cast<const uint2*>(B + br * C::SB + C::pb(br, kk * 2 + (t >> 1)) * 16 + (t & 1) * 8);
                        b[0] = v.x;
                        b[1] = v.y;
                        // an FP8 scale covers a 32 x 32 block: one for the n8 tile's columns
                        const int blk = (wn * C::WN + j * 8) / kBlk;
                        wsc[j][0] = wsc[j][1] = ue8m0(wss[(wi * (C::BN / kBlk) + blk) * nkb + kb]);
                    }
#pragma unroll
                    for (int i = 0; i < C::MT; i++) {
                        if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                        mma_e4m3(d[i][j], af[i], b);
                    }
                }
                if (!C::FMT && FAST) {
                    // FP8 with every weight scale in [2^-104, 1] (run_tile checked the slab): d * (sa * ws) is the
                    // same exact product as (d * sa) * ws (powers of two, every intermediate a normal float), so the
                    // same FMA result, with one multiply a row and K block instead of one an output
#pragma unroll
                    for (int j = 0; j < C::NT; j++)
#pragma unroll
                        for (int i = 0; i < C::MT; i++) {
                            if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                            const float s0 = sa[i][0][kk] * wsc[j][0], s1 = sa[i][1][kk] * wsc[j][0];
                            part[wi][i][j][0] = fmaf(d[i][j][0], s0, part[wi][i][j][0]);
                            part[wi][i][j][1] = fmaf(d[i][j][1], s0, part[wi][i][j][1]);
                            part[wi][i][j][2] = fmaf(d[i][j][2], s1, part[wi][i][j][2]);
                            part[wi][i][j][3] = fmaf(d[i][j][3], s1, part[wi][i][j][3]);
                        }
                } else
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int i = 0; i < C::MT; i++) {
                        if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                        part[wi][i][j][0] += d[i][j][0] * sa[i][0][kk] * wsc[j][0];
                        part[wi][i][j][1] += d[i][j][1] * sa[i][0][kk] * wsc[j][1];
                        part[wi][i][j][2] += d[i][j][2] * sa[i][1][kk] * wsc[j][0];
                        part[wi][i][j][3] += d[i][j][3] * sa[i][1][kk] * wsc[j][1];
                    }
            }
            if (FOLD && ((kb + 1) % fold == 0 || kb + 1 == nkb)) {
                // a split's end: its partial into the sum (tf_dsn_reduce: z[0], then + z[1], ...), the next from zero
#pragma unroll
                for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                    for (int i = 0; i < C::MT; i++)
#pragma unroll
                        for (int j = 0; j < C::NT; j++)
#pragma unroll
                            for (int c = 0; c < 4; c++) {
                                acc[wi][i][j][c] = first ? part[wi][i][j][c] : acc[wi][i][j][c] + part[wi][i][j][c];
                                part[wi][i][j][c] = 0.f;
                            }
                first = false;
            }
        }
    }
    wait_groups<0>();
    if (!FOLD) {
#pragma unroll
        for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
            for (int i = 0; i < C::MT; i++)
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[wi][i][j][c] = part[wi][i][j][c];
    }
}

// run_tile1 with the row check compiled out when every row of the tile is valid
template <class C, bool FOLD>
__device__ __forceinline__ void run_tile(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb,
                                         int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    // FP8: the fast scaling when every weight scale of the tile is a power of two in [2^-104, 1] (UE8M0 bytes 23..127:
    // with an activation scale in [2^-22, 2^100] no product of the scales leaves the normal range)
    bool fast = false;
    if (!C::FMT) {
        bool ok = true;
        for (int i = threadIdx.x; i < C::NWT * (C::BN / kBlk) * nkb; i += C::THREADS) ok = ok && wss[i] >= 23 && wss[i] <= 127;
        fast = __syncthreads_and(ok);
    }
    if (fast) {
        if (mvalid >= C::BM) run_tile1<C, FOLD, true, true>(smem, srow, wss, o, nkb, fold, mvalid, acc);
        else run_tile1<C, FOLD, false, true>(smem, srow, wss, o, nkb, fold, mvalid, acc);
    } else {
        if (mvalid >= C::BM) run_tile1<C, FOLD, true, false>(smem, srow, wss, o, nkb, fold, mvalid, acc);
        else run_tile1<C, FOLD, false, false>(smem, srow, wss, o, nkb, fold, mvalid, acc);
    }
}

// run_tile1<C, true, FULL, FAST> of FP8 weights for a K split whose `fold` (K blocks a split) and nkb are multiples of
// KBS: each split's stages in an inner loop, its partial summed from zero over its K blocks with run_tile1's expressions
// in the same order, then merged once at the split's end (acc = the first split's partial, then acc + each later one,
// tf_dsn_reduce's order). The MMAs, the scaling and the merge are run_tile1's; only the loop nesting differs, so every
// output has run_tile1's bits.
template <class C, bool FULL, bool FAST>
__device__ __forceinline__ void run_tile_split1(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb,
                                                int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    static_assert(!C::FMT, "FP8 weights");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[wi][i][j][c] = 0.f;
    const int nst = nkb / C::KBS;
    const int sps = fold / C::KBS;  // stages a split
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) load_stage<C>(smem + s * C::STAGE, srow, o, s * C::KBS);
        commit();
    }
    int st = 0;
    for (int split = 0; st < nst; split++) {
        float part[C::NWT][C::MT][C::NT][4];
#pragma unroll
        for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
            for (int i = 0; i < C::MT; i++)
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int c = 0; c < 4; c++) part[wi][i][j][c] = 0.f;
        const int st_end = min(nst, st + sps);
        for (; st < st_end; st++) {
            wait_groups<C::STAGES - 2>();
            __syncthreads();
            {
                const int nx = st + C::STAGES - 1;
                if (nx < nst) load_stage<C>(smem + (nx % C::STAGES) * C::STAGE, srow, o, nx * C::KBS);
                commit();
            }
            const char* A = smem + (st % C::STAGES) * C::STAGE;
            const char* B = A + C::A_BYTES;
            const float* AS = reinterpret_cast<const float*>(B + C::B_BYTES);
            const int kb0 = st * C::KBS;
            float sa[C::MT][2][C::KBS];
#pragma unroll
            for (int i = 0; i < C::MT; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int r = wm * C::WM + i * 16 + g + h * 8;
#pragma unroll
                    for (int kk = 0; kk < C::KBS; kk++) sa[i][h][kk] = AS[r * C::KBS + kk];
                }
#pragma unroll
            for (int kk = 0; kk < C::KBS; kk++) {
                const int kb = kb0 + kk;
                uint32_t af[C::MT][4];
#pragma unroll
                for (int i = 0; i < C::MT; i++) {
                    const int r = wm * C::WM + i * 16 + g;
                    const int pc = kk * 2 + (t >> 1), po = (t & 1) * 8;
                    const uint2 lo = *reinterpret_cast<const uint2*>(A + r * C::SA + C::pa(r, pc) * 16 + po);
                    const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * C::SA + C::pa(r + 8, pc) * 16 + po);
                    af[i][0] = lo.x;
                    af[i][1] = hi.x;
                    af[i][2] = lo.y;
                    af[i][3] = hi.y;
                }
#pragma unroll
                for (int wi = 0; wi < C::NWT; wi++) {
                    // every MMA of the K block first (independent, so their latencies overlap), then the scaled sums
                    float d[C::MT][C::NT][4];
                    float wsc[C::NT][2];
#pragma unroll
                    for (int j = 0; j < C::NT; j++) {
                        const int br = wi * C::BN + wn * C::WN + j * 8 + g;
                        uint32_t b[2];
                        const uint2 v = *reinterpret_cast<const uint2*>(B + br * C::SB + C::pb(br, kk * 2 + (t >> 1)) * 16 + (t & 1) * 8);
                        b[0] = v.x;
                        b[1] = v.y;
                        // an FP8 scale covers a 32 x 32 block: one for the n8 tile's columns
                        const int blk = (wn * C::WN + j * 8) / kBlk;
                        wsc[j][0] = wsc[j][1] = ue8m0(wss[(wi * (C::BN / kBlk) + blk) * nkb + kb]);
#pragma unroll
                        for (int i = 0; i < C::MT; i++) {
                            if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                            mma_e4m3(d[i][j], af[i], b);
                        }
                    }
                    if (FAST) {
                        // run_tile1's fast scaling: every weight scale of the tile in [2^-104, 1] (run_tile_split
                        // checked the slab), one multiply a row and K block
#pragma unroll
                        for (int j = 0; j < C::NT; j++)
#pragma unroll
                            for (int i = 0; i < C::MT; i++) {
                                if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                                const float s0 = sa[i][0][kk] * wsc[j][0], s1 = sa[i][1][kk] * wsc[j][0];
                                part[wi][i][j][0] = fmaf(d[i][j][0], s0, part[wi][i][j][0]);
                                part[wi][i][j][1] = fmaf(d[i][j][1], s0, part[wi][i][j][1]);
                                part[wi][i][j][2] = fmaf(d[i][j][2], s1, part[wi][i][j][2]);
                                part[wi][i][j][3] = fmaf(d[i][j][3], s1, part[wi][i][j][3]);
                            }
                    } else
#pragma unroll
                    for (int j = 0; j < C::NT; j++)
#pragma unroll
                        for (int i = 0; i < C::MT; i++) {
                            if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                            part[wi][i][j][0] += d[i][j][0] * sa[i][0][kk] * wsc[j][0];
                            part[wi][i][j][1] += d[i][j][1] * sa[i][0][kk] * wsc[j][1];
                            part[wi][i][j][2] += d[i][j][2] * sa[i][1][kk] * wsc[j][0];
                            part[wi][i][j][3] += d[i][j][3] * sa[i][1][kk] * wsc[j][1];
                        }
                }
            }
        }
        // the split's end: its partial into the sum (tf_dsn_reduce: z[0], then + z[1], ...)
        if (split == 0) {
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                for (int i = 0; i < C::MT; i++)
#pragma unroll
                    for (int j = 0; j < C::NT; j++)
#pragma unroll
                        for (int c = 0; c < 4; c++) acc[wi][i][j][c] = part[wi][i][j][c];
        } else {
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                for (int i = 0; i < C::MT; i++)
#pragma unroll
                    for (int j = 0; j < C::NT; j++)
#pragma unroll
                        for (int c = 0; c < 4; c++) acc[wi][i][j][c] = acc[wi][i][j][c] + part[wi][i][j][c];
        }
    }
    wait_groups<0>();
}

// MXFP4 tiles (pf_gu's and pf_dn's) with each weight row staged 128 bytes at a time (TF_DS_PF_RUN128): the weights of two
// stages (8 K blocks) in one slot of their own two-slot ring, loaded with every even activation stage (two stages ahead,
// as run_tile1 loads), each row's 128 bytes one run of 16-byte copies; the activations in the usual ring of STAGES
// slots. A slot of weights: NWT x BN rows of 128 bytes (piece q: the pair's stage q / 4, its K block q % 4; the row's
// low three bits turn the pieces, so the 8 rows a fragment load reads sit on distinct banks), then their scales, two
// 4-byte words a column (one a stage).
template <class C>
struct W2 {
    static_assert(C::FMT == 1 && C::KBS == 4, "MXFP4 tiles of 4 K blocks a stage");
    static_assert(C::STAGES >= 2 && C::STAGES <= 3, "the pair ring refills a slot one iteration after its last reader");
    static constexpr int ROW = 2 * C::KBS * 16;  // a weight row's bytes in a slot: two stages' K blocks
    static constexpr int B_BYTES = C::NWT * C::BN * ROW;
    static constexpr int WS_BYTES = C::NWT * C::BN * 8;
    static constexpr int SLOT = (B_BYTES + WS_BYTES + 127) / 128 * 128;
    static constexpr int A_SLOT = (C::A_BYTES + C::AS_BYTES + 127) / 128 * 128;  // an activation stage and its scales
    static constexpr int SMEM = C::STAGES * A_SLOT + 2 * SLOT;
    __device__ static __forceinline__ int piece(int r, int q) { return q ^ (r & 7); }
};

// an activation stage (load_stage's activation rows and scales) into an activation slot
template <class C, bool THREE = false>
__device__ __forceinline__ void load_a(char* st, const int* srow, const Ops& o, int kb0) {
    const int tid = threadIdx.x;
    const ABase ab = abase<false, THREE>(o, kb0);
    char* A = st;
    char* AS = A + C::A_BYTES;
    const uint64_t pa = o.stream ? evict_last() : 0;
    constexpr int CPA = C::KBS * 2;  // 16-byte pieces of an activation row's stage
    for_copies<C::BM * CPA, C::THREADS>(tid, [&](int c) {
        const int r = c / CPA, cc = c % CPA;
        const int row = srow[r];
        const bool ok = row >= 0;
        void* dst = A + r * C::SA + C::pa(r, cc) * 16;
        const uint8_t* src = ab.q + size_t(ok ? row : 0) * ab.ldq + ab.kb * 32 + cc * 16;
        if (o.stream) cp16h(dst, src, ok, pa);
        else cp16(dst, src, ok);
    });
    for_copies<C::BM, C::THREADS>(tid, [&](int r) {
        const int row = srow[r];
        const bool ok = row >= 0;
        cp16(AS + r * 16, ab.s + size_t(ok ? row : 0) * ab.lds + ab.kb, ok);
    });
}

// the weights of K blocks kb0 .. kb0 + 7 (those below nkb) and their scales into a weight slot: a row's 128 bytes from
// 8 consecutive threads
template <class C>
__device__ __forceinline__ void load_b2(char* slot, const Ops& o, int kb0, int nkb) {
    using P = W2<C>;
    const int tid = threadIdx.x;
    char* B = slot;
    char* WS = B + P::B_BYTES;
    const uint64_t pw = o.stream ? evict_first() : 0;
    for_copies<C::NWT * C::BN * 8, C::THREADS>(tid, [&](int c) {
        const int r = c >> 3, q = c & 7;
        if (kb0 + q >= nkb) return;  // a last pair of one stage
        const int wi = r / C::BN, col = o.n0 + r % C::BN;
        void* dst = B + r * P::ROW + P::piece(r, q) * 16;
        const uint8_t* src = o.w(wi) + size_t(col) * o.ldw + (kb0 + q) * 16;
        if (o.stream) cp16h(dst, src, true, pw);
        else cp16(dst, src, true);
    });
    for_copies<C::NWT * C::BN, C::THREADS>(tid, [&](int r) {
        const int wi = r / C::BN, col = o.n0 + r % C::BN;
        const uint8_t* src = o.ws(wi) + size_t(col) * o.ldws + kb0;
        cp4(WS + r * 8, src, true);
        if (kb0 + C::KBS < nkb) cp4(WS + r * 8 + 4, src + C::KBS, true);
    });
}

// run_tile1<C, false, FULL, false> of MXFP4 weights staged by W2: each stage's K blocks, MMAs, conversions, scales and
// sums are run_tile1's in the same order (the weight bytes read from the pair slot at the stage's half), so its bits
template <class C, bool FULL>
__device__ __forceinline__ void run_tile_w128_1(char* smem, const int* srow, const Ops& o, int nkb, int mvalid,
                                                float (&acc)[C::NWT][C::MT][C::NT][4]) {
    using P = W2<C>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    char* aring = smem;
    char* bring = smem + C::STAGES * P::A_SLOT;
    float part[C::NWT][C::MT][C::NT][4];
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) part[wi][i][j][c] = 0.f;
    const int nst = nkb / C::KBS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) {
            load_a<C>(aring + s * P::A_SLOT, srow, o, s * C::KBS);
            if ((s & 1) == 0) load_b2<C>(bring + ((s >> 1) & 1) * P::SLOT, o, s * C::KBS, nkb);
        }
        commit();
    }
    for (int st = 0; st < nst; st++) {
        wait_groups<C::STAGES - 2>();
        __syncthreads();
        {
            const int nx = st + C::STAGES - 1;
            if (nx < nst) {
                load_a<C>(aring + (nx % C::STAGES) * P::A_SLOT, srow, o, nx * C::KBS);
                if ((nx & 1) == 0) load_b2<C>(bring + ((nx >> 1) & 1) * P::SLOT, o, nx * C::KBS, nkb);
            }
            commit();
        }
        const char* A = aring + (st % C::STAGES) * P::A_SLOT;
        const float* AS = reinterpret_cast<const float*>(A + C::A_BYTES);
        const char* B = bring + ((st >> 1) & 1) * P::SLOT;
        const uint32_t* WS = reinterpret_cast<const uint32_t*>(B + P::B_BYTES);
        const int half = st & 1;
        float sa[C::MT][2][C::KBS];
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = wm * C::WM + i * 16 + g + h * 8;
#pragma unroll
                for (int kk = 0; kk < C::KBS; kk++) sa[i][h][kk] = AS[r * C::KBS + kk];
            }
        uint32_t wsw[C::NWT][C::NT][2];
#pragma unroll
        for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 2; c++) wsw[wi][j][c] = WS[(wi * C::BN + wn * C::WN + j * 8 + 2 * t + c) * 2 + half];
#pragma unroll
        for (int kk = 0; kk < C::KBS; kk++) {
            uint32_t af[C::MT][4];
#pragma unroll
            for (int i = 0; i < C::MT; i++) {
                const int r = wm * C::WM + i * 16 + g;
                const int pc = kk * 2 + (t >> 1), po = (t & 1) * 8;
                const uint2 lo = *reinterpret_cast<const uint2*>(A + r * C::SA + C::pa(r, pc) * 16 + po);
                const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * C::SA + C::pa(r + 8, pc) * 16 + po);
                af[i][0] = lo.x;
                af[i][1] = hi.x;
                af[i][2] = lo.y;
                af[i][3] = hi.y;
            }
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++) {
                float d[C::MT][C::NT][4];
                float wsc[C::NT][2];
#pragma unroll
                for (int j = 0; j < C::NT; j++) {
                    const int br = wi * C::BN + wn * C::WN + j * 8 + g;
                    uint32_t b[2];
                    const uint32_t v = *reinterpret_cast<const uint32_t*>(B + br * P::ROW + P::piece(br, half * 4 + kk) * 16 + t * 4);
                    b[0] = e2m1x4(v & 0xffffu);
                    b[1] = e2m1x4(v >> 16);
                    wsc[j][0] = ue8m0(uint8_t(wsw[wi][j][0] >> (8 * kk)));
                    wsc[j][1] = ue8m0(uint8_t(wsw[wi][j][1] >> (8 * kk)));
#pragma unroll
                    for (int i = 0; i < C::MT; i++) {
                        if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                        mma_e4m3(d[i][j], af[i], b);
                    }
                }
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int i = 0; i < C::MT; i++) {
                        if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                        part[wi][i][j][0] += d[i][j][0] * sa[i][0][kk] * wsc[j][0];
                        part[wi][i][j][1] += d[i][j][1] * sa[i][0][kk] * wsc[j][1];
                        part[wi][i][j][2] += d[i][j][2] * sa[i][1][kk] * wsc[j][0];
                        part[wi][i][j][3] += d[i][j][3] * sa[i][1][kk] * wsc[j][1];
                    }
            }
        }
    }
    wait_groups<0>();
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[wi][i][j][c] = part[wi][i][j][c];
}

// run_tile_w128_1 with the row check compiled out when every row of the tile is valid (run_tile's choice for MXFP4)
template <class C>
__device__ __forceinline__ void run_tile_w128(char* smem, const int* srow, const Ops& o, int nkb, int mvalid,
                                              float (&acc)[C::NWT][C::MT][C::NT][4]) {
    if (mvalid >= C::BM) run_tile_w128_1<C, true>(smem, srow, o, nkb, mvalid, acc);
    else run_tile_w128_1<C, false>(smem, srow, o, nkb, mvalid, acc);
}

// run_tile_split1 with run_tile's choices: the fast scaling when the tile's weight scales allow it, the row check
// compiled out when every row of the tile is valid
template <class C>
__device__ __forceinline__ void run_tile_split(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb,
                                               int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    bool ok = true;
    for (int i = threadIdx.x; i < C::NWT * (C::BN / kBlk) * nkb; i += C::THREADS) ok = ok && wss[i] >= 23 && wss[i] <= 127;
    const bool fast = __syncthreads_and(ok);
    if (fast) {
        if (mvalid >= C::BM) run_tile_split1<C, true, true>(smem, srow, wss, o, nkb, fold, mvalid, acc);
        else run_tile_split1<C, false, true>(smem, srow, wss, o, nkb, fold, mvalid, acc);
    } else {
        if (mvalid >= C::BM) run_tile_split1<C, true, false>(smem, srow, wss, o, nkb, fold, mvalid, acc);
        else run_tile_split1<C, false, false>(smem, srow, wss, o, nkb, fold, mvalid, acc);
    }
}

// the tile list of the grouped experts: (expert, first row) for every BM rows of every active expert, in act order
__device__ __forceinline__ void tiles_of(const int* off, const int* act, const int* nact, int bm, int2* tiles,
                                         int* ntiles) {
    int n = 0;
    for (int i = 0; i < nact[0]; i++) {
        const int e = act[i];
        const int cnt = off[e + 1] - off[e];
        for (int m0 = 0; m0 < cnt; m0 += bm) tiles[n++] = make_int2(e, m0);
    }
    ntiles[0] = n;
}

#ifndef PF_GEMM
#define PF_GEMM 128, 128, 2, 4, 1, 0, 2, 3
#endif
#ifndef PF_GEMM_FOLD
// WP2 (docs/prefill-plan.md): Hector's measured tuning for the K-split projections (wq_a / wkv / wq_b and every
// other dense FP8 GEMM that splits, since they share this kernel) -- 4 K-blocks a stage, 3 stages instead of 2
// and 4. Bit-identical to the previous config on real weights (tools/dsv41-zig/native/check_native.py,
// attn.wq_a/wq_b/wkv/wo_b and ffn.shared_experts.w1/w2); docs/prefill-plan-progress.md has the per-op timings.
#define PF_GEMM_FOLD 64, 128, 2, 4, 1, 0, 4, 3
#endif
#ifndef PF_GU4
#define PF_GU4 64, 128, 2, 4, 2, 1, 4, 3
#endif
#ifndef PF_GU8
#define PF_GU8 64, 64, 2, 4, 2, 0, 4, 3
#endif
#ifndef PF_DN4
#define PF_DN4 64, 128, 2, 4, 1, 1, 4, 3
#endif
#ifndef PF_DN8
#define PF_DN8 64, 128, 2, 4, 1, 0, 2, 3
#endif
#ifndef PF_MINB
#define PF_MINB 1
#endif
#ifndef PF_HINT
#define PF_HINT 0
#endif
using GemmCfg = Cfg<PF_GEMM>;          // dense FP8: warps 64 x 32
using GemmFoldCfg = Cfg<PF_GEMM_FOLD>;  // dense FP8 with a K split: warps 32 x 32 (partials and sums)
using GuCfg4 = Cfg<PF_GU4>;            // gate / up MXFP4: warps 32 x 32 of both weights
using GuCfg8 = Cfg<PF_GU8>;            // gate / up of the FP8 shared expert: column sub-tiles of the MXFP4 tile
using DnCfg4 = Cfg<PF_DN4>;            // down MXFP4
using DnCfg8 = Cfg<PF_DN8>;            // down of the FP8 shared expert
// The split FP8 GEMMs on run_tile_split (TF_DS_PF_SPLIT, mx.zig pfSplitCfg): a shape takes one when its fold and its K
// blocks are multiples of the configuration's K blocks a stage
using GemmSplitACfg = Cfg<128, 128, 2, 4, 1, 0, 2, 5>;  // A: 128-row tiles, 2 K blocks a stage, 5 stages
using GemmSplitBCfg = Cfg<64, 128, 2, 4, 1, 0, 4, 3>;   // B: the fold kernel's tile and stages
using GemmSplitC6Cfg = Cfg<64, 128, 2, 4, 1, 0, 2, 6>;  // C6: 2 K blocks a stage, 6 stages
using GemmSplitC7Cfg = Cfg<64, 128, 2, 4, 1, 0, 2, 7>;  // C7: 2 K blocks a stage, 7 stages
// The dense FP8 GEMM without a K split at 5 stages (TF_DS_PF_GEMM5): GemmCfg's tile and K blocks a stage
using GemmFiveCfg = Cfg<128, 128, 2, 4, 1, 0, 2, 5>;
static_assert(GemmFiveCfg::BM == GemmCfg::BM && GemmFiveCfg::BN == GemmCfg::BN && GemmFiveCfg::KBS == GemmCfg::KBS &&
                  GemmFiveCfg::WGM == GemmCfg::WGM && GemmFiveCfg::WGN == GemmCfg::WGN && GemmFiveCfg::SWZ == GemmCfg::SWZ,
              "tf_dsn_pf_gemm's tile, warps and stage width");
static_assert(GuCfg4::BM == 64 && GuCfg8::BM == 64 && DnCfg4::BM == 64 && DnCfg8::BM == 64, "the tile list's rows");
static_assert(GuCfg4::BN % GuCfg8::BN == 0 && DnCfg4::BN == DnCfg8::BN, "the FP8 expert's sub-tiles");
constexpr int kGemmSmem = GemmCfg::SMEM > GemmFoldCfg::SMEM ? GemmCfg::SMEM : GemmFoldCfg::SMEM;
constexpr int kGuSmem = GuCfg4::SMEM > GuCfg8::SMEM ? GuCfg4::SMEM : GuCfg8::SMEM;
constexpr int kDnSmem = DnCfg4::SMEM > DnCfg8::SMEM ? DnCfg4::SMEM : DnCfg8::SMEM;
constexpr int kBm = 64;  // the grouped tiles' rows

extern __shared__ __align__(128) char smem[];  // the stages (dynamic)

// gate / up's epilogue of a tile (gu_store for every valid entry): acc[0] the gate's sums, acc[1] the up's
template <class C>
__device__ __forceinline__ void gu_tile_store(const GuArgs& a, const float (&acc)[2][C::MT][C::NT][4], int n0, int base,
                                              int m0, int cnt) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int i = 0; i < C::MT; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int mi = m0 + wm * C::WM + i * 16 + g + (c >> 1) * 8;
                if (mi >= cnt) continue;
                const int col = n0 + wn * C::WN + j * 8 + 2 * t + (c & 1);
                gu_store(a, a.ent[base + mi], col, acc[0][i][j][c], acc[1][i][j][c]);
            }
}

// down's epilogue of a tile: each valid entry's columns into out (fp32; round_bf16: the bf16 value widened)
template <class C>
__device__ __forceinline__ void dn_tile_store(const DnArgs& a, const float (&acc)[1][C::MT][C::NT][4], int n0, int base,
                                              int m0, int cnt) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int i = 0; i < C::MT; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int mi = m0 + wm * C::WM + i * 16 + g + h * 8;
                if (mi >= cnt) continue;
                const int n = n0 + wn * C::WN + j * 8 + 2 * t;
                float v0 = acc[0][i][j][2 * h], v1 = acc[0][i][j][2 * h + 1];
                if (a.round_bf16) {
                    v0 = bf16r(v0);
                    v1 = bf16r(v1);
                }
                *reinterpret_cast<float2*>(a.out + size_t(a.ent[base + mi]) * a.ldo + n) = make_float2(v0, v1);
            }
}

}  // namespace pf

// The tile list after the grouping (one thread): tiles [P / 64 + E] int2, ntiles [1].
extern "C" __global__ void tf_dsn_pf_tiles_kernel(const int* __restrict__ off, const int* __restrict__ act,
                                                  const int* __restrict__ nact, int2* __restrict__ tiles,
                                                  int* __restrict__ ntiles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) pf::tiles_of(off, act, nact, pf::kBm, tiles, ntiles);
}

// A dense GEMM tile's fp32 result into y as whole coalesced lines (the _st kernels: y lies in host-mapped
// write-combined memory, where the MMA fragment's own stores - a warp's 8 rows of 32 bytes - are partial lines): the tile
// goes through shared memory in row chunks, then consecutive threads write 16 bytes each along a row. The values are the
// accumulators', stored as they are.
template <class C>
__device__ __forceinline__ void stage_store(const MxArgs& a, int m0, int n0, const float (&acc)[1][C::MT][C::NT][4]) {
    constexpr int LD = C::BN + 8;  // floats a staged row
    constexpr int RB = (64 * LD * 4 <= C::SMEM && C::BM % 64 == 0) ? 64 : 32;
    static_assert(RB * LD * 4 <= C::SMEM && C::BM % RB == 0 && C::BN % 4 == 0, "a staged chunk fits the tile's shared memory");
    float* tile = reinterpret_cast<float*>(pf::smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    asm volatile("cp.async.wait_all;" ::: "memory");
    for (int c = 0; c < C::BM / RB; c++) {
        __syncthreads();  // the pipeline's stages (first chunk) and the last chunk's lines are read
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int rl = wm * C::WM + i * 16 + g + h * 8;
                    if (rl / RB != c) continue;
                    *reinterpret_cast<float2*>(tile + (rl % RB) * LD + wn * C::WN + j * 8 + 2 * t) = make_float2(acc[0][i][j][2 * h], acc[0][i][j][2 * h + 1]);
                }
        __syncthreads();
        for (int idx = threadIdx.x; idx < RB * (C::BN / 4); idx += pf::kThreads) {
            const int r = idx / (C::BN / 4), c4 = idx % (C::BN / 4);
            const int m = m0 + c * RB + r;
            if (m >= a.M) continue;
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(a.y) + size_t(m) * a.ldy + n0 + c4 * 4) = *reinterpret_cast<const float4*>(tile + r * LD + c4 * 4);
        }
    }
}

// The dense MX GEMM of a prompt chunk (MxArgs as tf_dsn_mx_gemm, FP8 weights, N a multiple of 128, K of 64): grid
// (N / 128, ceil(M / 128)), 256 threads, kPfGemmSmem dynamic bytes. Splits > 1 fold in the CTA (no partials buffer).
template <class C, bool FOLD, bool SPLIT = false, bool STAGE = false>
__device__ __forceinline__ void pf_gemm(const MxArgs& a) {
    __shared__ int srow[C::BM];
    __shared__ uint8_t wss[pf::kWss];
    int tr, tc;
    pf_tile_of(a.raster, tr, tc);  // row tiles fastest by default: a wave's CTAs share their weight tiles in L2
    const int m0 = tr * C::BM;
    for (int r = threadIdx.x; r < C::BM; r += C::THREADS) {
        const int m = m0 + r;
        srow[r] = m < a.M ? (a.rows ? a.rows[m] : m) : -1;
    }
    pf::Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = o.w1 = a.w; o.ws0 = o.ws1 = a.ws; o.ldw = a.ldw; o.ldws = a.ldws;
    o.n0 = tc * C::BN;
    const int nkb = a.K / kBlk;
    pf::fp8_scales<C>(wss, o, nkb);
    __syncthreads();
    float acc[1][C::MT][C::NT][4];
    if constexpr (SPLIT) pf::run_tile_split<C>(pf::smem, srow, wss, o, nkb, a.kblocks, a.M - m0, acc);
    else pf::run_tile<C, FOLD>(pf::smem, srow, wss, o, nkb, a.kblocks, a.M - m0, acc);
    if constexpr (STAGE) return stage_store<C>(a, m0, o.n0, acc);  // y in write-combined host memory
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
                if (a.accumulate) {
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

// The dense MX GEMM of a prompt chunk (MxArgs as tf_dsn_mx_gemm, FP8 weights, N a multiple of 128, K of 64 and at most
// 8192), 256 threads, kPfGemmSmem dynamic bytes: splits 1, grid (ceil(M / 128), N / 128); splits > 1 (folded in the
// CTA, no partials buffer): tf_dsn_pf_gemm_fold, grid (ceil(M / 64), N / 128).
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gemm_kernel(MxArgs a) { pf_gemm<pf::GemmCfg, false>(a); }
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gemm_fold_kernel(MxArgs a) { pf_gemm<pf::GemmFoldCfg, true>(a); }
// tf_dsn_pf_gemm_fold's arguments and bits on run_tile_split (a shape whose fold and K blocks are multiples of the
// configuration's K blocks a stage): grid (ceil(M / BM), N / 128), the configuration's dynamic bytes (mx.zig pf_split_*)
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitA_kernel(MxArgs a) { pf_gemm<pf::GemmSplitACfg, true, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitB_kernel(MxArgs a) { pf_gemm<pf::GemmSplitBCfg, true, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitC6_kernel(MxArgs a) { pf_gemm<pf::GemmSplitC6Cfg, true, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitC7_kernel(MxArgs a) { pf_gemm<pf::GemmSplitC7Cfg, true, true>(a); }
// tf_dsn_pf_gemm with 5 stages (4 in flight): the same tile, K blocks a stage and K order, so its bits; grid
// (ceil(M / 128), N / 128), mx.zig pf_gemm5_smem dynamic bytes
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm5_kernel(MxArgs a) { pf_gemm<pf::GemmFiveCfg, false>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm5_st_kernel(MxArgs a) { pf_gemm<pf::GemmFiveCfg, false, false, true>(a); }
// The same kernels storing their fp32 tile as whole lines (stage_store), for a y in write-combined host memory (the
// transport's arena); y fp32, 16-byte aligned, ldy a multiple of 4, no accumulate. Their values are the kernels'.
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gemm_st_kernel(MxArgs a) { pf_gemm<pf::GemmCfg, false, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gemm_fold_st_kernel(MxArgs a) { pf_gemm<pf::GemmFoldCfg, true, false, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitA_st_kernel(MxArgs a) { pf_gemm<pf::GemmSplitACfg, true, true, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitB_st_kernel(MxArgs a) { pf_gemm<pf::GemmSplitBCfg, true, true, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitC6_st_kernel(MxArgs a) { pf_gemm<pf::GemmSplitC6Cfg, true, true, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_gemm_splitC7_st_kernel(MxArgs a) { pf_gemm<pf::GemmSplitC7Cfg, true, true, true>(a); }

namespace pf {
// The MXFP4 tiles of TF_DS_PF_RUN128's twins (pf::W2, run_tile_w128): GuCfg4's and DnCfg4's as this file sets them,
// on their own so that a PF_GU4 / PF_DN4 of a sweep or a stage change leaves the twins as they are (W2's pair ring
// holds for at most 3 stages)
using GuR128Cfg = Cfg<64, 128, 2, 4, 2, 1, 4, 3>;
using DnR128Cfg = Cfg<64, 128, 2, 4, 1, 1, 4, 3>;
// the same tiles on 16 warps a block (TF_DS_GEMM_W16): a warp's tile half as tall (MT 2 -> 1) or half as wide
// (NT 4 -> 2), so a warp holds half the C fragments; the tile, its K blocks a stage, its stages and its shared memory
// are the 8-warp ones'. The warp grid is the one whose fragments, scale words and addresses fit 128 registers a
// thread (a 512-thread block's limit), which the 512-thread build's spill count decides.
using GuR128CfgW16 = Cfg<64, 128, 4, 4, 2, 1, 4, 3>;
using DnR128CfgW16 = Cfg<64, 128, 2, 8, 1, 1, 4, 3>;
static_assert(W2<GuR128CfgW16>::SMEM == W2<GuR128Cfg>::SMEM && W2<DnR128CfgW16>::SMEM == W2<DnR128Cfg>::SMEM, "the 16-warp tiles' shared memory");
// the FP8 shared expert's column sub-tiles (GuCfg8, DnCfg8) on 16 warps: the same weights, K blocks and stages, a
// warp's rows halved (MT 2 -> 1) and, for gate / up, its columns halved (BN 64 -> 32, a quarter of the MXFP4 tile's
// columns a sub-tile), so a warp's C fragments quarter; the sub-tile loop takes C4::BN / C::BN of them
using GuCfg8W16 = Cfg<64, 32, 4, 4, 2, 0, 4, 3>;
using DnCfg8W16 = Cfg<64, 128, 4, 4, 1, 0, 2, 3>;
#ifndef PF_SWEEP
static_assert(GuCfg8W16::SMEM <= W2<GuR128CfgW16>::SMEM && DnCfg8W16::SMEM <= W2<DnR128CfgW16>::SMEM, "the 16-warp sub-tiles fit the expert kernels' shared memory");
#endif
static_assert(GuR128Cfg::BM == kBm && DnR128Cfg::BM == kBm && GuR128Cfg::BN % GuCfg8::BN == 0 && DnR128Cfg::BN == DnCfg8::BN,
              "the twins' tiles: the tile list's rows, the FP8 expert's sub-tiles");
#ifndef PF_SWEEP
// the base kernels' tiles, warps and K blocks a stage (their arithmetic in its order); the stage count is each one's own
template <class A, class B>
constexpr bool same_tiles = A::BM == B::BM && A::BN == B::BN && A::WGM == B::WGM && A::WGN == B::WGN && A::NWT == B::NWT &&
                            A::FMT == B::FMT && A::KBS == B::KBS && A::SWZ == B::SWZ;
static_assert(same_tiles<GuR128Cfg, GuCfg4> && same_tiles<DnR128Cfg, DnCfg4>, "the twins' tiles are the base kernels'");
#endif

// Gate / up of the grouped experts for a prompt chunk on the MXFP4 tiles of C4 (W128: their weights staged 128 bytes a
// row, run_tile_w128; else run_tile); the FP8 shared expert on GuCfg8's column sub-tiles.
template <class C4, bool W128>
__device__ __forceinline__ void pf_gu(const GuArgs& a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    const int c0 = blockIdx.x * C4::BN;
    const int main = (a.n_tiles - a.rest_tiles) * 32;
    if (a.owner && c0 >= main && a.owner[e] == 0) return;
    __shared__ int srow[kBm];
    __shared__ uint8_t wss[kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < kBm; r += kThreads) srow[r] = m0 + r < cnt ? a.erow[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = reinterpret_cast<const uint8_t*>(a.w1[e]);
    o.w1 = reinterpret_cast<const uint8_t*>(a.w3[e]);
    o.ws0 = reinterpret_cast<const uint8_t*>(a.s1[e]);
    o.ws1 = reinterpret_cast<const uint8_t*>(a.s3[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = PF_HINT;
    if (fp8) {
        using C = GuCfg8;
        for (int half = 0; half < C4::BN / C::BN; half++) {
            o.n0 = c0 + half * C::BN;
            if (half) __syncthreads();  // the stages' buffers and the scales are the next sub-tile's
            fp8_scales<C>(wss, o, a.D / kBlk);
            __syncthreads();
            float acc[2][C::MT][C::NT][4];
            run_tile<C, false>(smem, srow, wss, o, a.D / kBlk, 0, cnt - m0, acc);
            gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
        }
    } else {
        using C = C4;
        o.n0 = c0;
        __syncthreads();
        float acc[2][C::MT][C::NT][4];
        if constexpr (W128) run_tile_w128<C>(smem, srow, o, a.D / kBlk, cnt - m0, acc);
        else run_tile<C, false>(smem, srow, wss, o, a.D / kBlk, 0, cnt - m0, acc);
        gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

// Down of the grouped experts for a prompt chunk on the MXFP4 tiles of C4 (W128 as pf_gu's); the FP8 shared expert on
// DnCfg8's tile.
template <class C4, bool W128>
__device__ __forceinline__ void pf_dn(const DnArgs& a, const int2* __restrict__ tiles, const int* __restrict__ ntiles) {
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    __shared__ int srow[kBm];
    __shared__ uint8_t wss[kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < kBm; r += kThreads) srow[r] = m0 + r < cnt ? a.ent[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = o.w1 = reinterpret_cast<const uint8_t*>(a.w2[e]);
    o.ws0 = o.ws1 = reinterpret_cast<const uint8_t*>(a.s2[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = PF_HINT;
    o.n0 = blockIdx.x * C4::BN;
    if (fp8) {
        using C = DnCfg8;
        fp8_scales<C>(wss, o, a.I / kBlk);
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        run_tile<C, false>(smem, srow, wss, o, a.I / kBlk, 0, cnt - m0, acc);
        dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    } else {
        using C = C4;
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        if constexpr (W128) run_tile_w128<C>(smem, srow, o, a.I / kBlk, cnt - m0, acc);
        else run_tile<C, false>(smem, srow, wss, o, a.I / kBlk, 0, cnt - m0, acc);
        dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

constexpr int kGuR128Smem = W2<GuR128Cfg>::SMEM > GuCfg8::SMEM ? W2<GuR128Cfg>::SMEM : GuCfg8::SMEM;
constexpr int kDnR128Smem = W2<DnR128Cfg>::SMEM > DnCfg8::SMEM ? W2<DnR128Cfg>::SMEM : DnCfg8::SMEM;
}  // namespace pf

// Gate / up of the grouped experts for a prompt chunk (GuArgs as tf_dsn_gu, splits 1; the pack's columns a multiple of
// 128, its main block's too): grid (n_tiles * 32 / 128, P / 64 + E), 256 threads, kPfGuSmem dynamic bytes; blockIdx.y
// the tile (tiles[], ntiles from tf_dsn_pf_tiles).
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gu_kernel(GuArgs a, const int2* __restrict__ tiles,
                                                                         const int* __restrict__ ntiles) {
    pf::pf_gu<pf::GuCfg4, false>(a, tiles, ntiles);
}

// Down of the grouped experts for a prompt chunk (DnArgs as tf_dsn_dn, splits 1; N a multiple of 128): grid
// (N / 128, P / 64 + E), 256 threads, kPfDnSmem dynamic bytes, the tile list of the same grouping.
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_dn_kernel(DnArgs a, const int2* __restrict__ tiles,
                                                                         const int* __restrict__ ntiles) {
    pf::pf_dn<pf::DnCfg4, false>(a, tiles, ntiles);
}

// tf_dsn_pf_gu / tf_dsn_pf_dn with their MXFP4 tiles' weights staged 128 bytes a row (TF_DS_PF_RUN128: pf::W2,
// run_tile_w128) on the twins' own configurations: the same body, arguments, grid, tiles and FP8 shared-expert
// sub-tiles, the same arithmetic in the same order, so their bits; kGuR128Smem / kDnR128Smem dynamic bytes.
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gu_r128_kernel(GuArgs a, const int2* __restrict__ tiles,
                                                                              const int* __restrict__ ntiles) {
    pf::pf_gu<pf::GuR128Cfg, true>(a, tiles, ntiles);
}

extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_dn_r128_kernel(DnArgs a, const int2* __restrict__ tiles,
                                                                              const int* __restrict__ ntiles) {
    pf::pf_dn<pf::DnR128Cfg, true>(a, tiles, ntiles);
}

// The bf16 GEMM of a prompt chunk (BfArgs as tf_dsn_bf16_gemm, splits 1): the same MMA chain (m16n8k16 accumulating
// in its C fragment, 16-column K steps in order, the standard fragment layout), so the same bits, on 128 x 128 CTA tiles
// staged by cp.async (32 K columns a stage, 4 stages) and read by ldmatrix. N a multiple of 128, K of 32; grid
// (ceil(M / 128), N / 128, groups), 256 threads, kPfBfSmem dynamic bytes.
namespace pf {
constexpr int kBfBM = 128, kBfBN = 128, kBfKS = 32, kBfStages = 4;
constexpr int kBfRow = kBfKS * 2 + 16;  // a staged row's bytes (80: ldmatrix's eight rows on distinct banks)
constexpr int kBfStage = (kBfBM + kBfBN) * kBfRow;
constexpr int kBfSmem = kBfStages * kBfStage;

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sptr(p)));
}
}  // namespace pf

extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_bf16_kernel(BfArgs a) {
    using namespace pf;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / 4, wn = warp % 4;  // warp tile 64 rows x 32 columns
    const int grp = blockIdx.z;
    const __nv_bfloat16* x = a.x + size_t(grp) * a.gx;
    const __nv_bfloat16* w = a.w + size_t(grp) * a.gw;
    int tr, tc;
    pf_tile_of(a.raster, tr, tc);  // row tiles fastest by default (the weight tiles shared in L2)
    const int m0 = tr * kBfBM, n0 = tc * kBfBN;
    const int nst = a.K / kBfKS;
    auto load = [&](int st, int slot) {
        char* A = smem + slot * kBfStage;
        char* B = A + kBfBM * kBfRow;
        const int k0 = st * kBfKS;
        // 4 16-byte pieces a row, 128 + 128 rows: 4 pieces a thread
        for (int c = tid; c < (kBfBM + kBfBN) * 4; c += kThreads) {
            const int r = c >> 2, cc = c & 3;
            if (r < kBfBM) {
                const int m = m0 + r;
                const bool ok = m < a.M;
                cp16(A + r * kBfRow + cc * 16, x + size_t(ok ? m : 0) * a.ldx + k0 + cc * 8, ok);
            } else {
                const int n = n0 + r - kBfBM;
                cp16(B + (r - kBfBM) * kBfRow + cc * 16, w + size_t(n) * a.ldw + k0 + cc * 8, true);
            }
        }
    };
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
    const int mvalid = a.M - m0;
#pragma unroll
    for (int s = 0; s < kBfStages - 1; s++) {
        if (s < nst) load(s, s);
        commit();
    }
    for (int st = 0; st < nst; st++) {
        wait_groups<kBfStages - 2>();
        __syncthreads();
        if (st + kBfStages - 1 < nst) load(st + kBfStages - 1, (st + kBfStages - 1) % kBfStages);
        commit();
        const char* A = smem + (st % kBfStages) * kBfStage;
        const char* B = A + kBfBM * kBfRow;
#pragma unroll
        for (int ks = 0; ks < kBfKS / 16; ks++) {
            // A: m16 tile i's x4 (rows +0..7 / +8..15, k +0..7 / +8..8): lanes 0-15 rows, 16-31 the high k half
            uint32_t af[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r = wm * 64 + i * 16 + (lane & 15);
                ldsm4(af[i], A + r * kBfRow + ks * 32 + (lane >> 4) * 16);
            }
            // B: two n8 tiles a x4 (n rows +0..7 at k +0, +8; the next n8 tile's the same)
            uint32_t bf[4][2];
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                uint32_t r4[4];
                const int n = wn * 32 + jj * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm4(r4, B + n * kBfRow + ks * 32 + ((lane >> 3) & 1) * 16);
                bf[2 * jj][0] = r4[0];
                bf[2 * jj][1] = r4[1];
                bf[2 * jj + 1][0] = r4[2];
                bf[2 * jj + 1][1] = r4[3];
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (wm * 64 + i * 16 >= mvalid) continue;
#pragma unroll
                for (int j = 0; j < 4; j++) mma_bf16(acc[i][j], af[i], bf[j]);
            }
        }
    }
    wait_groups<0>();
    char* yb = reinterpret_cast<char*>(a.y);
    const int esz = a.y_dtype == 2 ? 4 : 2;
    void* y = yb + size_t(grp) * a.gy * esz;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * 64 + i * 16 + g + (c >> 1) * 8;
                if (m >= a.M) continue;
                const int n = n0 + wn * 32 + j * 8 + 2 * t + (c & 1);
                bf_store(a, y, m, n, acc[i][j][c]);
            }
}

// tf_dsn_pf_bf16 on a 64 x 64 CTA tile (TF_DS_PF_T64): 128 threads (4 warps of 32 x 32), the same staging (cp.async, 32 K columns a
// stage, 4 stages of 80-byte rows), the same ldmatrix reads and m16n8k16 chain accumulating in the C fragment over 16-column K steps in
// order, so every output element's K chain is the 128 x 128 kernel's and the bits are its bits (the tile changes M and N only; no
// split-K). Two CTAs share an SM (kBf64Smem + the 1 KiB each reserves, twice, within 100 KiB). N a multiple of 64, K of 32; grid
// (ceil(M / 64), N / 64, groups), 128 threads, kBf64Smem dynamic bytes.
namespace pf {
constexpr int kBf64BM = 64, kBf64BN = 64;
constexpr int kBf64Stage = (kBf64BM + kBf64BN) * kBfRow;
constexpr int kBf64Smem = kBfStages * kBf64Stage;
static_assert(kBf64Smem == 40960, "tf_dsn_pf_bf16_t64: mx.zig pf_bf16_t64_smem");
static_assert(2 * (kBf64Smem + 1024) <= 100 * 1024, "tf_dsn_pf_bf16_t64: two CTAs an SM");
}  // namespace pf

extern "C" __global__ void __launch_bounds__(128, 2) tf_dsn_pf_bf16_t64_kernel(BfArgs a) {
    using namespace pf;
    constexpr int kT = 128;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / 2, wn = warp % 2;  // warp tile 32 rows x 32 columns
    const int grp = blockIdx.z;
    const __nv_bfloat16* x = a.x + size_t(grp) * a.gx;
    const __nv_bfloat16* w = a.w + size_t(grp) * a.gw;
    int tr, tc;
    pf_tile_of(a.raster, tr, tc);
    const int m0 = tr * kBf64BM, n0 = tc * kBf64BN;
    const int nst = a.K / kBfKS;
    auto load = [&](int st, int slot) {
        char* A = smem + slot * kBf64Stage;
        const int k0 = st * kBfKS;
        // 4 16-byte pieces a row, 64 + 64 rows: 4 pieces a thread
        for (int c = tid; c < (kBf64BM + kBf64BN) * 4; c += kT) {
            const int r = c >> 2, cc = c & 3;
            if (r < kBf64BM) {
                const int m = m0 + r;
                const bool ok = m < a.M;
                cp16(A + r * kBfRow + cc * 16, x + size_t(ok ? m : 0) * a.ldx + k0 + cc * 8, ok);
            } else {
                const int n = n0 + r - kBf64BM;
                cp16(A + r * kBfRow + cc * 16, w + size_t(n) * a.ldw + k0 + cc * 8, true);
            }
        }
    };
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
    const int mvalid = a.M - m0;
#pragma unroll
    for (int s = 0; s < kBfStages - 1; s++) {
        if (s < nst) load(s, s);
        commit();
    }
    for (int st = 0; st < nst; st++) {
        wait_groups<kBfStages - 2>();
        __syncthreads();
        if (st + kBfStages - 1 < nst) load(st + kBfStages - 1, (st + kBfStages - 1) % kBfStages);
        commit();
        const char* A = smem + (st % kBfStages) * kBf64Stage;
        const char* B = A + kBf64BM * kBfRow;
#pragma unroll
        for (int ks = 0; ks < kBfKS / 16; ks++) {
            uint32_t af[2][4];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = wm * 32 + i * 16 + (lane & 15);
                ldsm4(af[i], A + r * kBfRow + ks * 32 + (lane >> 4) * 16);
            }
            uint32_t bf[4][2];
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                uint32_t r4[4];
                const int n = wn * 32 + jj * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm4(r4, B + n * kBfRow + ks * 32 + ((lane >> 3) & 1) * 16);
                bf[2 * jj][0] = r4[0];
                bf[2 * jj][1] = r4[1];
                bf[2 * jj + 1][0] = r4[2];
                bf[2 * jj + 1][1] = r4[3];
            }
#pragma unroll
            for (int i = 0; i < 2; i++) {
                if (wm * 32 + i * 16 >= mvalid) continue;
#pragma unroll
                for (int j = 0; j < 4; j++) mma_bf16(acc[i][j], af[i], bf[j]);
            }
        }
    }
    wait_groups<0>();
    char* yb = reinterpret_cast<char*>(a.y);
    const int esz = a.y_dtype == 2 ? 4 : 2;
    void* y = yb + size_t(grp) * a.gy * esz;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * 32 + i * 16 + g + (c >> 1) * 8;
                if (m >= a.M) continue;
                const int n = n0 + wn * 32 + j * 8 + 2 * t + (c & 1);
                bf_store(a, y, m, n, acc[i][j][c]);
            }
}

// ---------------------------------------------------------------------------------------------------------------------
// The indexer's pruned top-k of a prompt chunk in one launch (TF_DS_TOPK_TILES; tri_index.topkSelectPruned's chain:
// the kk tiles with the largest tile maxima, their keys gathered, the kk largest of those, their indices sorted). Keys
// are int64, unique within a row (the score's ordered bits above, the inverted position below: kernels.py), so "the kk
// largest" is one set however it is found. A CTA a row: (1) the kk largest of the row's nt tile maxima tmax [R, nt],
// by a radix select (8-bit digits from the top, a value's unsigned order its key's with the sign bit flipped) to the
// smallest threshold that exactly kk maxima reach, those tiles listed; (2) the same select over those tiles' keys read
// in place from keys [R, ks] (positions below n only); (3) the kk keys' positions sorted ascending (a bitonic sort),
// each at or past vis[r] written -1: out [R, kk] int64, _topk_finish's tail. A key among a row's kk largest lies in a
// tile whose maximum is at least that key, and fewer than kk maxima exceed it, so the kk tiles hold the answer (the
// argument the pruned chain rests on). Each tile maximum is one of its tile's keys (the prompt path's index score
// writes them so, DEAD_PAD off: tri_index's test pins it), so the maxima are unique too; with values tied across a
// threshold either selection would keep the first kk by arrival, run to run different. kk <= kTopkMax; 1,024 threads.
namespace topk {
constexpr int kTopkMax = 512;

// the kk-th largest of N distinct values val(i), i in [0, N) where val(i) is valid (else skipped): the smallest
// threshold that exactly kk values reach. hist: 256 counters; sh: 3 ints. Every thread of the CTA calls it.
template <class Val>
__device__ uint64_t select_kth(const Val& val, int N, int kk, unsigned* hist, int* sh) {
    uint64_t prefix = 0;  // the digits chosen so far, in place
    int need = kk;
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const uint64_t hi = shift == 56 ? 0 : ~0ull << (shift + 8);
        for (int base = 0; base < N; base += blockDim.x) {
            const int i = base + threadIdx.x;
            int d = 256;  // no value of this prefix
            uint64_t u;
            if (i < N && val(i, u) && (u & hi) == prefix) d = int((u >> shift) & 255);
            const unsigned peers = __match_any_sync(0xffffffffu, d);
            if (d != 256 && (threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(hist + d, __popc(peers));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            // the digit holding the need-th largest value of this prefix, counted from the top bin
            int above = 0, d = 255;
            for (; d > 0; d--) {
                if (above + int(hist[d]) >= need) break;
                above += int(hist[d]);
            }
            sh[0] = d;
            sh[1] = need - above;
            sh[2] = int(hist[d]) == need - above;
        }
        __syncthreads();
        prefix |= uint64_t(sh[0]) << shift;
        need = sh[1];
        const bool all = sh[2] != 0;
        __syncthreads();
        // every value of these digits is among the kk: the smallest of them (lower digits zero) is the threshold
        if (all) return prefix;
    }
    return prefix;
}

__device__ __forceinline__ uint64_t flip(int64_t key) { return uint64_t(key) ^ (1ull << 63); }
}  // namespace topk

extern "C" __global__ void __launch_bounds__(1024) tf_dsn_topk_tiles_kernel(const int64_t* __restrict__ keys, int64_t ks,
                                                                            const int64_t* __restrict__ tmax, int nt, int n,
                                                                            int kk, const int64_t* __restrict__ vis,
                                                                            int64_t* __restrict__ out) {
    using namespace topk;
    __shared__ unsigned hist[256];
    __shared__ int sh[4];
    __shared__ int tl[kTopkMax];  // the kk tiles
    __shared__ int pl[kTopkMax];  // the kk keys' positions
    const int r = blockIdx.x;
    const int64_t* tm = tmax + size_t(r) * nt;
    const int64_t* kr = keys + size_t(r) * ks;
    // (1) the kk tiles with the largest maxima
    const auto tval = [&](int i, uint64_t& u) {
        u = flip(tm[i]);
        return true;
    };
    const uint64_t t1 = select_kth(tval, nt, kk, hist, sh);
    if (threadIdx.x == 0) sh[3] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < nt; i += blockDim.x)
        if (flip(tm[i]) >= t1) {
            const int at = atomicAdd(sh + 3, 1);
            if (at < kk) tl[at] = i;  // (exactly kk: the maxima are unique)
        }
    __syncthreads();
    // (2) the kk largest keys of those tiles (their positions below n)
    const auto kval = [&](int c, uint64_t& u) {
        const int pos = tl[c >> 6] * 64 + (c & 63);
        if (pos >= n) return false;
        u = flip(kr[pos]);
        return true;
    };
    const uint64_t t2 = select_kth(kval, kk * 64, kk, hist, sh);
    if (threadIdx.x == 0) sh[3] = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < kk * 64; c += blockDim.x) {
        const int pos = tl[c >> 6] * 64 + (c & 63);
        if (pos < n && flip(kr[pos]) >= t2) {
            const int at = atomicAdd(sh + 3, 1);
            if (at < kk) pl[at] = pos;  // (exactly kk: the keys are unique)
        }
    }
    for (int i = kk + threadIdx.x; i < kTopkMax; i += blockDim.x) pl[i] = 0x7fffffff;  // past kk: last after the sort
    __syncthreads();
    // (3) the positions ascending
    for (int size = 2; size <= kTopkMax; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = threadIdx.x; t < kTopkMax / 2; t += blockDim.x) {
                const int i = 2 * stride * (t / stride) + t % stride, j = i + stride;
                const int a = pl[i], b = pl[j];
                if ((a > b) == ((i & size) == 0)) {
                    pl[i] = b;
                    pl[j] = a;
                }
            }
            __syncthreads();
        }
    const int64_t v = vis[r];
    for (int i = threadIdx.x; i < kk; i += blockDim.x) out[size_t(r) * kk + i] = pl[i] < v ? int64_t(pl[i]) : -1;
}

// wo_a's groups of a prompt chunk on their FP8 rows (mx.woA8Prefill): tf_dsn_pf_bf16's tile, MMA chain and epilogue on
// the bf16 values its operands would hold, made from FP8 in the kernel. A: the activation's E4M3 codes q [M, K] (row
// stride ldq bytes) with their act_quant scales s [M, K / 32] (row stride lds floats), each value
// bf16(from_e4m3(code) * scale): act_quant's quant-dequant value (its `fake` rows). B: the weight's E4M3 rows w [N, K]
// (row stride ldw bytes) with UE8M0 scales ws [N / 32, K / 32] (row stride ldws bytes), each value the top 16 bits of
// from_e4m3(code) * 2^(e - 127): the loader's bf16 rows (weights.zig woABf16), for every code but the E4M3 NaNs
// (0x7f, 0xff) and every scale byte but 255 (the loader admits only those). FP8 stages (one K block: 128 rows of A and
// B, A's 128 scales) stream through a ring of kW8Stages by cp.async; each is converted once, by the whole CTA, into
// one of two bf16 stages in pf_bf16's layout, which pf_bf16's ldmatrix and m16n8k16 chain read in K order. y [M, N]
// (row stride ldy, y_dtype). N a multiple of 128, K of 32; grid (ceil(M / 128), N / 128), 256 threads, kW8Smem
// dynamic bytes.
struct W8Args {
    const uint8_t* q;
    const float* s;
    int64_t ldq, lds;
    const uint8_t* w;
    const uint8_t* ws;
    int64_t ldw, ldws;
    int M, N, K;
    int y_dtype;
    void* y;
    int64_t ldy;
    // TF_DS_WOA_MERGE: the groups of a layer in ONE launch (grid.z = groups): group z's operands are the base pointers above plus z times these byte strides (q's codes, s's scales, w's codes, ws's
    // scales, y's output); 0 in a one-group launch (grid.z 1), so the per-group launches are as they were. Each CTA is one group's tile and runs that group's code unchanged: the bytes are the two launches'.
    int64_t gq, gs, gw, gws, gy;
};

// W8Args of the group this CTA serves (blockIdx.z): the base operands moved by the group strides (zero strides: unchanged).
__device__ __forceinline__ void w8_group(W8Args& a) {
    const size_t z = blockIdx.z;
    a.q += z * size_t(a.gq);
    a.s = reinterpret_cast<const float*>(reinterpret_cast<const char*>(a.s) + z * size_t(a.gs));
    a.w += z * size_t(a.gw);
    a.ws += z * size_t(a.gws);
    a.y = static_cast<char*>(a.y) + z * size_t(a.gy);
}

namespace pf {
constexpr int kW8Stages = 5;
constexpr int kW8A = kBfBM * kBfKS;                  // A's codes a stage: 128 rows x 32
constexpr int kW8B = kBfBN * kBfKS;                  // B's codes a stage
constexpr int kW8Stage = kW8A + kW8B + kBfBM * 4;    // and A's 128 row scales: 8,704 bytes (68 x 128)
constexpr int kW8Smem = kW8Stages * kW8Stage + 2 * kBfStage;
constexpr int kW8MaxK = kWss / (kBfBN / kBlk) * kBfKS;  // the largest K: a tile's weight scales fill wss
}  // namespace pf

// tf_dsn_pf_bf16w8 on 16 warps a block (TF_DS_GEMM_W16): the same 128 x 128 tile, FP8 ring, conversions and MMA
// chain on 512 threads, a warp two m16 tiles (32 rows) of its column group instead of four: each output element's
// chain of MMAs walks the same K blocks in the same order, so the bytes are the 8-warp kernel's. The tile and its
// stages do not move, so its shared memory (pf::kW8Smem, mx.zig pf_w8_smem) is unchanged.
extern "C" __global__ void __launch_bounds__(512, 1) tf_dsn_pf_bf16w8_w16_kernel(W8Args a) {
    using namespace pf;
    w8_group(a);  // (TF_DS_WOA_MERGE: this CTA's group; a one-group launch moves nothing)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / 4, wn = warp % 4;  // warp tile 32 rows x 32 columns (pf_bf16's, two m16 tiles)
    const int nst = a.K / kBfKS;  // one 32-column K block a stage
    // the tile's weight scales [BN / 32][K / 32] (fp8_scales' layout for one weight)
    __shared__ uint8_t wss[kWss];
    for (int i = tid; i < (kBfBN / kBlk) * nst; i += 512) {
        const int blk = i / nst, kb = i % nst;
        wss[i] = a.ws[size_t(n0 / kBlk + blk) * a.ldws + kb];
    }
    char* ring = smem;                         // the FP8 stages
    char* bfs = smem + kW8Stages * kW8Stage;   // two bf16 stages, pf_bf16's layout (A rows, then B rows, kBfRow bytes)
    auto load = [&](int st, int slot) {
        char* A = ring + slot * kW8Stage;
        char* B = A + kW8A;
        char* AS = B + kW8B;
        const int k0 = st * kBfKS;
        // two 16-byte pieces a row, 128 + 128 rows: 2 pieces a thread
        for (int c = tid; c < (kBfBM + kBfBN) * 2; c += 512) {
            const int r = c >> 1, cc = c & 1;
            if (r < kBfBM) {
                const int m = m0 + r;
                const bool ok = m < a.M;
                cp16(A + r * kBfKS + cc * 16, a.q + size_t(ok ? m : 0) * a.ldq + k0 + cc * 16, ok);
            } else {
                const int n = n0 + r - kBfBM;
                cp16(B + (r - kBfBM) * kBfKS + cc * 16, a.w + size_t(n) * a.ldw + k0 + cc * 16, true);
            }
        }
        for (int r = tid; r < kBfBM; r += 512) {
            const int m = m0 + r;
            const bool ok = m < a.M;
            cp4(AS + r * 4, a.s + size_t(ok ? m : 0) * a.lds + st, ok);
        }
    };
    // stage st's FP8 (slot) into bf16 stage b: a thread 16 values of a row of A and 16 of the same row of B
    auto convert = [&](int st, int slot, int b) {
        const char* A = ring + slot * kW8Stage;
        const char* B = A + kW8A;
        const float* AS = reinterpret_cast<const float*>(B + kW8B);
        char* D = bfs + b * kBfStage;
        if (tid >= 2 * kBfBM) return;  // the first 128 threads' pairs own the rows
        const int r = tid >> 1, half = tid & 1;
        {
            const uint4 v = *reinterpret_cast<const uint4*>(A + r * kBfKS + half * 16);
            const uint8_t* c = reinterpret_cast<const uint8_t*>(&v);
            const float sc = AS[r];
            uint32_t o[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(from_e4m3(c[2 * i]) * sc));
                const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(from_e4m3(c[2 * i + 1]) * sc));
                o[i] = lo | hi << 16;
            }
            uint4* dst = reinterpret_cast<uint4*>(D + r * kBfRow + half * 32);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        {
            const uint4 v = *reinterpret_cast<const uint4*>(B + r * kBfKS + half * 16);
            const uint8_t* c = reinterpret_cast<const uint8_t*>(&v);
            const float sc = ue8m0(wss[(r / kBlk) * nst + st]);
            uint32_t o[8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                o[i] = __byte_perm(__float_as_uint(from_e4m3(c[2 * i]) * sc), __float_as_uint(from_e4m3(c[2 * i + 1]) * sc), 0x7632);
            uint4* dst = reinterpret_cast<uint4*>(D + (kBfBM + r) * kBfRow + half * 32);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
    };
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
    const int mvalid = a.M - m0;
#pragma unroll
    for (int s = 0; s < kW8Stages; s++) {
        if (s < nst) load(s, s);
        commit();
    }
    wait_groups<kW8Stages - 1>();  // FP8 stage 0
    __syncthreads();
    if (nst > 0) convert(0, 0, 0);
    for (int st = 0; st < nst; st++) {
        // FP8 stage st + 1 landed; every thread done with iteration st - 1 (its MMAs on bf16 stage (st + 1) & 1, its
        // conversion of stage st into bf16 stage st & 1 from slot st % kW8Stages)
        wait_groups<kW8Stages - 2>();
        __syncthreads();
        if (st + kW8Stages < nst) load(st + kW8Stages, st % kW8Stages);
        commit();
        if (st + 1 < nst) convert(st + 1, (st + 1) % kW8Stages, (st + 1) & 1);
        const char* A = bfs + (st & 1) * kBfStage;
        const char* B = A + kBfBM * kBfRow;
#pragma unroll
        for (int ks = 0; ks < kBfKS / 16; ks++) {
            uint32_t af[2][4];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int r = wm * 32 + i * 16 + (lane & 15);
                ldsm4(af[i], A + r * kBfRow + ks * 32 + (lane >> 4) * 16);
            }
            uint32_t bf[4][2];
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                uint32_t r4[4];
                const int n = wn * 32 + jj * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm4(r4, B + n * kBfRow + ks * 32 + ((lane >> 3) & 1) * 16);
                bf[2 * jj][0] = r4[0];
                bf[2 * jj][1] = r4[1];
                bf[2 * jj + 1][0] = r4[2];
                bf[2 * jj + 1][1] = r4[3];
            }
#pragma unroll
            for (int i = 0; i < 2; i++) {
                if (wm * 32 + i * 16 >= mvalid) continue;
#pragma unroll
                for (int j = 0; j < 4; j++) mma_bf16(acc[i][j], af[i], bf[j]);
            }
        }
    }
    wait_groups<0>();
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * 32 + i * 16 + g + (c >> 1) * 8;
                if (m >= a.M) continue;
                const int n = n0 + wn * 32 + j * 8 + 2 * t + (c & 1);
                store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, acc[i][j][c]);
            }
}
extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_bf16w8_kernel(W8Args a) {
    using namespace pf;
    w8_group(a);  // (TF_DS_WOA_MERGE: this CTA's group; a one-group launch moves nothing)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / 4, wn = warp % 4;  // warp tile 64 rows x 32 columns (pf_bf16's)
    const int nst = a.K / kBfKS;  // one 32-column K block a stage
    // the tile's weight scales [BN / 32][K / 32] (fp8_scales' layout for one weight)
    __shared__ uint8_t wss[kWss];
    for (int i = tid; i < (kBfBN / kBlk) * nst; i += kThreads) {
        const int blk = i / nst, kb = i % nst;
        wss[i] = a.ws[size_t(n0 / kBlk + blk) * a.ldws + kb];
    }
    char* ring = smem;                         // the FP8 stages
    char* bfs = smem + kW8Stages * kW8Stage;   // two bf16 stages, pf_bf16's layout (A rows, then B rows, kBfRow bytes)
    auto load = [&](int st, int slot) {
        char* A = ring + slot * kW8Stage;
        char* B = A + kW8A;
        char* AS = B + kW8B;
        const int k0 = st * kBfKS;
        // two 16-byte pieces a row, 128 + 128 rows: 2 pieces a thread
        for (int c = tid; c < (kBfBM + kBfBN) * 2; c += kThreads) {
            const int r = c >> 1, cc = c & 1;
            if (r < kBfBM) {
                const int m = m0 + r;
                const bool ok = m < a.M;
                cp16(A + r * kBfKS + cc * 16, a.q + size_t(ok ? m : 0) * a.ldq + k0 + cc * 16, ok);
            } else {
                const int n = n0 + r - kBfBM;
                cp16(B + (r - kBfBM) * kBfKS + cc * 16, a.w + size_t(n) * a.ldw + k0 + cc * 16, true);
            }
        }
        for (int r = tid; r < kBfBM; r += kThreads) {
            const int m = m0 + r;
            const bool ok = m < a.M;
            cp4(AS + r * 4, a.s + size_t(ok ? m : 0) * a.lds + st, ok);
        }
    };
    // stage st's FP8 (slot) into bf16 stage b: a thread 16 values of a row of A and 16 of the same row of B
    auto convert = [&](int st, int slot, int b) {
        const char* A = ring + slot * kW8Stage;
        const char* B = A + kW8A;
        const float* AS = reinterpret_cast<const float*>(B + kW8B);
        char* D = bfs + b * kBfStage;
        const int r = tid >> 1, half = tid & 1;
        {
            const uint4 v = *reinterpret_cast<const uint4*>(A + r * kBfKS + half * 16);
            const uint8_t* c = reinterpret_cast<const uint8_t*>(&v);
            const float sc = AS[r];
            uint32_t o[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(from_e4m3(c[2 * i]) * sc));
                const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(from_e4m3(c[2 * i + 1]) * sc));
                o[i] = lo | hi << 16;
            }
            uint4* dst = reinterpret_cast<uint4*>(D + r * kBfRow + half * 32);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        {
            const uint4 v = *reinterpret_cast<const uint4*>(B + r * kBfKS + half * 16);
            const uint8_t* c = reinterpret_cast<const uint8_t*>(&v);
            const float sc = ue8m0(wss[(r / kBlk) * nst + st]);
            uint32_t o[8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                o[i] = __byte_perm(__float_as_uint(from_e4m3(c[2 * i]) * sc), __float_as_uint(from_e4m3(c[2 * i + 1]) * sc), 0x7632);
            uint4* dst = reinterpret_cast<uint4*>(D + (kBfBM + r) * kBfRow + half * 32);
            dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
            dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
    };
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
    const int mvalid = a.M - m0;
#pragma unroll
    for (int s = 0; s < kW8Stages; s++) {
        if (s < nst) load(s, s);
        commit();
    }
    wait_groups<kW8Stages - 1>();  // FP8 stage 0
    __syncthreads();
    if (nst > 0) convert(0, 0, 0);
    for (int st = 0; st < nst; st++) {
        // FP8 stage st + 1 landed; every thread done with iteration st - 1 (its MMAs on bf16 stage (st + 1) & 1, its
        // conversion of stage st into bf16 stage st & 1 from slot st % kW8Stages)
        wait_groups<kW8Stages - 2>();
        __syncthreads();
        if (st + kW8Stages < nst) load(st + kW8Stages, st % kW8Stages);
        commit();
        if (st + 1 < nst) convert(st + 1, (st + 1) % kW8Stages, (st + 1) & 1);
        const char* A = bfs + (st & 1) * kBfStage;
        const char* B = A + kBfBM * kBfRow;
#pragma unroll
        for (int ks = 0; ks < kBfKS / 16; ks++) {
            uint32_t af[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r = wm * 64 + i * 16 + (lane & 15);
                ldsm4(af[i], A + r * kBfRow + ks * 32 + (lane >> 4) * 16);
            }
            uint32_t bf[4][2];
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                uint32_t r4[4];
                const int n = wn * 32 + jj * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm4(r4, B + n * kBfRow + ks * 32 + ((lane >> 3) & 1) * 16);
                bf[2 * jj][0] = r4[0];
                bf[2 * jj][1] = r4[1];
                bf[2 * jj + 1][0] = r4[2];
                bf[2 * jj + 1][1] = r4[3];
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (wm * 64 + i * 16 >= mvalid) continue;
#pragma unroll
                for (int j = 0; j < 4; j++) mma_bf16(acc[i][j], af[i], bf[j]);
            }
        }
    }
    wait_groups<0>();
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * 64 + i * 16 + g + (c >> 1) * 8;
                if (m >= a.M) continue;
                const int n = n0 + wn * 32 + j * 8 + 2 * t + (c & 1);
                store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, acc[i][j][c]);
            }
}

// the prefill kernels' dynamic shared memory (mx.zig pf_*_smem)
#ifndef PF_SWEEP
static_assert(pf::kGemmSmem == 76800 && pf::kGuSmem == 79872 && pf::kDnSmem == 53760 && pf::kBfSmem == 81920, "prefill shared memory");
static_assert(pf::GemmSplitACfg::SMEM == 87040 && pf::GemmSplitBCfg::SMEM == 76800 && pf::GemmSplitC6Cfg::SMEM == 76800 &&
              pf::GemmSplitC7Cfg::SMEM == 89600, "the split GEMMs' shared memory");
static_assert(pf::kW8Stage == 8704 && pf::kW8Smem == 84480, "the FP8 wo_a kernel's shared memory");
static_assert(pf::GemmFiveCfg::SMEM == 87040, "the 5-stage GEMM's shared memory");
#endif
static_assert(pf::kGuR128Smem == 97280 && pf::kDnR128Smem == 62464, "the 128-byte-run MoE tiles' shared memory");
static_assert(sizeof(W8Args) == 144 && offsetof(W8Args, M) == 64 && offsetof(W8Args, y) == 80 && offsetof(W8Args, raster) == 96 && offsetof(W8Args, gq) == 104 && offsetof(W8Args, gy) == 136, "W8Args layout");
static_assert(pf::kW8MaxK == 16384, "the FP8 wo_a kernel's largest K (weights.zig w8_max_k)");

// Decode tensor-core kernels ("dec"): a decode round's few rows on the MMA with the operands swapped (the m16 side 16
// weight rows, the n8 side 8 activation rows), the weights streamed through a per-warp ring of cp.async stages in
// shared memory (16-byte copies, 128 bytes of each of the tile's 16 rows a stage, several stages in flight: the
// GB10's memory wants many bytes outstanding and whole 128-byte runs of a row) and read back with ldmatrix. A warp
// owns a 16-row weight tile over a K chunk; the CTA's `ks` warps on one tile, and `cs` CTAs on it (grid z), split K
// as the layer's shape alone decides, the warps' partials summed in warp order in shared memory and the CTAs' in
// grid-z order by the last CTA to finish (a counter a tile group, reset by that CTA). Every (weight row, activation
// row) result is its own MMA output and the K order is fixed, so a row's bits never depend on the other rows of a
// launch or on how many there are.
//   - FP8 (fmt 0): 4 K blocks a stage, the MMA's natural K order;
//   - MXFP4 (fmt 1): 8 K blocks a stage. Each E2M1 code goes to the E4M3 byte (code & 7) << 2 | sign << 7, which is
//     the value times 2^-6 exactly (the subnormal 0.5 too), the 2^6 folded into the block's power-of-two weight scale;
//     a thread's 4 bytes (values 8t .. 8t + 7 of its row's block) feed k slots 4t + i (the even values) and
//     16 + 4t + i (the odd ones), the activation's bytes permuted the same way;
//   - bf16 (fmt 2): 64 values a stage (four m16n8k16 steps, k in order), the sum in the MMA's fp32 C.
// FP8 / FP4, a K block: acc = fmaf(mma_sum, scale_a * scale_w, acc), both scales powers of two (kernel.py's
// accum += C_local * scale_a * scale_b, one rounding).
namespace dec {

#ifndef TF_DSN_DEC_LDG
#define TF_DSN_DEC_LDG 1
#endif
#if TF_DSN_DEC_LDG
#define DEC_LD(p) __ldg(p)
#else
#define DEC_LD(p) (*(p))
#endif

// the two configurations (a weight row's bytes a stage, 64 or 128; the ring's stages): the dense linears' (few long
// warps: a deep ring each) and the experts' (many short ones: more resident warps)
#ifndef TF_DSN_DEC_RB_D
#define TF_DSN_DEC_RB_D 64
#endif
#ifndef TF_DSN_DEC_NS_D
#define TF_DSN_DEC_NS_D 4
#endif
#ifndef TF_DSN_DEC_RB_E
#define TF_DSN_DEC_RB_E 64
#endif
#ifndef TF_DSN_DEC_NS_E
#define TF_DSN_DEC_NS_E 2
#endif
#ifndef TF_DSN_DEC_RB_G
#define TF_DSN_DEC_RB_G 128
#endif
#ifndef TF_DSN_DEC_NS_G
#define TF_DSN_DEC_NS_G 2
#endif
// (the experts' gate / up apart: rows of 2560 bytes take 128-byte stages better, down's of 576 bytes 64-byte ones)
constexpr int kRbD = TF_DSN_DEC_RB_D, kNsD = TF_DSN_DEC_NS_D, kRbE = TF_DSN_DEC_RB_E, kNsE = TF_DSN_DEC_NS_E;
constexpr int kRbG = TF_DSN_DEC_RB_G, kNsG = TF_DSN_DEC_NS_G;
#ifndef TF_DSN_DEC_SCB
#define TF_DSN_DEC_SCB RB
#endif
template <int RB>
__host__ __device__ constexpr int slotBytes() { return 16 * RB + TF_DSN_DEC_SCB; }  // a 16-row tile's stage and its scales

// programmatic dependent launch: let the stream's next kernel start (its weights' prologue) and wait for this one's
// predecessor (every global write before it visible); both no-ops for a launch without the attribute
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
// TF_DSN_DEC_EVICT=1 (default; 0 = off): the weights' copies carry an L2 evict-first policy (a decode round reads each weight byte once and
#define TF_DSN_DEC_EVICT 1
__device__ __forceinline__ void cp4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr)
                 : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
// four E2M1 codes (the low / high nibbles of r's bytes) as E4M3 bytes of the values times 2^-6
__device__ __forceinline__ uint32_t fp4lo(uint32_t r) { return ((r << 2) & 0x1C1C1C1Cu) | ((r << 4) & 0x80808080u); }
__device__ __forceinline__ uint32_t fp4hi(uint32_t r) { return ((r >> 2) & 0x1C1C1C1Cu) | (r & 0x80808080u); }

// a tile row's 16-byte chunk c in a stage, swizzled so ldmatrix's 8 rows of one chunk take 8 different bank groups
template <int kRowB>
__device__ __forceinline__ int swz(int row, int c) {
    return kRowB == 64 ? row * kRowB + ((c ^ ((row >> 1) & 3)) << 4) : row * kRowB + ((c ^ (row & 7)) << 4);
}

template <int FMT>
__device__ __forceinline__ float wscale(uint32_t e) {
    // FP4: the 2^6 of the E4M3 codes folded in
    return FMT == 1 ? __uint_as_float((e + 6) << 23) : __uint_as_float(e << 23);
}

// a format's K values a stage, and a row's bytes for K values
template <int FMT>
__host__ __device__ constexpr int64_t rowBytes(int64_t k) { return FMT == 1 ? k / 2 : FMT == 0 ? k : 2 * k; }

// An activation load of the tile loop: read-only global memory (DEC_LD), or (XS) the CTA's shared memory, where a
// kernel that quantizes its own activations put their codes and scales.
template <bool XS, typename T>
__device__ __forceinline__ T xld(const T* p) {
    if constexpr (XS) return *p;
    else return DEC_LD(p);
}

// core's step between the weights' first stages and the tile loop, after the wait: none, or (a kernel that quantizes
// its own activations) the CTA's quant of them.
struct NoPro {
    __device__ __forceinline__ void operator()() const {}
};

// One warp: weight rows n0 .. n0 + 15 of NM matrices (w[m], row stride ldw bytes; scales ws[m], row stride ldws
// bytes), stages st0 .. st1 - 1 of a row of rb bytes (the last stage may hold 64 bytes: K a multiple of 64 values for
// FP4, 64 bytes FP8 / bf16), activation rows i < cnt (row rows_of(i); FP8 / FP4: codes xq, row stride ldq bytes, and
// scales s, lds floats; bf16: xq bf16, ldq elements) in RG groups of 8. acc[m][g][c]: weight row n0 + lane / 4 +
// 8 (c >> 1), activation row 8 g + 2 (lane % 4) + (c & 1). XS: the activations in shared memory, written by `pro`
// (every warp of the CTA runs it after the wait, before the loop).
__device__ __forceinline__ void core(const uint8_t* const (&w)[NM], const uint8_t* const (&ws)[NM], int64_t ldw,
                                     int64_t ldws, int64_t rb, int n0, int st0, int st1, int sst, const void* xq, int64_t ldq,
                                     const float* s, int64_t lds, int cnt, RowOf rows_of, uint8_t* ring,
                                     float (&acc)[NM][RG][4], Pro pro = {}) {
    constexpr int kRowB = RB, kChunks = RB / 16, kTileB = 16 * RB, kSlotB = slotBytes<RB>(), kStages = NS;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < NM; m++)
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[m][r][c] = 0.f;
    const int ng = (cnt + 7) >> 3;  // the row groups in use (warp-uniform)
#pragma unroll
    for (int r = 0; r < RG; r++) {
        const int i = r * 8 + g, j0 = r * 8 + 2 * t, j1 = j0 + 1;
        const int64_t esz = FMT == 2 ? 2 : 1;
    }
    const uint32_t ring_s = smem_u32(ring);
    // ldmatrix: matrices (rows 0-7, chunk b), (rows 8-15, chunk b), (rows 0-7, b + 1), (rows 8-15, b + 1)
    const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1), lch = lane >> 4;
#pragma unroll
#pragma unroll
        const int c = lane + 32 * i, row = c / kChunks, ch = c % kChunks;
    }
    constexpr int wpr = kChunks / 4;
                }
            }
        }
        cp_commit();
    };
#pragma unroll
    // the weights' first stages are in flight: now the activations (the predecessor's output) may be read
    pdl_wait();
    pro();
    for (int st = st0; st < st1; st += sst) {
        cp_wait<kStages - 2>();
        __syncwarp();
        // a whole stage with compile-time counts, a short last stage (64 bytes) apart
        auto consume = [&](auto vcc) {
        constexpr int vc = decltype(vcc)::value;
        uint32_t r[NM][kChunks / 2][4];
#pragma unroll
        for (int m = 0; m < NM; m++) {
#pragma unroll
            for (int p = 0; p < kChunks / 2; p++)
        }
        if (FMT == 2) {
            // 16-value steps: chunks 2p, 2p + 1
#pragma unroll
            for (int p = 0; p < kChunks / 2; p++) {
                if (2 * p >= vc) break;
#pragma unroll
                for (int rg = 0; rg < RG; rg++) {
                    if (rg >= ng) break;
#pragma unroll
                    for (int m = 0; m < NM; m++) mma_bf16(acc[m][rg], r[m][p], b);
                }
            }
        } else {
            const int nb = FMT == 1 ? vc : vc / 2;
            uint32_t sw[NM][2];  // FP4: rows g, g + 8, blocks 0-3 / 4-7 (bytes); FP8: the tile's word
#pragma unroll
            for (int m = 0; m < NM; m++) {
                const uint8_t* sc = sbase + m * kSlotB + kTileB;
                if (FMT == 1) {
                    sw[m][0] = *reinterpret_cast<const uint32_t*>(sc + g * kChunks + 0);
                    sw[m][1] = kChunks > 4 ? *reinterpret_cast<const uint32_t*>(sc + g * kChunks + 4) : 0u;
                } else {
                    sw[m][0] = *reinterpret_cast<const uint32_t*>(sc);
                }
            }
            uint32_t sw8[NM][2];
#pragma unroll
            for (int m = 0; m < NM; m++) {
                const uint8_t* sc = sbase + m * kSlotB + kTileB;
                if (FMT == 1) {
                    sw8[m][0] = *reinterpret_cast<const uint32_t*>(sc + (g + 8) * kChunks + 0);
                    sw8[m][1] = kChunks > 4 ? *reinterpret_cast<const uint32_t*>(sc + (g + 8) * kChunks + 4) : 0u;
                }
            }
#pragma unroll
            for (int j = 0; j < NB; j++) {
                if (j >= nb) break;
                uint32_t a[NM][4];
                float swg[NM], swh[NM];
#pragma unroll
                for (int m = 0; m < NM; m++) {
                    if (FMT == 1) {
                        const uint32_t* rr = r[m][j >> 1];
                        const uint32_t x0 = rr[(j & 1) * 2], x1 = rr[(j & 1) * 2 + 1];
                        a[m][0] = fp4lo(x0);
                        a[m][1] = fp4lo(x1);
                        a[m][2] = fp4hi(x0);
                        a[m][3] = fp4hi(x1);
                        swg[m] = wscale<1>((sw[m][j >> 2] >> (8 * (j & 3))) & 0xff);
                        swh[m] = wscale<1>((sw8[m][j >> 2] >> (8 * (j & 3))) & 0xff);
                    } else {
#pragma unroll
                        for (int c = 0; c < 4; c++) a[m][c] = r[m][j][c];
                        swg[m] = swh[m] = wscale<0>((sw[m][0] >> (8 * (((st * NB) & 3) + j))) & 0xff);
                    }
                }
#pragma unroll
                for (int rg = 0; rg < RG; rg++) {
                    if (rg >= ng) break;
                    uint32_t b[2];
                        if (FMT == 1) {
                            b[0] = prmt(v.x, v.y, 0x6420);
                            b[1] = prmt(v.x, v.y, 0x7531);
                        } else {
                        }
#pragma unroll
                    for (int m = 0; m < NM; m++) {
                        float d[4];
                        mma_e4m3(d, a[m], b);
                        acc[m][rg][0] = fmaf(d[0], sa0 * swg[m], acc[m][rg][0]);
                        acc[m][rg][1] = fmaf(d[1], sa1 * swg[m], acc[m][rg][1]);
                        acc[m][rg][2] = fmaf(d[2], sa0 * swh[m], acc[m][rg][2]);
                        acc[m][rg][3] = fmaf(d[3], sa1 * swh[m], acc[m][rg][3]);
                    }
                }
            }
        }
        };
            if (vc_rt == kChunks) consume(std::integral_constant<int, kChunks>{});
            else consume(std::integral_constant<int, 4>{});
    }
    cp_wait<0>();
    __syncwarp();
}

// The partials of a tile's `ks` warps (warp w: tile w / ks, chunk w % ks) summed in chunk order into the chunk-0
// warp's acc (true for that warp). Every warp of the CTA calls it (it holds __syncthreads).
template <int NM, int RG>
__device__ __forceinline__ bool reduce(float (&acc)[NM][RG][4], int ks, uint8_t* smem, int warp_bytes) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (ks == 1) return true;
    float* mine = reinterpret_cast<float*>(smem + warp * warp_bytes);
#pragma unroll
    for (int m = 0; m < NM; m++)
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) mine[((m * RG + r) * 4 + c) * 32 + lane] = acc[m][r][c];
    __syncthreads();
    if (warp % ks != 0) return false;
    for (int w2 = 1; w2 < ks; w2++) {
        const float* o = reinterpret_cast<const float*>(smem + (warp + w2) * warp_bytes);
#pragma unroll
        for (int m = 0; m < NM; m++)
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[m][r][c] += o[((m * RG + r) * 4 + c) * 32 + lane];
    }
    return true;
}

// a warp's bytes of the CTA's dynamic shared memory: its ring, or its partials (the larger)
template <int RB, int NS>
__host__ __device__ constexpr int warpBytes(int nm, int rg) {
    return NS * nm * slotBytes<RB>() > nm * rg * 4 * 32 * 4 ? NS * nm * slotBytes<RB>() : nm * rg * 4 * 32 * 4;
}

// a K split's part c of `parts` of a stage count: stages lo, lo + step, ... below hi. Interleaved (the default): a
// CTA's warps on a tile read neighbouring stages of its rows at once (longer runs of each row in flight); else
// contiguous chunks.
#ifndef TF_DSN_DEC_INTERLEAVE
#define TF_DSN_DEC_INTERLEAVE 1
#endif
__device__ __forceinline__ void chunk(int nst, int parts, int c, int& lo, int& hi, int& step) {
    if (TF_DSN_DEC_INTERLEAVE) {
        lo = c;
        hi = nst;
        step = parts;
        return;
    }
    const int per = (nst + parts - 1) / parts;
    lo = min(nst, c * per);
    hi = min(nst, lo + per);
    step = 1;
}

}  // namespace dec

// The dense decode linears' arguments (FP8 / FP4 / bf16 weights): y [M, N] (row stride ldy, y_dtype; accumulate: fp32
// y +=) = x [M, K] . w [N, K]^T. FP8 / FP4: x quantized (q, s; rows: optional gather of the launch's rows); bf16: x
// bf16 at q (ldq elements), the bf16 GEMM's epilogue (scale, round_first). ks warps a tile in a CTA (blockDim / 32
// warps, blockDim / 32 / ks tiles), cs CTAs a tile group (grid z); cs > 1: the CTAs' partials in z [cs, M, N] and the
// last one's sum, cnt [the tile groups] int zeros (left zero).
struct DecArgs {
    const uint8_t* q;
    const float* s;
    const int* rows;
    int64_t ldq, lds;
    const uint8_t* w;
    const uint8_t* ws;
    int64_t ldw, ldws;  // bytes
    int fmt;
    int M, N, K;
    int ks, cs;
    float* z;
    int* cnt;
    void* y;
    int64_t ldy;
    int y_dtype;
    int accumulate;
    float scale;
    int round_first;
    // block-diagonal groups (wo_a): output columns n read the inputs of group n / gn, at q + group * gq bytes and
    // s + group * gs floats (gn 0: one group)
    int gn;
    int64_t gq, gs;
};

// The quant-into-GEMV prototype's input (tf-dsv41-quant-gemv-bench; no engine path launches it): the bf16 rows x (row
// stride ldx elements; a block-diagonal launch's group g at x + g * gk elements) that a kernel quantizes itself.
struct QxArgs {
    const __nv_bfloat16* x;
    int64_t ldx;
    int64_t gk;
};

// tf_dsn_dec_quant's codes and scales of bf16 rows [M, K] (row stride ldx elements) by the whole CTA, into q (row stride
// ldq bytes) and s (lds floats): a thread a 32-value group (its 64 bytes in four 16-byte reads: ldx and the rows' first
// element multiples of 8), the same arithmetic: each value rounded to bf16 (exact for bf16 rows), the group's largest
// magnitude (a maximum: any order gives the same), its scale and each code.
__device__ __forceinline__ void quant_rows(const __nv_bfloat16* __restrict__ x, int64_t ldx, int M, int K, uint8_t* q,
                                           int64_t ldq, float* s, int lds) {
    const int groups = K / kBlk;
    for (int i = threadIdx.x; i < M * groups; i += blockDim.x) {
        const int m = i / groups, g = i % groups;
        const uint4* src = reinterpret_cast<const uint4*>(x + int64_t(m) * ldx + g * kBlk);
        float v[kBlk];
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const uint4 u = src[c];
            const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __bfloat1622float2(h[j]);
                v[c * 8 + 2 * j] = bf16r(f.x);
                v[c * 8 + 2 * j + 1] = bf16r(f.y);
            }
        }
        float amax = fabsf(v[0]);
#pragma unroll
        for (int j = 1; j < kBlk; j++) amax = fmaxf(amax, fabsf(v[j]));
        const float sc = round_scale(fmaxf(amax, 1e-4f));
        uint32_t word[8];
#pragma unroll
        for (int j = 0; j < 8; j++)
            word[j] = uint32_t(to_e4m3(v[4 * j] / sc)) | uint32_t(to_e4m3(v[4 * j + 1] / sc)) << 8 |
                      uint32_t(to_e4m3(v[4 * j + 2] / sc)) << 16 | uint32_t(to_e4m3(v[4 * j + 3] / sc)) << 24;
        uint4* dst = reinterpret_cast<uint4*>(q + int64_t(m) * ldq + g * kBlk);
        dst[0] = make_uint4(word[0], word[1], word[2], word[3]);
        dst[1] = make_uint4(word[4], word[5], word[6], word[7]);
        s[int64_t(m) * lds + g] = sc;
    }
}

template <int FMT, int RG>
__device__ __forceinline__ void dec_dense(const DecArgs& a, int bx) {
    // (no launch_dependents: the next kernel may be any of the round's, programmatic or not, and some of those
    // complete without waiting for their predecessor, which would let a later kernel overtake this one; only
    // tf_dsn_dec_quant, always followed by one of these kernels, which waits, lets its successor start early)
    extern __shared__ __align__(16) uint8_t dsm[];
    __shared__ int last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wb = dec::warpBytes<dec::kRbD, dec::kNsD>(1, RG);
    const int tiles_cta = (blockDim.x >> 5) / a.ks;
    const int n0 = (bx * tiles_cta + warp / a.ks) * 16;
    const bool on = n0 < a.N;
    int lo = 0, hi = 0, sst = 1;
    const int64_t rb = dec::rowBytes<FMT>(a.K);
    if (on) dec::chunk(int((rb + dec::kRbD - 1) / dec::kRbD), a.ks * a.cs, blockIdx.z * a.ks + warp % a.ks, lo, hi, sst);
    float acc[1][RG][4];
    const uint8_t* w[1] = {a.w};
    const uint8_t* ws[1] = {a.ws};
    const int grp = a.gn && on ? n0 / a.gn : 0;
                          a.s ? a.s + grp * a.gs : a.s, a.lds, a.M,
                          [&](int i) { return a.rows ? a.rows[i] : i; }, dsm + warp * wb, acc);
    const bool lead = dec::reduce<1, RG>(acc, a.ks, dsm, wb);
    if (a.cs > 1) {
        // the CTA's partial into z, then the last CTA of the tile group sums the cs partials in order
        if (lead && on) {
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int m = r * 8 + 2 * t + (c & 1), n = n0 + g + 8 * (c >> 1);
                    if (m < a.M) a.z[(size_t(blockIdx.z) * a.M + m) * a.N + n] = acc[0][r][c];
                }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const int prev = atomicAdd(a.cnt + bx, 1);
            last = prev == a.cs - 1;
            if (last) a.cnt[bx] = 0;
        }
        __syncthreads();
        if (!last) return;
        __threadfence();
        if (!lead || !on) return;
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = r * 8 + 2 * t + (c & 1), n = n0 + g + 8 * (c >> 1);
                if (m >= a.M) continue;
                float v = __ldcg(a.z + size_t(m) * a.N + n);
                for (int p = 1; p < a.cs; p++) v += __ldcg(a.z + (size_t(p) * a.M + m) * a.N + n);
                acc[0][r][c] = v;
            }
    } else if (!lead || !on) {
        return;
    }
#pragma unroll
    for (int r = 0; r < RG; r++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int m = r * 8 + 2 * t + (c & 1), n = n0 + g + 8 * (c >> 1);
            if (m >= a.M) continue;
            float v = acc[0][r][c];
            if (FMT == 2) {
                if (a.round_first) v = bf16r(v);
                if (a.scale != 1.f) v = v * a.scale;
            }
            if (a.accumulate) reinterpret_cast<float*>(a.y)[size_t(m) * a.ldy + n] += v;
            else store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v);
        }
}

// The quant-into-GEMV prototype (FP8, at most 8 rows): dec_dense whose CTA quantizes its own activation rows after the
// wait (qx: their bf16 rows; quant_rows) into shared memory past the warps' rings (codes M rows of K + 16 bytes, then
// scales M rows of K / 32 floats), which the tile loop reads there; a.q, a.s and a.rows unread. Decode linears never
// split K over CTAs (cs 1), so every warp of a tile reads all of its group's K values; a CTA's tiles lie in one group.
__device__ __forceinline__ void dec_dense_qx(const DecArgs& a, int bx, const QxArgs& qx) {
    extern __shared__ __align__(16) uint8_t dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wb = dec::warpBytes<dec::kRbD, dec::kNsD>(1, 1);
    const int tiles_cta = (blockDim.x >> 5) / a.ks;
    const int n0 = (bx * tiles_cta + warp / a.ks) * 16;
    const bool on = n0 < a.N;
    int lo = 0, hi = 0, sst = 1;
    const int64_t rb = dec::rowBytes<0>(a.K);
    if (on) dec::chunk(int((rb + dec::kRbD - 1) / dec::kRbD), a.ks, warp % a.ks, lo, hi, sst);
    float acc[1][1][4];
    const uint8_t* w[1] = {a.w};
    const uint8_t* ws[1] = {a.ws};
    // (16 bytes a row more than K: the 8 rows' 4-byte reads of one K block take 8 bank groups)
    uint8_t* xq = dsm + (blockDim.x >> 5) * wb;
    const int64_t ldq = a.K + 16;
    float* xs = reinterpret_cast<float*>(xq + int64_t(a.M) * ldq);
    const int grp = a.gn ? bx * tiles_cta * 16 / a.gn : 0;
    auto pro = [&]() {
        quant_rows(qx.x + grp * qx.gk, qx.ldx, a.M, a.K, xq, ldq, xs, a.K / kBlk);
        __syncthreads();
    };
    auto rows = [](int i) { return i; };
                                                                   xs, a.K / kBlk, a.M, rows, dsm + warp * wb, acc, pro);
    if (!dec::reduce<1, 1>(acc, a.ks, dsm, wb) || !on) return;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int m = 2 * t + (c & 1), n = n0 + g + 8 * (c >> 1);
        if (m >= a.M) continue;
        const float v = acc[0][0][c];
        if (a.accumulate) reinterpret_cast<float*>(a.y)[size_t(m) * a.ldy + n] += v;
        else store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v);
    }
}

// gate / up (GuArgs; splits: the warps a tile and matrix): the pack's 16-column tiles of expert act[blockIdx.y], each
// tile's gate (w1) and up (w3) on warps of their own (warp w: tile w / (2 splits), matrix w / splits % 2, chunk
// w % splits), the tile's partials summed in chunk order, then the epilogue on the gate's lead warp. An expert's
// entries go in passes of 8 RG.
template <int RG>
__device__ __forceinline__ void dec_gu(const GuArgs& a) {
    // (no launch_dependents: the next kernel may be any of the round's, programmatic or not, and some of those
    // complete without waiting for their predecessor, which would let a later kernel overtake this one; only
    // tf_dsn_dec_quant, always followed by one of these kernels, which waits, lets its successor start early)
    extern __shared__ __align__(16) uint8_t dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wb = dec::warpBytes<dec::kRbG, dec::kNsG>(1, RG);
    const int ks = a.splits, per_tile = 2 * ks;
    // the experts in reverse id order: the shared expert (the last id, the L2 prefetch's) first
    const int na = a.nact[0];
    // Track C (docs/UNIFIED-DECODE-IMPROVEMENTS.md, TF_DS_EXPERT_EXIT=1): the grid's rows past the active experts
    // (it is sized min(experts, rows x slots) on the host, which cannot see the count) leave at once, the whole CTA
    // together (no barrier is entered, nothing is written, no other CTA waits on them)
    if (a.early_exit && int(blockIdx.y) >= na) return;
    const bool live = int(blockIdx.y) < na;
    const int e = live ? a.act[na - 1 - blockIdx.y] : 0;
    const int tiles_cta = (blockDim.x >> 5) / per_tile;
    const int n0 = (blockIdx.x * tiles_cta + warp / per_tile) * 16;
    const int mat = warp / ks % 2, ch = warp % ks;
    const int ncols = a.n_tiles * 32;
    bool on = live && n0 < ncols;
    if (on && a.owner && n0 / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) on = false;
    const bool fp8 = e == a.shared;
    const int base = live ? a.off[e] : 0, all = live ? a.off[e + 1] - base : 0;
    const uint8_t* w[1] = {reinterpret_cast<const uint8_t*>(mat ? a.w3[e] : a.w1[e])};
    const uint8_t* ws[1] = {reinterpret_cast<const uint8_t*>(mat ? a.s3[e] : a.s1[e])};
    for (int p0 = 0; p0 < max(all, 1); p0 += 8 * RG) {
        const int cnt = on ? min(8 * RG, all - p0) : 0;
        auto rows_of = [&](int i) { return a.erow[base + p0 + i]; };
        float acc[1][RG][4];
        int lo = 0, hi = 0, sst = 1;
        if (fp8) {
            const int64_t rb = dec::rowBytes<0>(a.D);
            if (on) dec::chunk(int((rb + dec::kRbG - 1) / dec::kRbG), ks, ch, lo, hi, sst);
                                dsm + warp * wb, acc);
        } else {
            const int64_t rb = dec::rowBytes<1>(a.D);
            if (on) dec::chunk(int((rb + dec::kRbG - 1) / dec::kRbG), ks, ch, lo, hi, sst);
                                dsm + warp * wb, acc);
        }
        // every warp's partials into its region, then the tile's lead warp (matrix 0, chunk 0) sums each matrix's
        float* mine = reinterpret_cast<float*>(dsm + warp * wb);
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) mine[(r * 4 + c) * 32 + lane] = acc[0][r][c];
        __syncthreads();
        if (warp % per_tile == 0 && on) {
            float gs[RG][4], us[RG][4];
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) gs[r][c] = us[r][c] = 0.f;
            for (int w2 = 0; w2 < ks; w2++) {
                const float* og = reinterpret_cast<const float*>(dsm + (warp + w2) * wb);
                const float* ou = reinterpret_cast<const float*>(dsm + (warp + ks + w2) * wb);
#pragma unroll
                for (int r = 0; r < RG; r++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        gs[r][c] = w2 ? gs[r][c] + og[(r * 4 + c) * 32 + lane] : og[(r * 4 + c) * 32 + lane];
                        us[r][c] = w2 ? us[r][c] + ou[(r * 4 + c) * 32 + lane] : ou[(r * 4 + c) * 32 + lane];
                    }
            }
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int i = r * 8 + 2 * t + (c & 1);
                    if (i >= cnt) continue;
                    gu_store(a, a.ent[base + p0 + i], n0 + g + 8 * (c >> 1), gs[r][c], us[r][c]);
                }
        }
        __syncthreads();
    }
}

// down (DnArgs; splits: the warps a tile): 16 output columns a tile of expert act[blockIdx.y]
template <int RG>
__device__ __forceinline__ void dec_dn(const DnArgs& a) {
    // (no launch_dependents: the next kernel may be any of the round's, programmatic or not, and some of those
    // complete without waiting for their predecessor, which would let a later kernel overtake this one; only
    // tf_dsn_dec_quant, always followed by one of these kernels, which waits, lets its successor start early)
    extern __shared__ __align__(16) uint8_t dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wb = dec::warpBytes<dec::kRbE, dec::kNsE>(1, RG);
    if (a.early_exit && int(blockIdx.y) >= a.nact[0]) return;  // Track C: as dec_gu
    const bool live = int(blockIdx.y) < a.nact[0];
    const int e = live ? a.act[blockIdx.y] : 0;
    const int tiles_cta = (blockDim.x >> 5) / a.splits;
    const int n0 = (blockIdx.x * tiles_cta + warp / a.splits) * 16;
    const bool on = live && n0 < a.N;
    const bool fp8 = e == a.shared;
    const int base = live ? a.off[e] : 0, all = live ? a.off[e + 1] - base : 0;
    const uint8_t* w[1] = {reinterpret_cast<const uint8_t*>(a.w2[e])};
    const uint8_t* ws[1] = {reinterpret_cast<const uint8_t*>(a.s2[e])};
    for (int p0 = 0; p0 < max(all, 1); p0 += 8 * RG) {
        const int cnt = on ? min(8 * RG, all - p0) : 0;
        auto rows_of = [&](int i) { return a.ent[base + p0 + i]; };
        float acc[1][RG][4];
        int lo = 0, hi = 0, sst = 1;
        if (fp8) {
            const int64_t rb = dec::rowBytes<0>(a.I);
            if (on) dec::chunk(int((rb + dec::kRbE - 1) / dec::kRbE), a.splits, warp % a.splits, lo, hi, sst);
            dec::core<dec::kRbE, dec::kNsE, 0, 1, RG>(w, ws, a.sldw, a.sldws, rb, on ? n0 : 0, lo, hi, sst, a.q, a.ldq, a.s, a.lds, cnt, rows_of,
                                dsm + warp * wb, acc);
        } else {
            const int64_t rb = dec::rowBytes<1>(a.I);
            if (on) dec::chunk(int((rb + dec::kRbE - 1) / dec::kRbE), a.splits, warp % a.splits, lo, hi, sst);
            dec::core<dec::kRbE, dec::kNsE, 1, 1, RG>(w, ws, a.ldw, a.ldws, rb, on ? n0 : 0, lo, hi, sst, a.q, a.ldq, a.s, a.lds, cnt, rows_of,
                                dsm + warp * wb, acc);
        }
        const bool lead = dec::reduce<1, RG>(acc, a.splits, dsm, wb);
        if (lead && on) {
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int i = r * 8 + 2 * t + (c & 1);
                    if (i >= cnt) continue;
                    const float v = acc[0][r][c];
                    a.out[size_t(a.ent[base + p0 + i]) * a.ldo + n0 + g + 8 * (c >> 1)] = a.round_bf16 ? bf16r(v) : v;
                }
        }
        if (a.splits > 1) __syncthreads();  // the regions reused by the next pass
    }
}

// Up to four dense linears of one format and row count in one launch (the linears of one input: a launch and its
// tail instead of each's): linear p takes the CTAs cta0[p] .. cta0[p + 1] - 1, each as its own launch would.
struct DecMulti {
    DecArgs p[4];
    int n;
    int cta0[5];
};

template <int FMT, int RG>
__device__ __forceinline__ void dec_multi(const DecMulti& m) {
    int p = 0;
    while (p + 1 < m.n && int(blockIdx.x) >= m.cta0[p + 1]) p++;
    dec_dense<FMT, RG>(m.p[p], blockIdx.x - m.cta0[p]);
}

    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_fp8_r##RG##_kernel(DecMulti a) { dec_multi<0, RG>(a); } \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_fp4_r##RG##_kernel(DecMulti a) { dec_multi<1, RG>(a); } \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_bf_r##RG##_kernel(DecMulti a) { dec_multi<2, RG>(a); }  \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_gu_r##RG##_kernel(GuArgs a) { dec_gu<RG>(a); }         \
#undef TF_DSN_DEC

// The quant-into-GEMV prototype (tf-dsv41-quant-gemv-bench; no engine path launches it): tf_dsn_dec_fp8_r1's launch of
// up to four FP8 linears on one input, each CTA quantizing that input's bf16 rows itself (qx) after its wait instead of
// reading tf_dsn_dec_quant's codes; the grid, the split and the dynamic shared memory tf_dsn_dec_fp8_r1's, plus the
// codes and scales (M (K + 16) bytes and M K / 8 bytes).
extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_fp8x_r1_kernel(DecMulti m, QxArgs qx) {
    int p = 0;
    while (p + 1 < m.n && int(blockIdx.x) >= m.cta0[p + 1]) p++;
    dec_dense_qx(m.p[p], blockIdx.x - m.cta0[p], qx);
}

// tf-dsv41-quant-gemv-bench's producer of a decode linear's input rows (as _kv_norm_rope is to wq_b's quant): n 16-byte
// words from src to dst, waiting for its predecessor and then letting its successor start, as the kit's kernels do.
extern "C" __global__ void tf_dsn_bench_rows_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, int n) {
    dec::pdl_wait();
    dec::pdl_launch();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

// act_quant (tf_dsn_act_quant's codes and scales, the same arithmetic) for decode launches: an ordinary launch that
// lets the next kernel (a decode tensor-core kernel, launched as a programmatic dependent launch) start its weights'
// prologue while this runs. (Itself not a dependent launch: a predecessor of another kind may let it start early yet
// complete before its own predecessor, so only an ordinary launch here keeps every earlier kernel done.)
extern "C" __global__ void tf_dsn_dec_quant_kernel(const void* __restrict__ x, int x_dtype, int64_t ldx, int M, int K,
                                                   uint8_t* __restrict__ q, int64_t ldq, float* __restrict__ s,
                                                   int64_t lds) {
    dec::pdl_launch();
    dec::pdl_wait();
    const int groups = K / kBlk;
    const int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= int64_t(M) * groups) return;
    const int m = int(warp / groups), g = int(warp % groups);
    const int k = g * kBlk + lane;
    const float v = bf16r(load_x(x, x_dtype, size_t(m) * ldx + k));
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = round_scale(fmaxf(amax, 1e-4f));
    q[size_t(m) * ldq + k] = to_e4m3(v / sc);
    if (lane == 0) s[size_t(m) * lds + g] = sc;
}

// The active experts' down weights touched into L2 while the 2D exchange of the intermediate runs (it leaves DRAM
// idle), in down's CTA order (act[] ascending), up to `budget` bytes: 8 KiB bulk L2 prefetches, one block of 128
// threads, a wave of 128 every wave_ns (DRAM's queue about a wave deep; chunk_bytes: the bulk prefetch's size, 0 = 8 KiB). w2 [E] pointers, the routed experts' down
// `bytes`, the shared expert's (id `shared`) `sbytes`. Writes nothing.
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dec_l2pf_kernel(const int* __restrict__ act,
                                                                         const int* __restrict__ nact,
                                                                         const uint64_t* __restrict__ w2, long long bytes,
                                                                         long long sbytes, int shared, long long budget,
                                                                         long long wave_ns, long long chunk_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const long long chunk = chunk_bytes > 0 ? chunk_bytes : (8 << 10);  // (TF_DS_L2_CHUNK_KIB; 0: the served 8 KiB)
    const int n = nact[0];
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    long long done = 0, wave = 0;
    int i = 0;
    long long off = 0;  // within expert act[i]
    while (i < n && done < budget) {
        if (wave_ns > 0) {
            __syncthreads();
            if (threadIdx.x == 0) {
                for (;;) {
                    unsigned long long t;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                    if (t - t0 >= (unsigned long long)(wave * wave_ns)) break;
                    __nanosleep(256);
                }
            }
            __syncthreads();
            ++wave;
        }
        // this wave's blockDim chunks: thread j takes the j-th chunk after (i, off)
        int ii = i;
        long long oo = off + (long long)threadIdx.x * chunk;
        long long len = 0;
        while (ii < n) {
            len = (act[ii] == shared ? sbytes : bytes) / 16 * 16;
            if (oo < len) break;
            oo -= len;
            ++ii;
        }
        if (ii < n && done + (long long)threadIdx.x * chunk < budget) {
            const long long left = len - oo;
            const unsigned size = (unsigned)(left < chunk ? left : chunk);
            const char* p = reinterpret_cast<const char*>(w2[act[ii]]) + oo;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(size) : "memory");
        }
        // advance (i, off) by a wave
        off += (long long)blockDim.x * chunk;
        done += (long long)blockDim.x * chunk;
        while (i < n) {
            const long long l = (act[i] == shared ? sbytes : bytes) / 16 * 16;
            if (off < l) break;
            off -= l;
            ++i;
        }
    }
#endif
}

// A decode launch's grouping in one CTA (tf_dsn_group_count, _scan and _place of the cnt memset together): the picks
// pick [RS] (ids past E or negative: another node's) counted in shared memory, off [E + 1] their exclusive prefix,
// act [] the experts with a pick in id order and nact [0] their count, then each entry's place: ent [off[e] + i] =
// the entry's index, erow its row (i: the entries of e before it, so in entry order). blockDim 1024, RS <= 4096.
extern "C" __global__ void __launch_bounds__(1024) tf_dsn_dec_group_kernel(const int* __restrict__ pick, int RS, int S,
                                                                           int E, int* __restrict__ off,
                                                                           int* __restrict__ act, int* __restrict__ nact,
                                                                           int* __restrict__ ent, int* __restrict__ erow) {
    extern __shared__ int sh[];  // cnt [E], then off [E + 1]
    int* cnt = sh;
    int* so = sh + E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < RS; i += blockDim.x) {
        const int e = pick[i];
        if (e >= 0 && e < E) atomicAdd(cnt + e, 1);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        // the prefix and the active list, a warp over E in steps of 32 (ballot order: id order)
        const int lane = threadIdx.x;
        int base = 0, nb = 0;
        for (int e0 = 0; e0 < E; e0 += 32) {
            const int e = e0 + lane;
            const int c = e < E ? cnt[e] : 0;
            int x = c;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (e < E) so[e] = base + x - c;
            const unsigned m = __ballot_sync(0xffffffffu, c != 0);
            if (c != 0) act[nb + __popc(m & ((1u << lane) - 1))] = e;
            nb += __popc(m);
            base += __shfl_sync(0xffffffffu, x, 31);
        }
        if (lane == 0) {
            so[E] = base;
            nact[0] = nb;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e <= E; e += blockDim.x) off[e] = so[e];
    for (int i = threadIdx.x; i < RS; i += blockDim.x) {
        const int e = pick[i];
        if (e < 0 || e >= E) continue;
        int before = 0;
        for (int j = 0; j < i; j++) before += pick[j] == e;
        const int at = so[e] + before;
        ent[at] = i;
        erow[at] = i / S;
    }
}

// tf_dsn_dec_group_kernel's outputs by a scan of one expert a thread and a ranked placement (TF_DS_DEC_GROUP2): the same
// off, act, nact, ent and erow for the same picks (blockDim 1024, E <= 1024, RS <= 4096). The picks are read from global
// memory once, into shared memory. The counts are shared-memory atomics (their sums do not depend on the order). The
// exclusive prefix of the counts, and the compaction of the experts with a pick (act in id order), take one warp scan of
// the 32 experts each thread holds and one scan of the 32 warp totals. The entries are then placed by warp 0 in steps of
// 32 entries: __match_any_sync gives each entry the entries of its warp step that share its expert, the rank among them
// is the count of lower lanes, and the expert's running count carries across steps, so an entry's place is the number
// of entries before it with its expert, as the one-CTA kernel's loop over every earlier entry counts it.
extern "C" __global__ void __launch_bounds__(1024) tf_dsn_dec_group2_kernel(const int* __restrict__ pick, int RS, int S,
                                                                            int E, int* __restrict__ off,
                                                                            int* __restrict__ act, int* __restrict__ nact,
                                                                            int* __restrict__ ent, int* __restrict__ erow) {
    extern __shared__ int sh[];  // cnt [E] (then each expert's running count), so [E + 1]
    int* cnt = sh;
    int* so = sh + E;
    __shared__ int sp[4096];
    __shared__ int wtot[32], wact[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int e = tid; e < E; e += blockDim.x) cnt[e] = 0;
    for (int i = tid; i < RS; i += blockDim.x) sp[i] = pick[i];
    __syncthreads();
    for (int i = tid; i < RS; i += blockDim.x) {
        const int e = sp[i];
        if (e >= 0 && e < E) atomicAdd(cnt + e, 1);
    }
    __syncthreads();
    const int c = tid < E ? cnt[tid] : 0;
    int x = c;  // inclusive scan of the warp's counts
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    const unsigned m = __ballot_sync(0xffffffffu, c != 0);
    if (lane == 31) {
        wtot[warp] = x;
        wact[warp] = __popc(m);
    }
    __syncthreads();
    if (warp == 0) {
        // the 32 warp totals' exclusive scans; lane 31 also holds the totals
        int t = wtot[lane], a = wact[lane];
        int xt = t, xa = a;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int yt = __shfl_up_sync(0xffffffffu, xt, o);
            const int ya = __shfl_up_sync(0xffffffffu, xa, o);
            if (lane >= o) {
                xt += yt;
                xa += ya;
            }
        }
        wtot[lane] = xt - t;
        wact[lane] = xa - a;
        if (lane == 31) {
            off[E] = xt;
            nact[0] = xa;
        }
    }
    __syncthreads();
    if (tid < E) {
        const int at = x - c + wtot[warp];
        so[tid] = at;
        off[tid] = at;
        cnt[tid] = 0;  // the running count of the placement
        if (c != 0) act[wact[warp] + __popc(m & ((1u << lane) - 1))] = tid;
    }
    __syncthreads();
    if (warp == 0) {
        for (int i0 = 0; i0 < RS; i0 += 32) {
            const int i = i0 + lane;
            const int e = i < RS ? sp[i] : -1;
            const bool v = e >= 0 && e < E;
            const unsigned peers = __match_any_sync(0xffffffffu, v ? e : -1);
            const int before = v ? cnt[e] + __popc(peers & ((1u << lane) - 1)) : 0;
            __syncwarp();
            if (v && lane == __ffs(peers) - 1) cnt[e] += __popc(peers);
            if (v) {
                const int at = so[e] + before;
                ent[at] = i;
                erow[at] = i / S;
            }
            __syncwarp();
        }
    }
}

// TF_DS_NACT_LOG (log only; launched by mx.nactLog, never when the log is off): one record a decode launch's grouping, what the weight bytes of
// the launch's gate / up and down depend on: the rows, the active experts (nact, distinct across the rows), how many of the routed ones this node
// owns (an unowned expert's rest tiles are not read) and whether the shared expert is among them (sh: 1 active and unowned, 2 active and owned).
// ring: word 0 the call counter, 8 words of header, then 4 words a record {call + 1, rows << 32 | nact, sh << 32 | owned, 0}.
extern "C" __global__ void __launch_bounds__(256) tf_dsn_nact_log_kernel(const int* __restrict__ act, const int* __restrict__ nact_p,
                                                                       const int* __restrict__ owner, int shared, int rows,
                                                                       unsigned long long* ring) {
    __shared__ int owned, sh;
    const int n = nact_p[0];
    if (threadIdx.x == 0) {
        owned = 0;
        sh = 0;
    }
    __syncthreads();
    int mine = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int e = act[j];
        if (e == shared) atomicAdd(&sh, owner == nullptr || owner[e] != 0 ? 2 : 1);
        else if (owner == nullptr || owner[e] != 0) mine++;
    }
    atomicAdd(&owned, mine);
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long i = atomicAdd(ring, 1ull);
        unsigned long long* r = ring + 8 + (i & 4095) * 4;
        r[0] = i + 1;
        r[1] = ((unsigned long long)rows << 32) | (unsigned)n;
        r[2] = ((unsigned long long)sh << 32) | (unsigned)owned;
        r[3] = 0;
    }
}

// the layouts mx.zig passes (its extern structs assert the same offsets)
static_assert(sizeof(MxArgs) == 136 && offsetof(MxArgs, z) == 96 && offsetof(MxArgs, raster) == 128, "MxArgs layout");
static_assert(offsetof(GuArgs, sldw) == 136 && offsetof(GuArgs, owner) == 152 && offsetof(GuArgs, z) == 216 && offsetof(GuArgs, early_exit) == 224 && sizeof(GuArgs) == 232, "GuArgs layout");
static_assert(offsetof(DnArgs, sldw) == 104 && offsetof(DnArgs, out) == 136 && offsetof(DnArgs, z) == 160 && offsetof(DnArgs, early_exit) == 172 && sizeof(DnArgs) == 176, "DnArgs layout");
static_assert(sizeof(DecMulti) == 4 * 168 + 24 && offsetof(DecMulti, cta0) == 676, "DecMulti layout");
static_assert(sizeof(DecArgs) == 168 && offsetof(DecArgs, gq) == 152 && offsetof(DecArgs, z) == 96 && offsetof(DecArgs, y) == 112, "DecArgs layout");
static_assert(offsetof(BfArgs, y) == 64 && offsetof(BfArgs, z) == 112 && offsetof(BfArgs, raster) == 120 && sizeof(BfArgs) == 128, "BfArgs layout");
