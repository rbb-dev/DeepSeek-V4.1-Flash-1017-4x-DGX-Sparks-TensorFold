// Byte-identity harness for the decode tensor-core kernels of zig/kernels/cuda/dsv41/native.cu (not part of the engine).
// Launches every dec kernel (dense fp8 / fp4 / bf16, the experts' gate-up and down, with and without the fused routed
// sum) on seeded random data at the 2D node's real layer shapes and at row counts that cover every row group (1 .. 64),
// every K split the engine uses plus the others the bench tries, and prints one FNV-1a hash per output buffer. Build it
// against two versions of native.cu and diff the two outputs: any difference is a changed byte.
//   nvcc -O3 -std=c++20 -arch=sm_120 -DNATIVE_CU='"/path/native.cu"' -o dec_equal dec_equal.cu && ./dec_equal > out.txt
// A control (CONTROL=1) flips one activation byte: the dense and expert hashes must change.
#include NATIVE_CU
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <array>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

enum Fill { F_FP4, F_FP8, F_SCALE, F_ONE, F_ZERO, F_BF16 };
__global__ void fill_kernel(uint8_t* p, size_t n, int f, uint32_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t r = uint32_t(i) * 2654435761u ^ seed;
        r ^= r >> 15; r *= 2246822519u; r ^= r >> 13; r *= 3266489917u; r ^= r >> 16;
        if (f == F_ONE) {
            if (i % 4 == 0 && i + 3 < n) *reinterpret_cast<float*>(p + i) = 1.0f / float(1 << (r % 4));
            continue;
        }
        if (f == F_BF16) {
            // bf16 values: sign random, exponent 118..127 (|x| in 2^-9 .. 1), mantissa random
            if (i % 2 == 0 && i + 1 < n) {
                uint16_t v = uint16_t((r & 0x8000) | ((118 + (r >> 4) % 10) << 7) | (r & 0x7f));
                *reinterpret_cast<uint16_t*>(p + i) = v;
            }
            continue;
        }
        p[i] = f == F_FP4 ? uint8_t(r) : f == F_FP8 ? (uint8_t(r) & 0xBF) : f == F_SCALE ? uint8_t(124 + r % 7) : 0;
    }
}
static uint32_t rng = 777;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void* dbuf(size_t n, Fill f) {
    void* p;
    CK(cudaMalloc(&p, n));
    CK(cudaMemset(p, 0, n));
    fill_kernel<<<256, 256>>>((uint8_t*)p, n, f, rnd());
    CK(cudaGetLastError());
    return p;
}
static uint64_t fnv(const void* dptr, size_t n) {
    std::vector<uint8_t> h(n);
    CK(cudaMemcpy(h.data(), dptr, n, cudaMemcpyDeviceToHost));
    uint64_t x = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { x ^= h[i]; x *= 1099511628211ull; }
    return x;
}
static cudaStream_t st;
static int rgOf(int R) { return R <= 8 ? 1 : R <= 16 ? 2 : R <= 32 ? 4 : 8; }
#define DEC_PICK(kind, rg) (rg == 1 ? (void*)tf_dsn_dec_##kind##_r1_kernel : rg == 2 ? (void*)tf_dsn_dec_##kind##_r2_kernel : rg == 4 ? (void*)tf_dsn_dec_##kind##_r4_kernel : (void*)tf_dsn_dec_##kind##_r8_kernel)
static void* decFn(const char* kind, int rg) {
    if (!strcmp(kind, "fp8")) return DEC_PICK(fp8, rg);
    if (!strcmp(kind, "fp4")) return DEC_PICK(fp4, rg);
    if (!strcmp(kind, "gu")) return DEC_PICK(gu, rg);
    if (!strcmp(kind, "dn")) return DEC_PICK(dn, rg);
    return DEC_PICK(bf, rg);
}
static const int warps_cta = 8;
static void decLaunch(const char* kind, int rg, int nm, dim3 grid, void* args) {
    void* f = decFn(kind, rg);
    const bool ex = !strcmp(kind, "gu") || !strcmp(kind, "dn");
    const int smem = warps_cta * (!strcmp(kind, "gu") ? dec::warpBytes<dec::kRbG, dec::kNsG>(nm, rg) : ex ? dec::warpBytes<dec::kRbE, dec::kNsE>(nm, rg) : dec::warpBytes<dec::kRbD, dec::kNsD>(nm, rg));
    CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    DecMulti mm{};
    if (!ex) { mm.p[0] = *reinterpret_cast<DecArgs*>(args); mm.n = 1; args = &mm; }
    void* pp[1] = {args};
    CK(cudaLaunchKernel(f, grid, dim3(32 * warps_cta), pp, smem, st));
}
struct Dense { const char* name; int fmt, K, N; };

int main() {
    CK(cudaStreamCreate(&st));
    const bool control = getenv("CONTROL") != nullptr;
    const int maxR = 64, maxK = 8192;
    uint8_t* q = (uint8_t*)dbuf(size_t(maxR) * maxK, F_FP8);
    float* s = (float*)dbuf(size_t(maxR) * maxK / 32 * 4, F_ONE);
    float* y = (float*)dbuf(size_t(maxR) * 40000 * 4, F_ZERO);
    float* z = (float*)dbuf(size_t(64) << 20, F_ZERO);
    __nv_bfloat16* xb = (__nv_bfloat16*)dbuf(size_t(maxR) * maxK * 2, F_BF16);
    int* cnt0 = (int*)dbuf(4 * 65536, F_ZERO);
    if (control) { uint8_t b = 0x38; CK(cudaMemcpy(q + 3, &b, 1, cudaMemcpyHostToDevice)); }
    const Dense dense[] = {
        {"wq_a fp8 5120->1280", 0, 5120, 1280},     {"wkv fp8 5120->512", 0, 5120, 512},
        {"wq_b fp8 1280->8192", 0, 1280, 8192},     {"idx_wq_b fp8 1280->4096", 0, 1280, 4096},
        {"wo_b fp8 4096->2560", 0, 4096, 2560},     {"wo_a(fp8) 4096->1024", 0, 4096, 1024},
        {"shared gu fp8 5120->640", 0, 5120, 640},  {"shared dn fp8 1152->2560", 0, 1152, 2560},
        {"expert gu fp4 5120->640", 1, 5120, 640},  {"expert dn fp4 1152->2560", 1, 1152, 2560},
        {"wo_a bf16 4096->1024", 2, 4096, 1024},    {"head bf16 5120->32320", 2, 5120, 32320},
        {"router bf16 5120->384", 2, 5120, 384},    {"comp_wkv bf16 5120->512", 2, 5120, 512},
        {"head fp4 5120->32320", 1, 5120, 32320},
    };
    const int rows_list[] = {1, 3, 6, 8, 9, 16, 17, 33, 64};
    const std::array<int, 2> kcs[] = {{8, 1}, {4, 2}, {8, 2}, {2, 1}, {1, 1}};
    // ONLY=substring: only the dense shapes whose name has it ("experts" for the experts alone; "none" for neither);
    // ROWS=a,b: only those row counts; KS=8: only that K split (all with cs 1): a profiler's (ncu) way to one launch
    const char* only = getenv("ONLY");
    std::vector<int> rows_sel;
    if (getenv("ROWS")) for (char* q2 = getenv("ROWS"); *q2;) { rows_sel.push_back(strtol(q2, &q2, 10)); if (*q2) q2++; }
    const int ks_sel = getenv("KS") ? atoi(getenv("KS")) : 0;
    auto rows_ok = [&](int R) { if (rows_sel.empty()) return true; for (int x : rows_sel) if (x == R) return true; return false; };
    for (const Dense& d : dense) {
        if (only && !strstr(d.name, only)) continue;
        const size_t wbytes = d.fmt == 1 ? size_t(d.N) * d.K / 2 : d.fmt == 0 ? size_t(d.N) * d.K : size_t(d.N) * d.K * 2;
        const size_t sbytes = d.fmt == 1 ? size_t(d.N) * d.K / 32 : d.fmt == 0 ? size_t(d.N / 32) * d.K / 32 : 0;
        void* w = dbuf(wbytes, d.fmt == 1 ? F_FP4 : d.fmt == 0 ? F_FP8 : F_BF16);
        void* ws = sbytes ? dbuf(sbytes, F_SCALE) : nullptr;
        for (int R : rows_list) {
            if (!rows_ok(R)) continue;
            for (auto kc : kcs) {
                const int ks = kc[0], cs = kc[1];
                if (ks_sel && (ks != ks_sel || cs != 1)) continue;
                if (cs > 1 && R > 16) continue;
                // the engine's rule: each warp keeps at least two stages of a row (decSplit)
                const int nst = (d.fmt == 1 ? d.K / 2 : d.fmt == 0 ? d.K : 2 * d.K + 63) / 64;
                if (ks > 1 && nst < 2 * ks * cs) continue;
                const int ntiles = d.N / 16, tpc = warps_cta / ks;
                for (int yd = 0; yd < 2; yd++) {  // fp32 / bf16 outputs (and accumulate)
                    CK(cudaMemset(y, 0, size_t(R) * d.N * 4 + 64));
                    CK(cudaMemset(cnt0, 0, 4 * 65536));
                    DecArgs a{};
                    a.ldq = d.K; a.lds = d.K / 32; a.fmt = d.fmt; a.M = R; a.N = d.N; a.K = d.K;
                    a.y = y; a.ldy = d.N; a.y_dtype = yd ? 2 : 0; a.ks = ks; a.cs = cs; a.z = z; a.cnt = cnt0; a.scale = 1.f;
                    a.accumulate = 0;
                    if (d.fmt == 2) {
                        a.q = (const uint8_t*)xb; a.w = (const uint8_t*)w; a.ldw = d.K * 2; a.round_first = yd; a.scale = yd ? 0.5f : 1.f;
                        a.y_dtype = yd ? 2 : 0;
                    } else {
                        a.q = q; a.s = s; a.w = (const uint8_t*)w; a.ws = (const uint8_t*)ws;
                        a.ldw = d.fmt == 1 ? d.K / 2 : d.K; a.ldws = d.K / 32;
                    }
                    if (d.fmt != 2 && yd == 0 && (R & 1)) {  // a gathered row order and block-diagonal groups on some launches
                        static int* rows = nullptr;
                        if (!rows) {
                            std::vector<int> h(64);
                            for (int i = 0; i < 64; i++) h[i] = (i * 37 + 5) % 64;
                            CK(cudaMalloc(&rows, 256));
                            CK(cudaMemcpy(rows, h.data(), 256, cudaMemcpyHostToDevice));
                        }
                        a.rows = rows;
                    }
                    decLaunch(d.fmt == 1 ? "fp4" : d.fmt == 0 ? "fp8" : "bf", rgOf(R), 1, dim3((ntiles + tpc - 1) / tpc, 1, cs), &a);
                    CK(cudaGetLastError());
                    CK(cudaStreamSynchronize(st));
                    const size_t esz = a.y_dtype == 2 ? 2 : 4;
                    printf("dense %-26s R=%-2d ks=%d cs=%d yd=%d %016llx\n", d.name, R, ks, cs, yd, (unsigned long long)fnv(y, size_t(R) * d.N * esz));
                }
            }
        }
        cudaFree(w);
        if (ws) cudaFree(ws);
    }

    // ---- experts: a 2D node's layer (D 5120, IH 1152, DN 2560; 384 routed + 1 shared)
    if (!only || !strcmp(only, "experts")) {
        const int E = 385, D = 5120, IH = 1152, DN = 2560, MAIN = 512, REST = 128, S = 7;
        std::vector<uint64_t> w1, w3, s1, s3, w2, s2;
        std::vector<int> owner_h(E);
        for (int e = 0; e < E; e++) owner_h[e] = (e % 2) == 0;
        for (int e = 0; e < E; e++) {
            const bool sh = e == E - 1;
            const int gr = MAIN + REST;
            const size_t gub = sh ? size_t(gr) * D : size_t(gr) * D / 2;
            const size_t gus = sh ? size_t(gr / 32) * (D / 32) : size_t(gr) * (D / 32);
            const size_t dnb = sh ? size_t(DN) * IH : size_t(DN) * IH / 2;
            const size_t dns = sh ? size_t(DN / 32) * (IH / 32) : size_t(DN) * (IH / 32);
            w1.push_back((uint64_t)dbuf(gub, sh ? F_FP8 : F_FP4));
            w3.push_back((uint64_t)dbuf(gub, sh ? F_FP8 : F_FP4));
            s1.push_back((uint64_t)dbuf(gus, F_SCALE));
            s3.push_back((uint64_t)dbuf(gus, F_SCALE));
            w2.push_back((uint64_t)dbuf(dnb, sh ? F_FP8 : F_FP4));
            s2.push_back((uint64_t)dbuf(dns, F_SCALE));
        }
        auto up = [&](std::vector<uint64_t>& v) { uint64_t* p; CK(cudaMalloc(&p, 8 * E)); CK(cudaMemcpy(p, v.data(), 8 * E, cudaMemcpyHostToDevice)); return p; };
        uint64_t *dw1 = up(w1), *dw3 = up(w3), *ds1 = up(s1), *ds3 = up(s3), *dw2 = up(w2), *ds2 = up(s2);
        int* owner;
        CK(cudaMalloc(&owner, 4 * E));
        CK(cudaMemcpy(owner, owner_h.data(), 4 * E, cudaMemcpyHostToDevice));
        int *pick, *cnt, *off, *fill, *act, *nact, *ent, *erow, *ccnt;
        CK(cudaMalloc(&pick, 4 * maxR * S));
        CK(cudaMalloc(&cnt, 4 * E)); CK(cudaMalloc(&off, 4 * (E + 1))); CK(cudaMalloc(&fill, 4 * E));
        CK(cudaMalloc(&act, 4 * E)); CK(cudaMalloc(&nact, 4)); CK(cudaMalloc(&ent, 4 * maxR * S)); CK(cudaMalloc(&erow, 4 * maxR * S));
        CK(cudaMalloc(&ccnt, 4 * 4096)); CK(cudaMemset(ccnt, 0, 4 * 4096));
        float* wts = (float*)dbuf(4 * maxR * S, F_ONE);
        __nv_bfloat16* h = (__nv_bfloat16*)dbuf(size_t(maxR) * S * IH * 2, F_ZERO);
        uint8_t* hq = (uint8_t*)dbuf(size_t(maxR) * S * IH, F_FP8);
        float* hs = (float*)dbuf(size_t(maxR) * S * IH / 32 * 4, F_ONE);
        float* out = (float*)dbuf(size_t(maxR) * S * DN * 4, F_ZERO);
        float* yc = (float*)dbuf(size_t(maxR) * DN * 4, F_ZERO);
        for (int R : rows_list) {
            if (!rows_ok(R)) continue;
            std::vector<int> ph(R * S);
            for (int r = 0; r < R; r++) {
                for (int k = 0; k < 6; k++) {
                    int e; bool dup;
                    do { e = rnd() % (E - 1); dup = false; for (int j = 0; j < k; j++) dup |= ph[r * S + j] == e; } while (dup);
                    ph[r * S + k] = e;
                }
                ph[r * S + 6] = E - 1;
            }
            CK(cudaMemcpy(pick, ph.data(), 4 * R * S, cudaMemcpyHostToDevice));
            const int RS = R * S;
            CK(cudaMemset(cnt, 0, 4 * E));
            tf_dsn_group_count_kernel<<<1, 256>>>(pick, RS, E, cnt);
            tf_dsn_group_scan_kernel<<<1, 32>>>(cnt, E, off, fill, act, nact);
            tf_dsn_group_place_kernel<<<1, 256>>>(pick, RS, S, E, off, fill, ent, erow);
            CK(cudaDeviceSynchronize());
            const int ny = std::min(E, RS);
            for (int ks : {1, 2, 4}) {
                if (ks_sel && ks != ks_sel) continue;
                for (int early = 0; early < 2; early++) {
                    GuArgs g{};
                    g.q = q; g.s = s; g.ldq = D; g.lds = D / 32; g.ent = ent; g.erow = erow; g.off = off; g.act = act; g.nact = nact;
                    g.w1 = dw1; g.w3 = dw3; g.s1 = ds1; g.s3 = ds3; g.ldw = D / 2; g.ldws = D / 32;
                    g.fmt = 1; g.D = D; g.shared = E - 1; g.sldw = D; g.sldws = D / 32; g.owner = owner;
                    g.n_tiles = (MAIN + REST) / 32; g.rest_tiles = REST / 32; g.wts = wts; g.limit = 10.f; g.h = h; g.ldh = MAIN; g.P = RS;
                    g.splits = ks; g.early_exit = early;
                    CK(cudaMemset(h, 0, size_t(maxR) * S * IH * 2));
                    const int tpc = warps_cta / (2 * ks), nt = (MAIN + REST) / 16;
                    decLaunch("gu", rgOf(R), 1, dim3((nt + tpc - 1) / tpc, ny), &g);
                    CK(cudaGetLastError());
                    CK(cudaStreamSynchronize(st));
                    printf("experts gu R=%-2d ks=%d early=%d %016llx\n", R, ks, early, (unsigned long long)fnv(h, size_t(R) * S * IH * 2));
                }
            }
            for (int ks : {1, 2}) {
                if (ks_sel && ks != ks_sel) continue;
                for (int cb = 0; cb < 2; cb++) {
                    DnArgs dd{};
                    dd.q = hq; dd.s = hs; dd.ldq = IH; dd.lds = IH / 32; dd.ent = ent; dd.off = off; dd.act = act; dd.nact = nact;
                    dd.w2 = dw2; dd.s2 = ds2; dd.ldw = IH / 2; dd.ldws = IH / 32; dd.fmt = 1; dd.shared = E - 1;
                    dd.sldw = IH; dd.sldws = IH / 32; dd.I = IH; dd.N = DN; dd.out = out; dd.ldo = DN; dd.splits = ks; dd.P = RS;
                    dd.round_bf16 = cb;
                    CK(cudaMemset(out, 0, size_t(maxR) * S * DN * 4));
                    const int tpc = warps_cta / ks, nt = DN / 16;
                    decLaunch("dn", rgOf(R), 1, dim3((nt + tpc - 1) / tpc, ny), &dd);
                    CK(cudaGetLastError());
                    CK(cudaStreamSynchronize(st));
                    printf("experts dn R=%-2d ks=%d bf16=%d %016llx\n", R, ks, cb, (unsigned long long)fnv(out, size_t(RS) * DN * 4));
                }
            }
        }
    }
    return 0;
}
