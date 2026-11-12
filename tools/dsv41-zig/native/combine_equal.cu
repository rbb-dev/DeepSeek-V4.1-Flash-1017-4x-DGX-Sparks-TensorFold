// Byte-identity harness for the routed sum (tf_dsn_combine_kernel and dec_dn_cb's dec_dn_sum) of native.cu, not part of the
// engine: seeded random entries of wide magnitude (so the order of the additions shows in the bits), seeded picks with ties,
// invalid ids (negative, >= E: another node's) and S = 7 (the register path) and 1 / 5 / 8 / 16 (the general path), rows
// 1 .. 64, accumulate off / on. One FNV-1a hash per output. Build against two versions of native.cu and diff.
//   nvcc -O3 -std=c++20 -arch=sm_120 --expt-relaxed-constexpr -DNATIVE_CU='"/path/native.cu"' -o combine_equal combine_equal.cu
#include NATIVE_CU
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static uint32_t rng = 4242;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint64_t fnv(const std::vector<uint8_t>& h) { uint64_t x = 1469598103934665603ull; for (uint8_t b : h) { x ^= b; x *= 1099511628211ull; } return x; }
__global__ void sum_test(const float* out, int64_t ldo, CombArgs c, int n_lo, int n_hi) { dec_dn_sum(out, ldo, c, n_lo, n_hi); }
int main() {
    const int E = 385, N = 2560;
    const bool control = getenv("CONTROL") != nullptr;
    for (int S : {7, 1, 5, 8, 16}) {
        for (int R : {1, 2, 3, 5, 8, 33, 64}) {
            std::vector<int> pick(R * S);
            for (int i = 0; i < R * S; i++) {
                const uint32_t v = rnd() % 100;
                pick[i] = v < 8 ? -1 - int(rnd() % 3) : v < 16 ? E + int(rnd() % 3) : v < 40 ? int(rnd() % 6) : int(rnd() % E);  // ties are common
            }
            if (control && S == 7) std::swap(pick[0], pick[1]);
            std::vector<float> out(size_t(R) * S * N), y0(size_t(R) * N);
            for (auto& f : out) { const uint32_t r = rnd(); f = ((r & 1) ? -1.f : 1.f) * ldexpf(1.f + (r >> 8 & 0xffff) / 65536.f, int((r >> 24) % 21) - 10); }
            for (auto& f : y0) f = ldexpf(1.f + (rnd() & 0xffff) / 65536.f, int(rnd() % 21) - 10);
            int *dp; float *dout, *dy; int* dcnt;
            CK(cudaMalloc(&dp, 4 * R * S)); CK(cudaMemcpy(dp, pick.data(), 4 * R * S, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dout, 4 * out.size())); CK(cudaMemcpy(dout, out.data(), 4 * out.size(), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dy, 4 * y0.size())); CK(cudaMalloc(&dcnt, 4));
            for (int acc = 0; acc < 2; acc++) {
                CK(cudaMemcpy(dy, y0.data(), 4 * y0.size(), cudaMemcpyHostToDevice));
                tf_dsn_combine_kernel<<<(R * N + 255) / 256, 256>>>(dout, N, dp, R, S, E, N, dy, N, acc);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                std::vector<uint8_t> h(4 * y0.size()); CK(cudaMemcpy(h.data(), dy, h.size(), cudaMemcpyDeviceToHost));
                printf("combine S=%-2d R=%-2d acc=%d %016llx\n", S, R, acc, (unsigned long long)fnv(h));
                CK(cudaMemcpy(dy, y0.data(), 4 * y0.size(), cudaMemcpyHostToDevice));
                CombArgs c{dp, R, S, E, acc, dy, N, dcnt};
                sum_test<<<1, 256>>>(dout, N, c, 0, N);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(h.data(), dy, h.size(), cudaMemcpyDeviceToHost));
                printf("dn_sum  S=%-2d R=%-2d acc=%d %016llx\n", S, R, acc, (unsigned long long)fnv(h));
            }
            cudaFree(dp); cudaFree(dout); cudaFree(dy); cudaFree(dcnt);
        }
    }
    return 0;
}
