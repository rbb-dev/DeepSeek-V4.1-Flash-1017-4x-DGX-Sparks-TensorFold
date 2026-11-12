// Build and run on any CUDA GPU (sm_120 or later), correctness only:
//   nvcc -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O1 -Izig/kernels/cuda/dsv41 -gencode=arch=compute_120,code=sm_120 -o st_test tools/roci-st-test/st_test.cu && ./st_test
// stage_store against the fragment stores it replaces, on synthetic accumulators (the value is a function of (m, n)).
#include "native.cu"
#include <cstdio>
#include <vector>

template <class C>
__global__ void tk(MxArgs a, float* ref) {
    const int m0 = blockIdx.x * C::BM, n0 = blockIdx.y * C::BN;
    float acc[1][C::MT][C::NT][4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    for (int i = 0; i < C::MT; i++)
        for (int j = 0; j < C::NT; j++)
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * C::WM + i * 16 + g + h * 8;
                const int n = n0 + wn * C::WN + j * 8 + 2 * t;
                acc[0][i][j][2 * h] = float(m) * 0.5f + float(n) * 1e-3f + 1.0f;
                acc[0][i][j][2 * h + 1] = -float(m) * 0.25f + float(n + 1) * 1e-3f;
                if (m < a.M) {
                    ref[size_t(m) * a.ldy + n] = acc[0][i][j][2 * h];
                    ref[size_t(m) * a.ldy + n + 1] = acc[0][i][j][2 * h + 1];
                }
            }
    stage_store<C>(a, m0, n0, acc);
}

template <class C>
int run(const char* name, int M, int N, bool broken) {
    float *y, *ref;
    size_t n = size_t(M) * N;
    cudaMallocHost(&y, n * 4, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    cudaMalloc(&ref, n * 4);
    cudaMemset(ref, 0xa5, n * 4);
    float* ydev;
    cudaHostGetDevicePointer(&ydev, y, 0);
    cudaMemset(ydev, 0xa5, n * 4);
    MxArgs a{};
    a.M = M; a.N = N; a.y = ydev; a.ldy = N;
    cudaFuncSetAttribute(tk<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    tk<C><<<dim3((M + C::BM - 1) / C::BM, N / C::BN), 256, C::SMEM>>>(a, ref);
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> r(n);
    cudaMemcpy(r.data(), ref, n * 4, cudaMemcpyDeviceToHost);
    if (broken) y[size_t(M - 1) * N + N - 1] += 1.0f;  // the control
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += memcmp(&r[i], &y[i], 4) != 0;
    printf("%s M %d N %d: %zu words differ%s (%s)\n", name, M, N, bad, broken ? " [control: one word changed, must be 1]" : "", cudaGetErrorString(e));
    cudaFreeHost(y);
    cudaFree(ref);
    return broken ? (bad == 1 ? 0 : 1) : (bad == 0 ? 0 : 1);
}

int main() {
    int bad = 0;
    for (int M : {1, 63, 64, 129, 1000, 4096}) {
        bad += run<pf::GemmCfg>("pf GemmCfg", M, 2560, false);
        bad += run<pf::GemmFoldCfg>("pf GemmFoldCfg", M, 2560, false);
        bad += run<pf::GemmSplitACfg>("pf SplitA", M, 12800, false);
        bad += run<pf::GemmSplitC7Cfg>("pf SplitC7", M, 2560, false);
    }
    bad += run<pf::GemmCfg>("pf GemmCfg", 1000, 2560, true);
    printf(bad ? "FAILED\n" : "all hold\n");
    return bad;
}
