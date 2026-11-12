// tf_ds_sum_pairs_f32_kernel on a real GPU (correctness only): out[r] = in[r] + in[2 + r] bit for bit against the host's
// fp32 add, at the row counts and widths a decode round uses (R rows of 5,120 fp32), with the launch the engine makes
// (ops.zig sumPairsF32: 256 threads, ceil(count / 256) blocks, grid.y = 2), and a count that is not a block multiple:
//   nvcc -std=c++20 -O1 -Izig/kernels/cuda -gencode=arch=compute_120,code=sm_120 --fmad=false --ftz=false -o sum_test tools/ksplit-sum-test/sum_test.cu && ./sum_test
#include "../../zig/kernels/cuda/dsv41_ops.cu"
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    int bad = 0;
    for (size_t rows : {size_t(1), size_t(3), size_t(16), size_t(48)}) {
        size_t count = rows * 5120;
        std::vector<float> h(4 * count);
        unsigned s = 12345;
        for (auto& v : h) {
            s = s * 1664525u + 1013904223u;
            v = (float(int(s >> 8) - (1 << 23)) / float(1 << 23)) * float(1u << ((s >> 3) % 6));
        }
        float *in, *out;
        cudaMalloc(&in, 4 * count * 4);
        cudaMalloc(&out, 2 * count * 4);
        cudaMemcpy(in, h.data(), 4 * count * 4, cudaMemcpyHostToDevice);
        cudaMemset(out, 0xff, 2 * count * 4);
        tf_ds_sum_pairs_f32_kernel<<<dim3((count + 255) / 256, 2), 256>>>(in, out, count);
        cudaError_t e = cudaDeviceSynchronize();
        std::vector<float> g(2 * count);
        cudaMemcpy(g.data(), out, 2 * count * 4, cudaMemcpyDeviceToHost);
        size_t diff = 0;
        for (int r = 0; r < 2; r++)
            for (size_t i = 0; i < count; i++) {
                float want = h[r * count + i] + h[(2 + r) * count + i];
                diff += std::memcmp(&want, &g[r * count + i], 4) != 0;
            }
        printf("%zu rows: %zu of %zu values differ from the host's fp32 add (%s)\n", rows, diff, 2 * count, cudaGetErrorString(e));
        bad += diff != 0 || e != cudaSuccess;
        cudaFree(in);
        cudaFree(out);
    }
    printf(bad ? "FAILED\n" : "all hold\n");
    return bad;
}
