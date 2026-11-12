// tf_ds_collect2_kernel on a real GPU, y in write-combined host memory (correctness only):
//   nvcc -std=c++20 -O1 -Izig/kernels/cuda -gencode=arch=compute_120,code=sm_120 -o collect_test tools/roci-collect-test/collect_test.cu && ./collect_test
#include "../../zig/kernels/cuda/dsv41_ops.cu"
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    int bad = 0;
    for (size_t bytes : {size_t(16), size_t(4096), size_t(10485760), size_t(52428800)}) {
        std::vector<unsigned char> h(2 * bytes);
        for (size_t i = 0; i < h.size(); i++) h[i] = (unsigned char)(i * 2654435761u >> 13);
        unsigned char *src, *wc, *wdev;
        cudaMalloc(&src, 2 * bytes);
        cudaMemcpy(src, h.data(), 2 * bytes, cudaMemcpyHostToDevice);
        cudaMallocHost(&wc, 3 * bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        cudaHostGetDevicePointer(&wdev, wc, 0);
        cudaMemset(wdev, 0xa5, 3 * bytes);
        unsigned long long n16 = bytes / 16;
        unsigned bx = n16 / 256 + 1 < 384 ? n16 / 256 + 1 : 384;
        tf_ds_collect2_kernel<<<dim3(bx, 2), 256>>>((const uint4*)src, (uint4*)(wdev + bytes), (const uint4*)(src + bytes), (uint4*)wdev, n16);
        cudaError_t e = cudaDeviceSynchronize();
        size_t d0 = 0, d1 = 0;
        for (size_t i = 0; i < bytes; i++) { d0 += wc[bytes + i] != h[i]; d1 += wc[i] != h[bytes + i]; }
        bool tail_ok = true;
        for (size_t i = 0; i < bytes; i++) tail_ok &= wc[2 * bytes + i] == 0xa5;  // nothing past the blocks written
        printf("%zu bytes a block: %zu + %zu bytes differ, untouched tail %s (%s)\n", bytes, d0, d1, tail_ok ? "intact" : "WRITTEN", cudaGetErrorString(e));
        bad += (d0 || d1 || !tail_ok);
        cudaFree(src); cudaFreeHost(wc);
    }
    printf(bad ? "FAILED\n" : "all hold\n");
    return bad;
}
