// Development oracle only: validates offline packing against hardware converts.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "quantize/nvfp4_sf_reshape_sm120.cuh"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

// Host-only template oracle: no CUTLASS GEMM or model kernel is launched.
extern "C" int64_t sm100_layout_oracle(int64_t* offsets, int rows, int blocks, int operand) {
    if (rows <= 0 || blocks <= 0 || (operand != 0 && operand != 1)) return -1;
    using Config = cutlass::detail::Sm1xxBlockScaledConfig<16>;
    auto shape = cute::make_shape(rows, rows, blocks * 16, 1);
    auto a = Config::tile_atom_to_shape_SFA(shape);
    auto b = Config::tile_atom_to_shape_SFB(shape);
    for (int row = 0; row < rows; ++row)
        for (int block = 0; block < blocks; ++block)
            offsets[int64_t(row) * blocks + block] = operand == 0
                ? a(row, block * 16, 0) : b(row, block * 16, 0);
    return operand == 0 ? cute::cosize(a) : cute::cosize(b);
}

extern "C" int scale_layout_oracle(const uint8_t* input, uint8_t* output, int rows, int blocks) {
    if (rows <= 0 || blocks <= 0) return cudaErrorInvalidValue;
    size_t bytes = flash_rt::fp4::nvfp4_sf_swizzled_bytes(rows, blocks * 16);
    uint8_t *src = nullptr, *dst = nullptr;
    cudaError_t err = cudaMalloc(&src, size_t(rows) * blocks);
    if (err == cudaSuccess) err = cudaMalloc(&dst, bytes);
    if (err == cudaSuccess) err = cudaMemset(dst, 0, bytes);
    if (err == cudaSuccess) err = cudaMemcpy(src, input, size_t(rows) * blocks, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        flash_rt::fp4::nvfp4_sf_linear_to_swizzled(src, dst, rows, blocks * 16, false, nullptr);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaMemcpy(output, dst, bytes, cudaMemcpyDeviceToHost);
    cudaFree(src); cudaFree(dst);
    return err;
}

__global__ void convert(const float* x, uint8_t* fp4, uint8_t* fp8, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint16_t pair;
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %1, %1; cvt.u16.u8 %0, b; }"
        : "=h"(pair) : "f"(x[i]));
    fp4[i] = pair & 15;
    fp8[i] = __nv_fp8_e4m3(x[i]).__x;
}

extern "C" int quantization_oracle(const float* input, uint8_t* fp4, uint8_t* fp8, int count) {
    if (count <= 0) return cudaErrorInvalidValue;
    float* x = nullptr;
    uint8_t *a = nullptr, *b = nullptr;
    cudaError_t err = cudaMalloc(&x, count * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc(&a, count);
    if (err == cudaSuccess) err = cudaMalloc(&b, count);
    if (err == cudaSuccess) err = cudaMemcpy(x, input, count * sizeof(float), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        convert<<<(count + 255) / 256, 256>>>(x, a, b, count);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaMemcpy(fp4, a, count, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) err = cudaMemcpy(fp8, b, count, cudaMemcpyDeviceToHost);
    cudaFree(x); cudaFree(a); cudaFree(b);
    return err;
}
