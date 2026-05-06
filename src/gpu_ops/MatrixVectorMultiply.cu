#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"
#include <cstdint>

namespace {
constexpr int32_t BLOCK_SIZE = 256;

__device__ float input_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ float input_to_float(float value) {
    return value;
}

template<typename input_float_t>
__global__ void bf16_matvec_kernel(
    int32_t m,
    int32_t k,
    const __nv_bfloat16 *mat,
    const __nv_bfloat16 *bias,
    const input_float_t *vec,
    __nv_bfloat16 *out
) {
    __shared__ float shared[BLOCK_SIZE];

    int32_t row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const __nv_bfloat16 *row_ptr = mat + (static_cast<int64_t>(row) * k);
    float thread_sum = 0.0f;
    for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
        thread_sum += __bfloat162float(row_ptr[col]) * input_to_float(vec[col]);
    }

    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = shared[0];
        if (bias != nullptr) {
            sum += __bfloat162float(bias[row]);
        }
        out[row] = __float2bfloat16(sum);
    }
}
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    bf16_matvec_kernel<<<m, BLOCK_SIZE, 0, stream>>>(m, k, mat, bias, vec, out);
    checkCuda(cudaGetLastError());
}

// explicit instantiations
template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
