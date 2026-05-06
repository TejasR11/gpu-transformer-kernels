#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"
#include <cstdint>

namespace {
constexpr int32_t BLOCK_SIZE = 256;
constexpr int32_t ITEMS_PER_THREAD = 4;
constexpr int32_t ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

int32_t div_ceil(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

__global__ void sum_squares_kernel(const __nv_bfloat16 *hidden_state, float *partial_sums, int32_t len) {
    __shared__ float shared[BLOCK_SIZE];

    float thread_sum = 0.0f;
    int32_t block_start = blockIdx.x * ITEMS_PER_BLOCK;
    for (int32_t offset = threadIdx.x; offset < ITEMS_PER_BLOCK; offset += blockDim.x) {
        int32_t idx = block_start + offset;
        if (idx < len) {
            float val = __bfloat162float(hidden_state[idx]);
            thread_sum += val * val;
        }
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
        partial_sums[blockIdx.x] = shared[0];
    }
}

__global__ void inverse_rms_kernel(const float *partial_sums, float *inverse_rms, int32_t len, int32_t num_sum_blocks) {
    __shared__ float shared[BLOCK_SIZE];

    float thread_sum = 0.0f;
    for (int32_t idx = threadIdx.x; idx < num_sum_blocks; idx += blockDim.x) {
        thread_sum += partial_sums[idx];
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
        *inverse_rms = rsqrtf(shared[0] / static_cast<float>(len) + LayerNorm::EPS);
    }
}

__global__ void apply_layer_norm_kernel(
    const __nv_bfloat16 *hidden_state,
    const __nv_bfloat16 *weights,
    __nv_bfloat16 *output,
    const float *inverse_rms,
    int32_t len
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float normalized = __bfloat162float(hidden_state[idx]) * (*inverse_rms);
        output[idx] = __float2bfloat16(__bfloat162float(weights[idx]) * normalized);
    }
}
}

LayerNorm::LayerNorm(int32_t len) {
    this->len = len;
    num_sum_blocks = div_ceil(len, ITEMS_PER_BLOCK);
    temp_space = std::make_shared<CudaBuffer>((num_sum_blocks + 1) * sizeof(float));
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    auto *temp = static_cast<float*>(temp_space->data);
    float *partial_sums = temp;
    float *inverse_rms = temp + num_sum_blocks;

    sum_squares_kernel<<<num_sum_blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hidden_state->data), partial_sums, len);
    checkCuda(cudaGetLastError());

    inverse_rms_kernel<<<1, BLOCK_SIZE, 0, stream>>>(partial_sums, inverse_rms, len, num_sum_blocks);
    checkCuda(cudaGetLastError());

    int32_t num_apply_blocks = div_ceil(len, BLOCK_SIZE);
    apply_layer_norm_kernel<<<num_apply_blocks, BLOCK_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hidden_state->data),
        static_cast<__nv_bfloat16*>(weights->data),
        static_cast<__nv_bfloat16*>(output->data),
        inverse_rms,
        len);
    checkCuda(cudaGetLastError());
}
