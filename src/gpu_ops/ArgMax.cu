#include "ArgMax.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"
#include <cstdint>
#include <limits>

namespace {
constexpr int32_t BLOCK_SIZE = 256;
constexpr int32_t ITEMS_PER_THREAD = 4;
constexpr int32_t ITEMS_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

struct ArgMaxPair {
    float value;
    int32_t index;
};

__device__ bool argmax_better(ArgMaxPair candidate, ArgMaxPair current) {
    return candidate.value > current.value ||
        (candidate.value == current.value && candidate.index < current.index);
}

__global__ void bf16_argmax_blocks_kernel(const __nv_bfloat16 *data, ArgMaxPair *block_results, int32_t len) {
    __shared__ ArgMaxPair shared[BLOCK_SIZE];

    ArgMaxPair best{-INFINITY, std::numeric_limits<int32_t>::max()};
    int32_t block_start = blockIdx.x * ITEMS_PER_BLOCK;

    for (int32_t offset = threadIdx.x; offset < ITEMS_PER_BLOCK; offset += blockDim.x) {
        int32_t idx = block_start + offset;
        if (idx < len) {
            ArgMaxPair candidate{__bfloat162float(data[idx]), idx};
            if (argmax_better(candidate, best)) {
                best = candidate;
            }
        }
    }

    shared[threadIdx.x] = best;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride && argmax_better(shared[threadIdx.x + stride], shared[threadIdx.x])) {
            shared[threadIdx.x] = shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = shared[0];
    }
}

__global__ void argmax_pair_blocks_kernel(const ArgMaxPair *input, ArgMaxPair *block_results, int32_t len) {
    __shared__ ArgMaxPair shared[BLOCK_SIZE];

    ArgMaxPair best{-INFINITY, std::numeric_limits<int32_t>::max()};
    int32_t block_start = blockIdx.x * ITEMS_PER_BLOCK;

    for (int32_t offset = threadIdx.x; offset < ITEMS_PER_BLOCK; offset += blockDim.x) {
        int32_t idx = block_start + offset;
        if (idx < len && argmax_better(input[idx], best)) {
            best = input[idx];
        }
    }

    shared[threadIdx.x] = best;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride && argmax_better(shared[threadIdx.x + stride], shared[threadIdx.x])) {
            shared[threadIdx.x] = shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = shared[0];
    }
}

__global__ void write_argmax_index_kernel(const ArgMaxPair *result, int32_t *output_index) {
    *output_index = result[0].index;
}

int32_t div_ceil(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}
}

ArgMax::ArgMax(int32_t len) {
    this->len = len;
    max_num_blocks = div_ceil(len, ITEMS_PER_BLOCK);
    temp_space = std::make_shared<CudaBuffer>(
        2 * max_num_blocks * sizeof(ArgMaxPair) + sizeof(int32_t));
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    auto *temp_bytes = static_cast<uint8_t*>(temp_space->data);
    auto *buffer_a = reinterpret_cast<ArgMaxPair*>(temp_bytes);
    auto *buffer_b = reinterpret_cast<ArgMaxPair*>(temp_bytes + max_num_blocks * sizeof(ArgMaxPair));
    auto *output_index = reinterpret_cast<int32_t*>(
        temp_bytes + 2 * max_num_blocks * sizeof(ArgMaxPair));

    int32_t current_len = max_num_blocks;
    bf16_argmax_blocks_kernel<<<current_len, BLOCK_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16*>(bf16_data->data), buffer_a, len);
    checkCuda(cudaGetLastError());

    ArgMaxPair *input = buffer_a;
    ArgMaxPair *output = buffer_b;
    while (current_len > 1) {
        int32_t next_len = div_ceil(current_len, ITEMS_PER_BLOCK);
        argmax_pair_blocks_kernel<<<next_len, BLOCK_SIZE, 0, stream>>>(input, output, current_len);
        checkCuda(cudaGetLastError());
        current_len = next_len;

        ArgMaxPair *tmp = input;
        input = output;
        output = tmp;
    }

    write_argmax_index_kernel<<<1, 1, 0, stream>>>(input, output_index);
    checkCuda(cudaGetLastError());
    return output_index;
}
