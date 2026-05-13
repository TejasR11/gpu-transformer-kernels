#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "../ErrorCheck.h"

namespace group_query_attention_detail {
constexpr int32_t BLOCK_SIZE = 128;
constexpr int32_t TOKENS_PER_CHUNK = BLOCK_SIZE;

inline int32_t div_ceil(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

__device__ inline void combine_online_normalizer(
    float &base_max,
    float &base_denominator,
    float other_max,
    float other_denominator
) {
    if (other_denominator == 0.0f) {
        return;
    }

    if (base_denominator == 0.0f) {
        base_max = other_max;
        base_denominator = other_denominator;
        return;
    }

    float new_max = fmaxf(base_max, other_max);
    base_denominator = base_denominator * expf(base_max - new_max) +
        other_denominator * expf(other_max - new_max);
    base_max = new_max;
}

template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_partial_kernel(
    const __nv_bfloat16 *queries,
    const __nv_bfloat16 *k_cache,
    const __nv_bfloat16 *v_cache,
    float *partial_maxes,
    float *partial_denominators,
    float *partial_weighted_sums,
    int32_t layer_num,
    int32_t seq_len,
    int32_t num_chunks
) {
    using Config = Qwen2Config<QWEN2_SIZE>;

    __shared__ float shared_query[Config::head_size()];
    __shared__ float shared_scores_or_weights[TOKENS_PER_CHUNK];
    __shared__ float shared_maxes[BLOCK_SIZE];
    __shared__ float shared_denominators[BLOCK_SIZE];

    int32_t query_head_idx = blockIdx.x;
    int32_t chunk_idx = blockIdx.y;
    int32_t kv_head_idx = query_head_idx * Config::num_kv_heads() / Config::num_query_heads();

    if (threadIdx.x < Config::head_size()) {
        shared_query[threadIdx.x] = __bfloat162float(
            queries[query_head_idx * Config::head_size() + threadIdx.x]);
    }
    __syncthreads();

    int32_t chunk_start = chunk_idx * TOKENS_PER_CHUNK;
    int32_t sequence_pos = chunk_idx * TOKENS_PER_CHUNK + threadIdx.x;
    float score = -INFINITY;
    if (sequence_pos < seq_len) {
        const __nv_bfloat16 *key = k_cache
            + sequence_pos * (Config::num_layers() * Config::keys_size())
            + layer_num * Config::keys_size()
            + kv_head_idx * Config::head_size();

        float dot_product = 0.0f;
        for (int32_t el_idx = 0; el_idx < Config::head_size(); el_idx++) {
            dot_product += shared_query[el_idx] * __bfloat162float(key[el_idx]);
        }

        score = dot_product * rsqrtf(static_cast<float>(Config::head_size()));
    }

    shared_scores_or_weights[threadIdx.x] = score;
    shared_maxes[threadIdx.x] = score;
    shared_denominators[threadIdx.x] = sequence_pos < seq_len ? 1.0f : 0.0f;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            combine_online_normalizer(
                shared_maxes[threadIdx.x],
                shared_denominators[threadIdx.x],
                shared_maxes[threadIdx.x + stride],
                shared_denominators[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float chunk_max = shared_maxes[0];
    float weight = sequence_pos < seq_len ? expf(score - chunk_max) : 0.0f;
    shared_scores_or_weights[threadIdx.x] = weight;
    __syncthreads();

    int32_t partial_idx = query_head_idx * num_chunks + chunk_idx;
    if (threadIdx.x == 0) {
        partial_maxes[partial_idx] = chunk_max;
        partial_denominators[partial_idx] = shared_denominators[0];
    }

    if (threadIdx.x < Config::value_size()) {
        int32_t value_el_idx = threadIdx.x;
        float weighted_sum = 0.0f;
        for (int32_t token_offset = 0; token_offset < TOKENS_PER_CHUNK; token_offset++) {
            int32_t value_sequence_pos = chunk_start + token_offset;
            if (value_sequence_pos < seq_len) {
                const __nv_bfloat16 *value = v_cache
                    + value_sequence_pos * (Config::num_layers() * Config::values_size())
                    + layer_num * Config::values_size()
                    + kv_head_idx * Config::value_size();
                weighted_sum += shared_scores_or_weights[token_offset] * __bfloat162float(value[value_el_idx]);
            }
        }
        partial_weighted_sums[partial_idx * Config::value_size() + value_el_idx] = weighted_sum;
    }
}

template<Qwen2Size QWEN2_SIZE>
__global__ void gqa_finalize_kernel(
    const float *partial_maxes,
    const float *partial_denominators,
    const float *partial_weighted_sums,
    float *weighted_values,
    int32_t num_chunks
) {
    using Config = Qwen2Config<QWEN2_SIZE>;

    __shared__ float shared_maxes[BLOCK_SIZE];
    __shared__ float shared_denominators[BLOCK_SIZE];
    __shared__ float shared_global_max;
    __shared__ float shared_global_denominator;

    int32_t query_head_idx = blockIdx.x;

    float local_max = -INFINITY;
    float local_denominator = 0.0f;
    for (int32_t chunk_idx = threadIdx.x; chunk_idx < num_chunks; chunk_idx += blockDim.x) {
        int32_t partial_idx = query_head_idx * num_chunks + chunk_idx;
        combine_online_normalizer(
            local_max,
            local_denominator,
            partial_maxes[partial_idx],
            partial_denominators[partial_idx]);
    }

    shared_maxes[threadIdx.x] = local_max;
    shared_denominators[threadIdx.x] = local_denominator;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            combine_online_normalizer(
                shared_maxes[threadIdx.x],
                shared_denominators[threadIdx.x],
                shared_maxes[threadIdx.x + stride],
                shared_denominators[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        shared_global_max = shared_maxes[0];
        shared_global_denominator = shared_denominators[0];
    }
    __syncthreads();

    if (threadIdx.x < Config::value_size()) {
        int32_t value_el_idx = threadIdx.x;
        float weighted_sum = 0.0f;
        for (int32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            int32_t partial_idx = query_head_idx * num_chunks + chunk_idx;
            weighted_sum += partial_weighted_sums[partial_idx * Config::value_size() + value_el_idx] *
                expf(partial_maxes[partial_idx] - shared_global_max);
        }
        weighted_values[query_head_idx * Config::value_size() + value_el_idx] =
            weighted_sum / shared_global_denominator;
    }
}
}

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
    std::shared_ptr<CudaBuffer> temp_space;
    int32_t max_seq_len{};
    int32_t max_num_chunks{};

public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len) {
        this->max_seq_len = max_seq_len;
        max_num_chunks = group_query_attention_detail::div_ceil(
            max_seq_len, group_query_attention_detail::TOKENS_PER_CHUNK);

        size_t partial_state_count = Qwen2Config::num_query_heads() * max_num_chunks;
        temp_space = std::make_shared<CudaBuffer>(
            partial_state_count * (2 + Qwen2Config::value_size()) * sizeof(float));
    }

    /**
     * Scaled dot product attention with grouped queries, see https://arxiv.org/abs/2305.13245.
     * Performs softmax((QK^T)/sqrt(d_k))*V for all queries Q and their associated K and V
     * - dot product each query with its target value throughout the sequence
     * - numerically stable softmax
     * - save a weighted sum of values
     * Does not perform the output projection.
     *
     * All inputs and outputs are row-major
     *
     * @param queries (num_query_heads, head_size)
     * @param k_cache (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache (seq_len, num_layers, num_kv_heads, value_size)
     * @param weighted_values (num_query_heads, value_size) outputs
     * @param layer_num layer index, starting at 0
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
        if (seq_len <= 0 || seq_len > max_seq_len) {
            throw std::runtime_error("GroupQueryAttention seq_len is outside the configured max_seq_len");
        }

        int32_t num_chunks = group_query_attention_detail::div_ceil(
            seq_len, group_query_attention_detail::TOKENS_PER_CHUNK);
        size_t partial_state_count = Qwen2Config::num_query_heads() * num_chunks;
        auto *temp = static_cast<float*>(temp_space->data);
        float *partial_maxes = temp;
        float *partial_denominators = partial_maxes + partial_state_count;
        float *partial_weighted_sums = partial_denominators + partial_state_count;

        dim3 partial_grid(Qwen2Config::num_query_heads(), num_chunks);
        group_query_attention_detail::gqa_partial_kernel<QWEN2_SIZE>
            <<<partial_grid, group_query_attention_detail::BLOCK_SIZE, 0, stream>>>(
                queries,
                k_cache,
                v_cache,
                partial_maxes,
                partial_denominators,
                partial_weighted_sums,
                layer_num,
                seq_len,
                num_chunks);
        checkCuda(cudaGetLastError());

        group_query_attention_detail::gqa_finalize_kernel<QWEN2_SIZE>
            <<<Qwen2Config::num_query_heads(), group_query_attention_detail::BLOCK_SIZE, 0, stream>>>(
                partial_maxes,
                partial_denominators,
                partial_weighted_sums,
                weighted_values,
                num_chunks);
        checkCuda(cudaGetLastError());
    }
};
