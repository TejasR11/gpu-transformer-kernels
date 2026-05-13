#pragma once

#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

namespace qwen2_layer_detail {
constexpr int32_t BLOCK_SIZE = 256;

template<Qwen2Size QWEN2_SIZE>
__global__ void add_bf16_in_place_kernel(__nv_bfloat16 *dst, const __nv_bfloat16 *src, int32_t len) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        dst[idx] = __float2bfloat16(__bfloat162float(dst[idx]) + __bfloat162float(src[idx]));
    }
}

template<Qwen2Size QWEN2_SIZE>
inline void add_bf16_in_place(__nv_bfloat16 *dst, const __nv_bfloat16 *src, int32_t len, cudaStream_t stream) {
    int32_t num_blocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_bf16_in_place_kernel<QWEN2_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(dst, src, len);
    checkCuda(cudaGetLastError());
}
}

template<Qwen2Size QWEN2_SIZE>
class Qwen2Layer {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num),
    input_layernorm(Qwen2Config::hidden_size()),
    post_attention_layernorm(Qwen2Config::hidden_size()),
    group_query_attention(max_seq_len) {
        attn_input = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        queries = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(__nv_bfloat16));
        weighted_values = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(float));
        attn_output = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        ffn_input = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
        gate_output = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        up_output = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));
        ffn_output = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));
    }

    uint32_t layer_num;
    LayerNorm input_layernorm;                              // (hidden_size,)
    std::shared_ptr<CudaBuffer> q_proj_weight;              // (queries_size, hidden_size)
    std::shared_ptr<CudaBuffer> q_proj_bias;                // (queries_size,)
    std::shared_ptr<CudaBuffer> k_proj_weight;              // (keys_size, hidden_size)
    std::shared_ptr<CudaBuffer> k_proj_bias;                // (keys_size,)
    std::shared_ptr<CudaBuffer> v_proj_weight;              // (values_size, hidden_size)
    std::shared_ptr<CudaBuffer> v_proj_bias;                // (values_size,)
    std::shared_ptr<CudaBuffer> o_proj_weight;              // (hidden_size, queries_size)
    LayerNorm post_attention_layernorm;                     // (hidden_size,)
    std::shared_ptr<CudaBuffer> up_proj_weight;             // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> gate_proj_weight;           // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> down_proj_weight;           // (hidden_size, intermediate_size)

    /**
     * Pass the hidden state through this layer. Modifies the hidden state in-place.
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param hidden_state current hidden state bf16 (hidden_size,)
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
        if (seq_len <= 0) {
            throw std::runtime_error("Qwen2Layer::forward requires seq_len > 0");
        }

        auto *hidden_state_data = static_cast<__nv_bfloat16*>(hidden_state->data);
        auto *attn_input_data = static_cast<__nv_bfloat16*>(attn_input->data);
        auto *queries_data = static_cast<__nv_bfloat16*>(queries->data);
        auto *weighted_values_data = static_cast<float*>(weighted_values->data);
        auto *attn_output_data = static_cast<__nv_bfloat16*>(attn_output->data);
        auto *ffn_input_data = static_cast<__nv_bfloat16*>(ffn_input->data);
        auto *gate_output_data = static_cast<__nv_bfloat16*>(gate_output->data);
        auto *up_output_data = static_cast<__nv_bfloat16*>(up_output->data);
        auto *ffn_output_data = static_cast<__nv_bfloat16*>(ffn_output->data);

        int64_t current_pos = static_cast<int64_t>(seq_len) - 1;
        auto *k_cache_data = static_cast<__nv_bfloat16*>(k_cache->data);
        auto *v_cache_data = static_cast<__nv_bfloat16*>(v_cache->data);
        __nv_bfloat16 *new_keys = k_cache_data
            + current_pos * (Qwen2Config::num_layers() * Qwen2Config::keys_size())
            + static_cast<int64_t>(layer_num) * Qwen2Config::keys_size();
        __nv_bfloat16 *new_values = v_cache_data
            + current_pos * (Qwen2Config::num_layers() * Qwen2Config::values_size())
            + static_cast<int64_t>(layer_num) * Qwen2Config::values_size();

        input_layernorm.normalize_hidden_state(hidden_state, attn_input, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::queries_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(q_proj_weight->data),
            static_cast<__nv_bfloat16*>(q_proj_bias->data),
            attn_input_data,
            queries_data,
            stream);
        RoPE::apply_rope_to_qk(
            queries_data,
            Qwen2Config::num_query_heads(),
            Qwen2Config::head_size(),
            seq_len - 1,
            Qwen2Config::rope_theta_base(),
            stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::keys_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(k_proj_weight->data),
            static_cast<__nv_bfloat16*>(k_proj_bias->data),
            attn_input_data,
            new_keys,
            stream);
        RoPE::apply_rope_to_qk(
            new_keys,
            Qwen2Config::num_kv_heads(),
            Qwen2Config::head_size(),
            seq_len - 1,
            Qwen2Config::rope_theta_base(),
            stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::values_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(v_proj_weight->data),
            static_cast<__nv_bfloat16*>(v_proj_bias->data),
            attn_input_data,
            new_values,
            stream);

        group_query_attention.sdpa(
            queries_data,
            k_cache_data,
            v_cache_data,
            weighted_values_data,
            layer_num,
            seq_len,
            stream);

        MatrixVectorMultiply::bf16_matmul<float>(
            Qwen2Config::hidden_size(),
            Qwen2Config::queries_size(),
            static_cast<__nv_bfloat16*>(o_proj_weight->data),
            nullptr,
            weighted_values_data,
            attn_output_data,
            stream);
        qwen2_layer_detail::add_bf16_in_place<QWEN2_SIZE>(
            hidden_state_data, attn_output_data, Qwen2Config::hidden_size(), stream);

        post_attention_layernorm.normalize_hidden_state(hidden_state, ffn_input, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(gate_proj_weight->data),
            nullptr,
            ffn_input_data,
            gate_output_data,
            stream);
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(up_proj_weight->data),
            nullptr,
            ffn_input_data,
            up_output_data,
            stream);
        SiLUMult::silu_mult_in_place(gate_output, up_output, stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::hidden_size(),
            Qwen2Config::intermediate_size(),
            static_cast<__nv_bfloat16*>(down_proj_weight->data),
            nullptr,
            gate_output_data,
            ffn_output_data,
            stream);
        qwen2_layer_detail::add_bf16_in_place<QWEN2_SIZE>(
            hidden_state_data, ffn_output_data, Qwen2Config::hidden_size(), stream);
    }

private:
    std::shared_ptr<CudaBuffer> attn_input;         // (hidden_size,)
    std::shared_ptr<CudaBuffer> queries;            // (queries_size,)
    std::shared_ptr<CudaBuffer> weighted_values;    // (queries_size,), float
    std::shared_ptr<CudaBuffer> attn_output;        // (hidden_size,)
    std::shared_ptr<CudaBuffer> ffn_input;          // (hidden_size,)
    std::shared_ptr<CudaBuffer> gate_output;        // (intermediate_size,)
    std::shared_ptr<CudaBuffer> up_output;          // (intermediate_size,)
    std::shared_ptr<CudaBuffer> ffn_output;         // (hidden_size,)
    GroupQueryAttention<QWEN2_SIZE> group_query_attention;
};
