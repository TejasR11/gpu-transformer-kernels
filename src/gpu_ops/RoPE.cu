#include "RoPE.cuh"
#include <cuda_bf16.h>
#include "../ErrorCheck.h"
#include <cstdint>

namespace {
constexpr int32_t BLOCK_SIZE = 256;

__global__ void apply_rope_kernel(
    __nv_bfloat16 *x,
    int32_t num_heads,
    int32_t head_dim,
    int32_t position_idx,
    float theta_base
) {
    int32_t half_dim = head_dim / 2;
    int32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_pairs = num_heads * half_dim;
    if (pair_idx >= total_pairs) {
        return;
    }

    int32_t head = pair_idx / half_dim;
    int32_t theta_idx = pair_idx % half_dim;
    int32_t base_idx = head * head_dim;
    int32_t left_idx = base_idx + theta_idx;
    int32_t right_idx = base_idx + theta_idx + half_dim;

    float theta_idx_frac = static_cast<float>(theta_idx) / static_cast<float>(half_dim);
    float theta = powf(theta_base, -theta_idx_frac);
    float angle = theta * static_cast<float>(position_idx);
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float left = __bfloat162float(x[left_idx]);
    float right = __bfloat162float(x[right_idx]);

    x[left_idx] = __float2bfloat16((left * cos_val) - (right * sin_val));
    x[right_idx] = __float2bfloat16((right * cos_val) + (left * sin_val));
}
}

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
        int32_t position_idx, float theta_base, cudaStream_t stream) {
    int32_t total_pairs = num_heads * (head_dim / 2);
    int32_t num_blocks = (total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_rope_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        x, num_heads, head_dim, position_idx, theta_base);
    checkCuda(cudaGetLastError());
}
