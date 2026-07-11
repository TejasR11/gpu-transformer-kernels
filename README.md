# GPU Transformer Kernels — CUDA Inference for a Qwen2 LLM

A from-scratch CUDA implementation of the inference stack for
[Qwen2](https://arxiv.org/pdf/2407.10671), an open-weights LLM in the
[Llama](https://arxiv.org/abs/2407.21783) architecture family. Every GPU kernel
is hand-written — no high-level libraries (cuBLAS, CUTLASS, cub, thrust) are
used. The model runs single-token autoregressive decoding and produces output
matching a PyTorch reference implementation, with an interactive chat mode for
live generation.

## What I implemented

All of the GPU kernels and the model runtime that ties them together live in
[`src/gpu_ops/`](src/gpu_ops/) and [`src/qwen2/`](src/qwen2/):

- **Grouped-query attention** ([`GroupQueryAttention.cuh`](src/gpu_ops/GroupQueryAttention.cuh)) —
  a two-pass, numerically stable [online softmax](https://arxiv.org/pdf/1805.02867)
  (FlashAttention-style chunked reduction over the KV cache). Each chunk computes
  a partial max / denominator / weighted-value sum in shared memory, then a
  finalize kernel combines the partials into the final attention output.
- **RMS / LayerNorm**, **RoPE** (rotary positional embeddings),
  **matrix-vector multiply**, **SwiGLU** (SiLU-gated MLP), and **ArgMax** kernels.
- **`Qwen2Layer` / `Qwen2Model`** — the decode-time runtime. All scratch space is
  allocated once at initialization; nothing is allocated or freed inside the
  forward path.
- **Mixed precision** — `bfloat16` weights and KV cache to reduce memory
  bandwidth, with `float32` accumulation inside kernels to control rounding error.

Kernels are launched on CUDA streams and target coalesced global-memory access
and full-device occupancy.

## Profiling

Every kernel was profiled with NVIDIA Nsight Compute (`ncu`) at Qwen2-0.5B input
sizes to classify each as memory-bandwidth, latency/occupancy, or compute
limited, and to reason about arithmetic intensity. Screenshots and notes are in
[`Profiling/`](Profiling/).

## Build and run

```bash
mkdir -p build && cd build
cmake .. && cmake --build .

# Run tests (per-kernel correctness vs. the reference)
ctest

# Generate 100 tokens, matching the Python reference
./transformer

# Interactive chat
./transformer --interactive --max-seq-len 10000
```

A PyTorch reference implementation used for correctness checks lives in
[`pyref/`](pyref/).

## Attribution

Completed as a project for **Caltech CS 179 (GPU Programming)**. The assignment
specification, build scaffolding, and PyTorch reference were provided by the
course staff (Sam Foxman). The CUDA kernels in `src/gpu_ops/` and the Qwen2
runtime in `src/qwen2/` are my own implementation.
