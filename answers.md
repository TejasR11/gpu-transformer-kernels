## Part 1

1) The BF16 FLOPS for non-tensor core reaches up to 39 TFLOPS, while the tensore core reaches speeds up to 312 TFLOPS. This means that the tensor core has speeds of up to 8x faster. 

2) Although the tensore cores have 8x more TFLOPS, they are not necessarily faster for matrix-vector multiplications. Let's consider a matrix vector mulitplication:

y = Ax

where y has dimensions M x 1, A has dimensions M x k, and x has dimensions k x 1. Since each element in the matrix does roughly one multiply-add operation, we have 2 FLOPS per element in A[i, j]. However, since each of these elements must be read from memory (and a BF16 is two bytes), we have an arithmetic intensity of 2 FLOPS / 2 bytes = 1 FlOP / byte. Thus, the memory bandwidth roof is 1555 GB/s * 1 FLOP / byte = 1.555 TFLOPS. This is much lower than the 39 and 312 TFLOPS which means that the implementation is limited by memory and not by compute throughput. 

Profiling)

ArgMax:

For ArgMax, the Nsight Compute results suggest that the main `bf16_argmax_blocks_kernel` is not really bottlenecked by either peak DRAM bandwidth or peak compute throughput. The throughput screenshot shows about `24.89%` compute throughput and `13.76%` memory throughput, and it also reports a `Small Grid` warning with only `0.8` full waves across all SMs. That tells me the kernel is more limited by available parallelism and reduction overhead than by the raw hardware limits.

The memory screenshot tells a similar story. The cache hit rates are low (`1.94%` in L1/TEX and `13.17%` in L2), which makes sense because ArgMax mostly streams through the input once and does not reuse values very much. At the same time, the kernel does a noticeable amount of shared-memory traffic because each block has to reduce its local best `(value, index)` pair down to a single winner.

A reasonable optimization would be to reduce the shared-memory reduction overhead, for example by using warp-level shuffle instructions in the later stages of the reduction, or by using wider/vectorized loads to improve effective memory throughput. Out of the ArgMax kernels, the first pass over the full input is the one that matters most to optimize. The later reduction passes and final index write are much smaller and are mostly dominated by launch overhead. If the vocabulary were larger, or if decoding were batched, I would expect the first pass to matter even more and to achieve better occupancy because it would launch more blocks.

LayerNorm:

The three LayerNorm kernels have pretty different behavior. The main `apply_layer_norm_kernel` is the actual elementwise normalization pass, while `sum_squares_kernel` and `inverse_rms_kernel` are both reduction kernels and are much smaller.

For `apply_layer_norm_kernel`, Nsight Compute shows about `1.28%` compute throughput and `2.32%` memory throughput, along with a `Small Grid` warning. That tells me the kernel is not limited by peak compute or DRAM bandwidth here; it is mostly limited by the fact that the launch is too small to fully occupy the GPU. The memory access pattern looks reasonable, with straightforward global loads and stores and decent cache behavior (`23.43%` L1/TEX hit rate and `69.31%` L2 hit rate). In a larger hidden-size or batched setting, I would expect this kernel to behave more like a memory-bound elementwise pass.

The `sum_squares_kernel` is also dominated by low parallelism. Its throughput is only about `1.03%` for both compute and memory, and the screenshots again show a `Small Grid` warning. The memory chart shows that it reads a small amount of global data and then does a shared-memory reduction, which matches the structure of the kernel. A reasonable optimization would be to reduce shared-memory synchronization overhead with warp-level reduction primitives.

The `inverse_rms_kernel` is the smallest and least performance-critical of the three. Its utilization is extremely low (`0.15%` compute throughput and `1.09%` memory throughput), and most of the time is just launch overhead plus a very small reduction. The memory traffic is tiny, so this kernel is latency-limited rather than bandwidth-limited. Overall, the LayerNorm kernel that matters most to optimize is `apply_layer_norm_kernel`; the two reduction kernels are much smaller and would only become more important if the hidden dimension were much larger.

MatrixVectorMultiply:

The matrix-vector multiply kernel is the first one that looks like it is really putting the GPU to work. Nsight Compute reports `71.90%` compute throughput and `71.90%` memory throughput, and even labels it as `Balanced Throughput`. That suggests the kernel is using both compute and memory resources heavily, so reducing runtime would require improving both the arithmetic work and the memory traffic. This makes sense for the implementation here, since each block does a dot product, accumulates in FP32, and also performs a shared-memory reduction at the end.

The memory screenshot shows a large amount of global and shared-memory traffic, with about `19.80 MB` coming from device memory. The L1/TEX hit rate is moderate (`48.98%`), but the L2 hit rate is low (`5.15%`), which suggests that much of the matrix is streamed from memory with limited reuse. A likely optimization would be to improve reuse of the input vector, for example by staging parts of it in shared memory or by restructuring the kernel so that more work can be done per load. Among the Part 1 kernels, this is one of the most important to optimize because matrix-vector multiplies dominate the work inside the transformer layers. If the model size or batch size increased, this kernel would become even more important.

RoPE:

RoPE looks like a very small kernel. The throughput screenshot shows only `0.32%` compute throughput and `1.50%` memory throughput, along with a `Small Grid` warning. That means the kernel is not limited by peak compute or bandwidth at all; it is mostly limited by launch overhead and the fact that there is not enough work to fully occupy the GPU. This is expected, since RoPE only rotates a relatively small number of query/key elements per token.

The memory screenshot supports that interpretation. The total memory traffic is tiny, with only about `11.90 KB` read from device memory, and the cache hit rates are fairly decent (`37.50%` in L1/TEX and `87.54%` in L2). The accesses are simple and coalesced, so there is not an obvious memory-layout problem here. A possible optimization would be to fuse RoPE with neighboring operations such as the Q/K projection output handling, since as a standalone kernel it is too small for the GPU to use efficiently. Compared with other kernels, RoPE is much less important to optimize because it does very little work per token.

SiLUMult:

SiLUMult is another small kernel, although it does a little more work than RoPE. Nsight Compute shows `2.29%` compute throughput and `2.79%` memory throughput, again with a `Small Grid` warning. That means this kernel is also mostly limited by low parallelism and launch overhead rather than by the GPU's peak compute or memory bandwidth. Since it is just an elementwise fused activation and multiply, that is not too surprising.

The memory screenshot shows simple global loads and stores with modest cache reuse (`30.83%` L1/TEX hit rate and `63.89%` L2 hit rate), and only about `54.40 KB` is read from device memory. The access pattern is straightforward, so there is not much evidence of bad memory coalescing. A reasonable optimization would be to fuse SiLU with surrounding feed-forward work so that the intermediate vectors do not need to be written and read as separate kernel steps. Like RoPE, this kernel is less important to optimize than matrix-vector multiply, since it touches far less data and does much less total work in the model.


## Part 2

### Question 2.1
In one Qwen2 0.5B layer, the attention projection matrix-vector multiplies are
the query projection with matrix size (896, 896), the key projection with
matrix size (128, 896), the value projection with matrix size (128, 896), and
the attention output projection with matrix size (896, 896).

The feed-forward network has three matrix-vector multiplies. The gate
projection and up projection both have matrix size (4864, 896), and the down
projection has matrix size (896, 4864). I am not including the grouped-query
attention operations because the question says not to include grouped-query
attention.

### Question 2.2
Qwen2 0.5B has 14 query heads, 2 key/value heads, and head size 64.
Since there are only 2 key/value heads, each key head is shared by 7 query
heads.

So the grouped-query attention multiply can be viewed as two smaller matrix
multiplies. For key/value head 0, the shape is (7, 64) times (64, 1234), which
gives (7, 1234). For key/value head 1, the shape is the same: (7, 64) times
(64, 1234), which gives another (7, 1234). Together, the attention score output
has shape (14, 1234).

### Question 2.3
If off-chip memory bandwidth is the limiting factor, then the minimum latency
is basically the time needed to read the model weights from GPU memory. Qwen2
0.5B has about 0.5 billion parameters, and BF16 uses 2 bytes per parameter, so
the weights are about 1.0 GB.

The A100-PCIE-40GB has about 1555 GB/s of memory bandwidth. So the theoretical
minimum latency is about 1.0 GB / 1555 GB/s, which is 0.00064 seconds, or about
0.64 ms per generated token. This ignores compute time, kernel launch overhead,
and KV cache traffic.

### Question 2.4
For each token, the KV cache stores keys and values for every layer. In Qwen2
0.5B, there are 24 layers, 128 key values per layer, and 128 value values per
layer. So each token needs 24 times (128 + 128) = 6144 BF16 numbers.

Since BF16 uses 2 bytes, the KV cache is 6144 times 2 = 12288 bytes per token.
The model has about 494 million parameters, so the BF16 weights are about
988 MB. Ten percent of that is about 98.8 MB.

So the sequence length is about 98.8 MB / 12288 bytes, which is about 8041
tokens. Roughly, the KV cache becomes 10% of the model size at about 8000
tokens.

### Profiling
Nsight Compute summary for the last-token profile:

![Nsight Compute profile summary](image.png)

The profile captures 100 kernels from the last-token range. My implementation
launches about 20 kernels per Qwen2 layer, so this is about 5 layers total.
The screenshot reports a total duration of about 1232.4 microseconds, so the
implementation takes roughly:

1232.4 microseconds / 5 layers = 246.5 microseconds per layer.

The slowest part of the layer is the feed-forward network. The largest
individual kernels are the matrix-vector multiplies for the gate, up, and down
projections, which take about 50.69, 50.46, and 34.88 microseconds. This makes
sense because these are the largest matrices in the layer.

The attention part is smaller in this profile. Grouped-query attention takes
about 21.86 microseconds for the main partial attention kernel and about 4.70
microseconds for the finalization kernel. Since the sequence length is only 100
tokens, attention does less total work than the feed-forward matrix-vector
multiplies. For much longer sequences, I would expect attention to matter more
because it scales with the KV cache length.

Another thing I noticed is that the large matrix-vector kernels have much
higher compute and memory throughput, around 70% for the gate and up
projections. Smaller kernels like RoPE, LayerNorm, SiLU, and residual adds have
low throughput. These small kernels are mostly limited by launch overhead and
small grid sizes rather than by the peak compute capability of the GPU.

The memory workload chart for one of the smaller kernels also supports this.
It shows very little device memory traffic, only a few KB, and a high L2 cache
hit rate of about 88.6%. That means this kernel is not really limited by global
memory bandwidth. It is more likely limited by the fact that the kernel is
small and does not have enough work to fully use the GPU.
