1) The BF16 FLOPS for non-tensor core reaches up to 39 TFLOPS, while the tensore core reaches speeds up to 312 TFLOPS. This means that the tensor core has speeds of up to 8x faster. 

2) Although the tensore cores have 8x more TFLOPS, they are not necessarily faster for matrix-vector multiplications. Let's consider a matrix vector mulitplication:

y = Ax

where y has dimensions M x 1, A has dimensions M x k, and x has dimensions k x 1. Since each element in the matrix does roughly one multiply-add operation, we have 2 FLOPS per element in A[i, j]. However, since each of these elements must be read from memory (and a BF16 is two bytes), we have an arithmetic intensity of 2 FLOPS / 2 bytes = 1 FlOP / byte. Thus, the memory bandwidth roof is 1555 GB/s * 1 FLOP / byte = 1.555 TFLOPS. This is much lower than the 39 and 312 TFLOPS which means that the implementation is limited by memory and not by compute throughput. 

