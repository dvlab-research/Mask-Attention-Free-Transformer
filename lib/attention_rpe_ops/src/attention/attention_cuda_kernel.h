#ifndef _ATTENTION_CUDA_KERNEL
#define _ATTENTION_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void attention_forward_cuda(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, const int* rel_idx, const float* attn_weights, const float* table, float* output);
void attention_backward_cuda(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, const int* rel_idx, const float* attn_weights, const float* table, const float* grad_out, float* grad_table, float* grad_attn_weights);

#ifdef __cplusplus
extern "C" {
#endif

void attention_forward(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, at::Tensor rel_idx_tensor, at::Tensor attn_weights_tensor, at::Tensor table_tensor, at::Tensor output_tensor);
void attention_backward(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, at::Tensor rel_idx_tensor, at::Tensor attn_weights_tensor, at::Tensor table_tensor, at::Tensor grad_out_tensor, at::Tensor grad_table_tensor, at::Tensor grad_attn_weights_tensor);

#ifdef __cplusplus
}
#endif
#endif
