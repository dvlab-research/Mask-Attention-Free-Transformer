#include <vector>
#include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention_cuda_kernel.h"

/* ================================== attention ================================== */
// input rel_idx: (tgt_len, src_len, bsz, 3) int
// input attn_weights: (bsz, num_heads, tgt_len, src_len) float
// input table: (3, L, num_heads, head_dim) float
// output output: (bsz, num_heads, tgt_len, head_dim) float
void attention_forward(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, at::Tensor rel_idx_tensor, at::Tensor attn_weights_tensor, at::Tensor table_tensor, at::Tensor output_tensor){
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    const float *attn_weights = attn_weights_tensor.data_ptr<float>();
    const float *table = table_tensor.data_ptr<float>();
    float *output = output_tensor.data_ptr<float>();
    attention_forward_cuda(bsz, L, tgt_len, src_len, num_heads, head_dim, rel_idx, attn_weights, table, output);
}

/* ================================== attention ================================== */
// input xyz: (n, 3) float
// input batch_idxs: (n) int
// input batch_offsets: (B+1) int, batch_offsets[-1]
// output idx: (n * meanActive) dim 0 for number of points in the ball, idx in n
// output start_len: (n, 2), int
void attention_backward(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, at::Tensor rel_idx_tensor, at::Tensor attn_weights_tensor, at::Tensor table_tensor, at::Tensor grad_out_tensor, at::Tensor grad_table_tensor, at::Tensor grad_attn_weights_tensor){
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    const float *attn_weights = attn_weights_tensor.data_ptr<float>();
    const float *table = table_tensor.data_ptr<float>();
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    float *grad_attn_weights = grad_attn_weights_tensor.data_ptr<float>();
    float *grad_table = grad_table_tensor.data_ptr<float>();
    attention_backward_cuda(bsz, L, tgt_len, src_len, num_heads, head_dim, rel_idx, attn_weights, table, grad_out, grad_table, grad_attn_weights);
}
