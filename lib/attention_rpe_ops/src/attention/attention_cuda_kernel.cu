#include "../cuda_utils.h"
#include "attention_cuda_kernel.h"


/* ================================== attention ================================== */
__global__ void attention_forward_cuda_(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, const int* rel_idx, const float* attn_weights, const float* table, float* output) {
    // input rel_idx: (tgt_len, src_len, bsz, 3) int
    // input attn_weights: (bsz, num_heads, tgt_len, src_len) float
    // input table: (3, L, num_heads, head_dim) float
    // output output: (bsz, num_heads, tgt_len, head_dim) float

    int d_idx = threadIdx.x;
    int n_idx = blockIdx.x * blockDim.y + threadIdx.y;

    if (n_idx >= bsz * num_heads * tgt_len)
        return ;

    int tgt_idx = n_idx % tgt_len;
    int h_idx = (n_idx / tgt_len) % num_heads;
    int b_idx = n_idx / (num_heads * tgt_len);

    float s = 0.0;
    for(int src_idx = 0; src_idx < src_len; src_idx++){
        int r_idx0 = rel_idx[tgt_idx*src_len*bsz*3 + src_idx*bsz*3 + b_idx*3];
        int r_idx1 = rel_idx[tgt_idx*src_len*bsz*3 + src_idx*bsz*3 + b_idx*3 + 1];
        int r_idx2 = rel_idx[tgt_idx*src_len*bsz*3 + src_idx*bsz*3 + b_idx*3 + 2];
        float attn_weight = attn_weights[b_idx*num_heads*tgt_len*src_len + h_idx*tgt_len*src_len + tgt_idx*src_len + src_idx];
        float table_scalar = table[r_idx0*num_heads*head_dim + h_idx*head_dim + d_idx] + 
                        table[L*num_heads*head_dim + r_idx1*num_heads*head_dim + h_idx*head_dim + d_idx] + 
                        table[2*L*num_heads*head_dim + r_idx2*num_heads*head_dim + h_idx*head_dim + d_idx];
        s += attn_weight * table_scalar;
    }
    output[b_idx*num_heads*tgt_len*head_dim + h_idx*tgt_len*head_dim + tgt_idx*head_dim + d_idx] = s;
}


void attention_forward_cuda(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, const int* rel_idx, const float* attn_weights, const float* table, float* output) {
    // input rel_idx: (tgt_len, src_len, bsz, 3) int
    // input attn_weights: (bsz, num_heads, tgt_len, src_len) float
    // input table: (3, L, num_heads, head_dim) float
    // output output: (bsz, num_heads, tgt_len, head_dim) float

    cudaError_t err;

    dim3 blocks(DIVUP(bsz*num_heads*tgt_len*head_dim, THREADS_PER_BLOCK));
    dim3 threads(head_dim, THREADS_PER_BLOCK/head_dim);

    if(THREADS_PER_BLOCK % head_dim != 0) {
        fprintf(stderr, "THREADS_PER_BLOCK should be a multiple of head_dim %d\n", head_dim);
        exit(-1);
    }

    attention_forward_cuda_<<<blocks, threads, 0>>>(bsz, L, tgt_len, src_len, num_heads, head_dim, rel_idx, attn_weights, table, output);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

/* ================================== attention ================================== */
__global__ void attention_backward_cuda_(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, const int* rel_idx, const float* attn_weights, const float* table, const float* grad_out, float* grad_table, float* grad_attn_weights) {
    // input rel_idx: (tgt_len, src_len, bsz, 3) int
    // input attn_weights: (bsz, num_heads, tgt_len, src_len) float
    // input table: (3, L, num_heads, head_dim) float
    // input grad_out: (bsz, num_heads, tgt_len, head_dim) float
    // output grad_table: (3, L, num_heads, head_dim) float
    // output grad_attn_weights: (bsz, num_heads, tgt_len, src_len) float

    int tgt_idx = blockIdx.x % tgt_len;
    int h_idx = (blockIdx.x / tgt_len) % num_heads;
    int b_idx = blockIdx.x / (tgt_len * num_heads);
    int d_idx = threadIdx.x;

    float table_cache[3][48] = {0.0};
    float grad_out_scalar = grad_out[b_idx*num_heads*tgt_len*head_dim + h_idx*tgt_len*head_dim + tgt_idx*head_dim + d_idx];
    for(int src_idx = 0; src_idx < src_len; src_idx++){
        int r_idx0 = rel_idx[tgt_idx*src_len*bsz*3 + src_idx*bsz*3 + b_idx*3];
        int r_idx1 = rel_idx[tgt_idx*src_len*bsz*3 + src_idx*bsz*3 + b_idx*3 + 1];
        int r_idx2 = rel_idx[tgt_idx*src_len*bsz*3 + src_idx*bsz*3 + b_idx*3 + 2];
        float attn_weights_scalar = attn_weights[b_idx*num_heads*tgt_len*src_len + h_idx*tgt_len*src_len + tgt_idx*src_len + src_idx];
        float grad_table_scalar = grad_out_scalar * attn_weights_scalar;
        table_cache[0][r_idx0] += grad_table_scalar;
        table_cache[1][r_idx1] += grad_table_scalar;
        table_cache[2][r_idx2] += grad_table_scalar;
        float table_scalar = table[r_idx0*num_heads*head_dim + h_idx*head_dim + d_idx] + 
                        table[L*num_heads*head_dim + r_idx1*num_heads*head_dim + h_idx*head_dim + d_idx] + 
                        table[2*L*num_heads*head_dim + r_idx2*num_heads*head_dim + h_idx*head_dim + d_idx];
        atomicAdd(&grad_attn_weights[b_idx*num_heads*tgt_len*src_len + h_idx*tgt_len*src_len + tgt_idx*src_len + src_idx], grad_out_scalar * table_scalar);
    }
    for(int i = 0; i < 3; i++){
        for(int j=0; j < 48; j++){
            atomicAdd(&grad_table[i*L*num_heads*head_dim + j*num_heads*head_dim + h_idx*head_dim + d_idx], table_cache[i][j]);
        }
    }
}


void attention_backward_cuda(int bsz, int L, int tgt_len, int src_len, int num_heads, int head_dim, const int* rel_idx, const float* attn_weights, const float* table, const float* grad_out, float* grad_table, float* grad_attn_weights) {
    // input rel_idx: (tgt_len, src_len, bsz, 3) int
    // input attn_weights: (bsz, num_heads, tgt_len, src_len) float
    // input table: (3, L, num_heads, head_dim) float
    // input grad_out: (bsz, num_heads, tgt_len, head_dim) float
    // output grad_table: (3, L, num_heads, head_dim) float
    // output grad_attn_weights: (bsz, num_heads, tgt_len, src_len) float

    cudaError_t err;

    dim3 blocks(bsz*num_heads*tgt_len);
    dim3 threads(head_dim);

    attention_backward_cuda_<<<blocks, threads, 0>>>(bsz, L, tgt_len, src_len, num_heads, head_dim, rel_idx, attn_weights, table, grad_out, grad_table, grad_attn_weights);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
