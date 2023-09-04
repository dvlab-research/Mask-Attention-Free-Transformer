from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import attention_rpe_ops_cuda
import time

class Attention(Function):
    @staticmethod
    def forward(ctx, rel_idx, relative_pos_value_table, attn_weights):
        """
        input: rel_idx: [tgt_len, src_len, bsz, 3]
        input: relative_pos_value_table: [3, L, num_heads, v_head_dim]
        input: attn_weights: [bsz, num_heads, tgt_len, src_len]
        output: output: [bsz, num_heads, tgt_len, head_dim]
        """
        assert rel_idx.is_contiguous() and relative_pos_value_table.is_contiguous() and attn_weights.is_contiguous()
        
        tgt_len, src_len, bsz, _ = rel_idx.shape
        _, L, num_heads, head_dim = relative_pos_value_table.shape

        output = torch.cuda.FloatTensor(bsz, num_heads, tgt_len, head_dim).zero_()

        attention_rpe_ops_cuda.attention_forward(bsz, L, tgt_len, src_len, num_heads, head_dim, rel_idx, attn_weights, relative_pos_value_table, output)

        ctx.save_for_backward(rel_idx, relative_pos_value_table, attn_weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: rel_idx: [tgt_len, src_len, bsz, 3]
        input: relative_pos_value_table: [3, L, num_heads, v_head_dim]
        input: attn_weights: [bsz, num_heads, tgt_len, src_len]
        input: grad_output: [bsz, num_heads, tgt_len, head_dim]
        """
        
        rel_idx, relative_pos_value_table, attn_weights = ctx.saved_tensors
        
        tgt_len, src_len, bsz, _ = rel_idx.shape
        _, L, num_heads, head_dim = relative_pos_value_table.shape

        grad_output = grad_output.contiguous()
        assert rel_idx.is_contiguous() and relative_pos_value_table.is_contiguous() and attn_weights.is_contiguous() and grad_output.is_contiguous()

        grad_table = torch.cuda.FloatTensor(3, L, num_heads, head_dim).zero_()
        grad_attn_weights = torch.cuda.FloatTensor(bsz, num_heads, tgt_len, src_len).zero_()

        attention_rpe_ops_cuda.attention_backward(bsz, L, tgt_len, src_len, num_heads, head_dim, rel_idx, attn_weights, relative_pos_value_table, grad_output, grad_table, grad_attn_weights)
            
        return None, grad_table, grad_attn_weights

attention = Attention.apply
