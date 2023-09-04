import torch
import attention_rpe_ops

def vanilla(relative_pos_value_table_, rel_idx_transpose_, attn_output_weights_):

    rel_relative_pos_emb = relative_pos_value_table_[0, :, rel_idx_transpose_[0, :, :, :]] + \
                    relative_pos_value_table_[1, :, rel_idx_transpose_[1, :, :, :]] + \
                    relative_pos_value_table_[2, :, rel_idx_transpose_[2, :, :, :]] #[num_heads, bsz, tgt_len, src_len, v_head_dim]
    rel_relative_pos_emb = rel_relative_pos_emb.contiguous()
    attn_output_weights_transpose = attn_output_weights_.view(bsz, num_heads, tgt_len, src_len)

    return torch.bmm(attn_output_weights_transpose.contiguous().permute(1,0,2,3).contiguous().view(num_heads*bsz*tgt_len, 1, src_len), rel_relative_pos_emb.view(num_heads*bsz*tgt_len, src_len, v_head_dim)).view(num_heads, bsz, tgt_len, v_head_dim).permute(1,0,2,3).contiguous()

bsz = 4
num_heads = 8
v_head_dim = 32
L = 48
tgt_len = 1000
src_len = 400
relative_pos_value_table = torch.rand(3, num_heads, L, v_head_dim).cuda()
rel_idx =  (torch.rand(3, bsz, tgt_len, src_len) * L).long().cuda()
attn_weights = torch.rand(bsz, num_heads, tgt_len, src_len).cuda()

relative_pos_value_table.requires_grad = True
attn_weights.requires_grad = True

output = vanilla(relative_pos_value_table, rel_idx, attn_weights)

loss = output.sum()
loss.backward()

grad_table = relative_pos_value_table.grad.clone()
grad_attn_weights = attn_weights.grad.clone()

relative_pos_value_table.grad.zero_()
attn_weights.grad.zero_()

relative_pos_value_table_ = relative_pos_value_table.permute(0,2,1,3).contiguous().detach().clone() #[3, L, num_heads, head_dim]
rel_idx_ = rel_idx.permute(2,3,1,0).contiguous() #[tgt_len, src_len, bsz, 3]

relative_pos_value_table_.requires_grad = True
attn_weights.requires_grad = True

output_v2 = attention_rpe_ops.attention(rel_idx_.int(), relative_pos_value_table_, attn_weights)

loss2 = output_v2.sum()
loss2.backward()

grad_table_v2 = relative_pos_value_table_.grad.permute(0,2,1,3).clone()
grad_attn_weights_v2 = attn_weights.grad.clone()


print("output.shape: {}, output_v2.shape: {}".format(output.shape, output_v2.shape))

print("(output_v2-output).abs().max(): ", (output_v2-output).abs().max())

print("output_v2.abs().max(): {}, output.abs().max(): {}".format(output_v2.abs().max(), output.abs().max()))

print("output_v2[0,0,:5,:5]: ", output_v2[0,0,:5,:5])
print("output[0,0,:5,:5]: ", output[0,0,:5,:5])


print("(grad_table_v2-grad_table).abs().max(): ", (grad_table_v2-grad_table).abs().max())

print("grad_table_v2[0,0,:5,:5]: ", grad_table_v2[0,0,:5,:5])
print("grad_table[0,0,:5,:5]: ", grad_table[0,0,:5,:5])

print("(grad_table_v2==0).all(): ", (grad_table_v2==0).all())

print("(grad_attn_weights_v2-grad_attn_weights).abs().max(): ", (grad_attn_weights_v2-grad_attn_weights).abs().max())

print("grad_attn_weights_v2[0,0,:5,:5]: ", grad_attn_weights_v2[0,0,:5,:5])
print("grad_attn_weights[0,0,:5,:5]: ", grad_attn_weights[0,0,:5,:5])
