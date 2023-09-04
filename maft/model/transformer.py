# ------------------------------------------------------------------------
# Modified from Conditional DETR Transformer (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention
from .attention_rpe import MultiheadAttentionRPE
from .position_embedding import PositionEmbeddingCoordsSine

from timm.models.layers import DropPath, trunc_normal_

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, nhead=8, d_model=256, temperature=10000, pos_type="fourier", attn_mask_thresh=0.5):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.nhead = nhead
        self.norm = norm
        self.return_intermediate = return_intermediate
        
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

        self.position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=False, pos_type=pos_type, d_pos=d_model)
        self.attn_mask_thresh = attn_mask_thresh

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        self._reset_parameters()
        
        self.bbox_embed = MLP(d_model, d_model, 3, 3)

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, input_ranges, coords_float, mask_feats_batched, lengths,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ref_points_unsigmoid: Optional[Tensor] = None):
        output = tgt
        attn_masks = None

        intermediate = []
        reference_points = ref_points_unsigmoid.sigmoid().transpose(0, 1)
        
        ref_points = [reference_points]

        input_ranges_mins, input_ranges_maxs = [], []
        for i in range(len(input_ranges)):
            pos_i_min, pos_i_max = input_ranges[i]
            input_ranges_mins.append(pos_i_min) #[3]
            input_ranges_maxs.append(pos_i_max)
        input_ranges_mins = torch.stack(input_ranges_mins, dim=0).unsqueeze(0) #[1, bsz, 3]
        input_ranges_maxs = torch.stack(input_ranges_maxs, dim=0).unsqueeze(0) #[1, bsz, 3]

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :3].transpose(0, 1) #reference_points[..., :2].transpose(0, 1)  # [num_queries, batch_size, 3]

            reference_points_coords_float = torch.zeros_like(reference_points) #[batch_size, num_queries, 3]
            B = len(input_ranges)
            for b in range(B):
                pos_i_min, pos_i_max = input_ranges[b] #[3]
                reference_points_coords_float[b] = reference_points[b] * (pos_i_max - pos_i_min) + pos_i_min
            reference_points_coords_float = reference_points_coords_float.transpose(0, 1) #[num_queries, batch_size, 3]
    
            query_sine_embed = self.position_embedding(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(output, memory, query_coords_float=reference_points_coords_float, key_coords_float=coords_float, tgt_mask=tgt_mask,
                           memory_mask=attn_masks, #memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            output_norm = self.norm(output) #[num_queries, bsz, d_model]

            # get mask
            pred_masks = torch.einsum('nbd,mbd->bnm', output_norm, mask_feats_batched) #[bsz, num_queries, max_length]
            attn_masks = (pred_masks.sigmoid() < self.attn_mask_thresh).bool() #[bsz, tgt_len, src_len]
            for b in range(lengths.shape[0]):
                length = lengths[b]
                attn_masks[b, (attn_masks[b, :, :length].sum(-1) == length)] = False
                attn_masks[b, :, length:] = True
            attn_masks = attn_masks.unsqueeze(1).expand(-1, self.nhead, -1, -1).contiguous().flatten(0,1)

            obj_center_offset = self.bbox_embed(output_norm) #[num_queries, bsz, 3]
            new_reference_points = obj_center * (input_ranges_maxs - input_ranges_mins) + input_ranges_mins + obj_center_offset #[num_queries, bsz, 3]
            new_reference_points = (new_reference_points - input_ranges_mins) / (input_ranges_maxs - input_ranges_mins) #[num_queries, bsz, 3]
            new_reference_points = new_reference_points.transpose(0,1) #[bsz, num_queries, 3]
            
            if layer_id != len(self.layers) - 1:
                ref_points.append(new_reference_points)
            reference_points = new_reference_points.detach()


            if self.return_intermediate:
                intermediate.append(output_norm)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), torch.stack(ref_points, dim=0)]

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, quant_grid_length, grid_size, rel_query, rel_key, rel_value, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttentionRPE(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.quant_grid_length = quant_grid_length
        self.grid_size = grid_size
        self.rel_query, self.rel_key, self.rel_value = rel_query, rel_key, rel_value

        if rel_query:
            self.relative_pos_query_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_query_table, std=.02)
        if rel_key:
            self.relative_pos_key_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_key_table, std=.02)
        if rel_value:
            self.relative_pos_value_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_value_table, std=.02)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, query_coords_float, key_coords_float,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        
        # contextual relative position encoding
        # query_coords_float: [num_queries, B, 3]
        # key_coords_float: [max_length, B, 3]
        rel_pos = query_coords_float.unsqueeze(1) - key_coords_float.unsqueeze(0) #[num_queries, max_length, B, 3]
        rel_idx = torch.div(rel_pos, self.grid_size, rounding_mode='floor').long()
        rel_idx[rel_idx < -self.quant_grid_length] = -self.quant_grid_length
        rel_idx[rel_idx > self.quant_grid_length - 1] = self.quant_grid_length - 1
        
        rel_idx += self.quant_grid_length
        assert (rel_idx >= 0).all()
        assert (rel_idx <= 2*self.quant_grid_length-1).all()
        
        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   rel_idx=rel_idx, 
                                   relative_pos_query_table=self.relative_pos_query_table if self.rel_query else None, 
                                   relative_pos_key_table=self.relative_pos_key_table if self.rel_key else None, 
                                   relative_pos_value_table=self.relative_pos_value_table if self.rel_value else None)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, query_coords_float, key_coords_float,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        if self.normalize_before:
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, query_coords_float, key_coords_float, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
