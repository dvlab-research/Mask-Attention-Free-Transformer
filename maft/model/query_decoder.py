import torch
import torch.nn as nn
from .transformer import TransformerDecoder, TransformerDecoderLayer
from .position_embedding import PositionEmbeddingCoordsSine

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_offsets, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        """
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)  # (1, n, d_model)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask=attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
        return outputs


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        x Tensor (b, 100, c)
        """
        q = k = self.with_pos_embed(x, pe)
        output, _ = self.attn(q, k, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output


class QueryDecoder(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        pe=False,
        temperature=10000,
        pos_type="fourier",
        attn_mask_thresh=0.5,
        quant_grid_length=24,
        grid_size=0.05,
        rel_query=True, 
        rel_key=True, 
        rel_value=True
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.d_model = d_model
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.refpoint_embed = nn.Embedding(num_query, 3)
        
        self.key_position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=True, pos_type=pos_type, d_pos=d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, quant_grid_length, grid_size, rel_query, rel_key, rel_value, hidden_dim,
                                        dropout, activation_fn, normalize_before=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_layer, decoder_norm,
                                          return_intermediate=True,
                                          nhead=nhead,
                                          d_model=d_model,
                                          attn_mask_thresh=attn_mask_thresh)

        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.out_bbox = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

        nn.init.constant_(self.out_bbox[-1].weight.data, 0)
        nn.init.constant_(self.out_bbox[-1].bias.data, 0)
        
    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feat)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_offsets, input_ranges, ref_points):
        pred_labels = self.out_cls(query)
        pred_scores = self.out_score(query)
        pred_bboxes = self.out_bbox(query)
        for i, input_range in enumerate(input_ranges):
            min_xyz_i, max_xyz_i = input_range
            pred_bboxes[i] = ref_points[i] * (max_xyz_i - min_xyz_i) + min_xyz_i + pred_bboxes[i]
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)
        return pred_labels, pred_scores, pred_bboxes, pred_masks, attn_masks

    def forward_iter_pred(self, x, pos, batch_offsets):
        """
        x [B*M, inchannel]
        """

        B = len(batch_offsets) - 1
        d_model = self.d_model

        prediction_labels = []
        prediction_masks = []
        prediction_scores = []
        prediction_bboxes = []
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)

        query = self.refpoint_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (b, n, 3)
        num_queries = query.shape[1]
        
        query = query.permute(1,0,2).contiguous() #[num_queries, b, 3]
        lengths = batch_offsets[1:] - batch_offsets[:-1]
        max_length = lengths.max().item()
        inst_feats_batched = inst_feats.new_zeros(max_length, B, d_model)
        pos_batched = pos.new_zeros(max_length, B, d_model)
        coords_float_batched = pos.new_zeros(max_length, B, 3)
        key_padding_masks_batched = inst_feats.new_ones(B, max_length).bool()
        mask_feats_batched = mask_feats.new_zeros(max_length, B, d_model)
        input_ranges = []
        for i in range(B):
            start, end = batch_offsets[i], batch_offsets[i+1]
            inst_feats_batched[:lengths[i], i, :] = inst_feats[start:end]
            
            pos_i = pos[start:end]
            coords_float_batched[:lengths[i], i, :] = pos_i
            
            pos_i_min, pos_i_max = pos_i.min(0)[0], pos_i.max(0)[0]
            pos_emb_i = self.key_position_embedding(pos_i.unsqueeze(0), num_channels=d_model, input_range=(pos_i_min.unsqueeze(0), pos_i_max.unsqueeze(0)))[0] 
            pos_batched[:lengths[i], i, :] = pos_emb_i
            input_ranges.append((pos_i_min, pos_i_max))
            
            mask_feats_batched[:lengths[i], i, :] = mask_feats[start:end]

            key_padding_masks_batched[i, :lengths[i]] = False
            
        intermediate_results, ref_points = self.decoder(tgt=query.new_zeros(num_queries, B, d_model),
            memory=inst_feats_batched,
            input_ranges=input_ranges,
            coords_float=coords_float_batched,
            mask_feats_batched=mask_feats_batched,
            lengths=lengths,
            memory_key_padding_mask=key_padding_masks_batched,
            pos=pos_batched,
            ref_points_unsigmoid=query)

        for i in range(intermediate_results.shape[0]):
            ouptut_i = intermediate_results[i]
            pred_labels, pred_scores, pred_bboxes, pred_masks, attn_masks = self.prediction_head(ouptut_i, mask_feats, batch_offsets, input_ranges, ref_points[i])
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_bboxes.append(pred_bboxes)
            prediction_masks.append(pred_masks)

        return {
            'labels':
            pred_labels,
            'masks':
            pred_masks,
            'scores':
            pred_scores,
            'bboxes':
            pred_bboxes,
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'scores': c,
                'bboxes': d,
            } for a, b, c, d in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_scores[:-1],
                prediction_bboxes[:-1],
            )],
        }

    def forward(self, x, pos, batch_offsets):
        return self.forward_iter_pred(x, pos, batch_offsets)