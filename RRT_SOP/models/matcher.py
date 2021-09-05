import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from .transformer import TransformerEncoder, TransformerEncoderLayer


class MatchERT(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, normalize_before):
        super(MatchERT, self).__init__()
        assert (d_model % 2 == 0)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pos_encoder = PositionEmbeddingSine(d_model//2, normalize=True, scale=2.0)
        self.seg_encoder = nn.Embedding(6, d_model)
        self.classifier = nn.Linear(d_model, 1)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
            src_global, src_local, src_mask, src_positions,
            tgt_global, tgt_local, tgt_mask, tgt_positions,
            normalize=False):

        bsize = src_local.size(0)

        ##################################################################################################################
        # src_local = src_local + self.pos_encoder(src_positions)
        # tgt_local = tgt_local + self.pos_encoder(tgt_positions)
        ##################################################################################################################

        ##################################################################################################################
        src_local  = src_local + self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long))
        tgt_local  = tgt_local + self.seg_encoder(src_local.new_ones((bsize, 1), dtype=torch.long))
        cls_embed  = self.seg_encoder(2 * src_local.new_ones((bsize, 1), dtype=torch.long))
        sep_embed  = self.seg_encoder(3 * src_local.new_ones((bsize, 1), dtype=torch.long))
        src_global = src_global.unsqueeze(1) + self.seg_encoder(4 * src_local.new_ones((bsize, 1), dtype=torch.long))
        tgt_global = tgt_global.unsqueeze(1) + self.seg_encoder(5 * src_local.new_ones((bsize, 1), dtype=torch.long))
        ##################################################################################################################
        
        input_feats = torch.cat([cls_embed, src_global, src_local, sep_embed, tgt_global, tgt_local], 1).permute(1,0,2)
        input_mask = torch.cat([
            src_local.new_zeros((bsize, 2), dtype=torch.bool),
            src_mask,
            src_local.new_zeros((bsize, 2), dtype=torch.bool),
            tgt_mask
        ], 1)
        logits = self.encoder(input_feats, src_key_padding_mask=input_mask)
        logits = logits[0]
        return self.classifier(logits).view(-1)