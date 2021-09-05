import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from .transformer import TransformerEncoder, TransformerEncoderLayer


class MatchERT(nn.Module):
    def __init__(self, d_global, d_model, seq_len, d_K, nhead, num_encoder_layers, dim_feedforward, dropout, activation, normalize_before):
        super(MatchERT, self).__init__()
        assert (d_model % 2 == 0)
        encoder_layer = TransformerEncoderLayer(d_model, seq_len, d_K, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.pos_encoder = PositionEmbeddingSine(d_model//2, normalize=True, scale=2.0)
        self.remap = nn.Linear(d_global, d_model)
        self.scale_encoder = nn.Embedding(7, d_model)
        self.seg_encoder = nn.Embedding(6, d_model)
        self.classifier = nn.Linear(d_model, 1)
        self._reset_parameters()
        self.d_model = d_model
        self.seq_len = seq_len
        self.d_K = d_K
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
            src_global, src_local, src_mask, src_scales, src_positions,
            tgt_global, tgt_local, tgt_mask, tgt_scales, tgt_positions,
            normalize=True):
        # src: bsize, slen, fsize
        # tgt: bsize, slen, fsize
        src_global = self.remap(src_global)
        tgt_global = self.remap(tgt_global)
        if normalize:
            src_global = F.normalize(src_global, p=2, dim=-1)
            tgt_global = F.normalize(tgt_global, p=2, dim=-1)
            src_local  = F.normalize(src_local,  p=2, dim=-1)
            tgt_local  = F.normalize(tgt_local,  p=2, dim=-1)
        bsize, slen, fsize = src_local.size()

        ##################################################################################################################
        ## The final model does not use position embeddings
        src_local = src_local + self.scale_encoder(src_scales) # + self.pos_encoder(src_positions)
        tgt_local = tgt_local + self.scale_encoder(tgt_scales) # + self.pos_encoder(tgt_positions)
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