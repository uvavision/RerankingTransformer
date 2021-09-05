from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import MatchERT


class BaseModel(nn.Module):
    def __init__(self, num_local_features=256,
            ert_seq_len: int = 102,
            ert_dim_feedforward=1024,
            ert_nhead: int = 4, 
            ert_num_encoder_layers: int = 6, 
            ert_dropout: int = 0.0, 
            ert_activation: str = 'relu', 
            ert_normalize_before: bool = False,
            **kwargs) -> None:
        super(BaseModel, self).__init__()

        self.matcher = MatchERT(d_model=num_local_features, 
            nhead=ert_nhead, num_encoder_layers=ert_num_encoder_layers, 
            dim_feedforward=ert_dim_feedforward, dropout=ert_dropout, 
            activation=ert_activation, normalize_before=ert_normalize_before
        )

    def forward(self, images, points, 
            pairwise_matching=False, 
            src_global=None, src_local=None, src_mask=None, src_positions=None,
            tgt_global=None, tgt_local=None, tgt_mask=None, tgt_positions=None,):

        if pairwise_matching:
            logits = self.matcher(
                src_global=src_global, src_local=src_local, src_mask=src_mask, src_positions=src_positions,
                tgt_global=tgt_global, tgt_local=tgt_local, tgt_mask=tgt_mask, tgt_positions=tgt_positions,
            )
            return logits, None, None

        g, l = self.feature_extractor(images, points)
        return None, g, l