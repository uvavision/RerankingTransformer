# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mainly copy-paste from https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
The final model of the experiment does not use any postion embedding. 
The code is included here for future research.
"""
import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, positions):
        # positions: bsize, max_feature_num, 2
        x_embed = positions[:,:,0] * self.scale
        y_embed = positions[:,:,1] * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=positions.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=-1).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=-1).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=-1)#.permute(0, 2, 1)
        
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(64, num_pos_feats)
        self.col_embed = nn.Embedding(64, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, position_inds):
        x_emb = self.col_embed(position_inds[:,:,0])
        y_emb = self.row_embed(position_inds[:,:,1])
        pos = torch.cat([x_emb, y_emb], dim=-1)#.permute(0, 2, 1)
        return pos
