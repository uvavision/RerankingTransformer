import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict


def num_of_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def state_dict_to_cpu(state_dict: OrderedDict):
    new_state = OrderedDict()
    for k in state_dict.keys():
        newk = k.replace('module.', '')  # remove "module." if model was trained using DataParallel
        new_state[newk] = state_dict[k].cpu()
    return new_state


class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyWithLogits, self).__init__()

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')