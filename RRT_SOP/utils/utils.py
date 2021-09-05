from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def state_dict_to_cpu(state_dict: OrderedDict):
    """Moves a state_dict to cpu and removes the module. added by DataParallel.
    Parameters
    ----------
    state_dict : OrderedDict
        State_dict containing the tensors to move to cpu.
    Returns
    -------
    new_state_dict : OrderedDict
        State_dict on cpu.
    """
    new_state = OrderedDict()
    for k in state_dict.keys():
        newk = k.replace('module.', '')  # remove "module." if model was trained using DataParallel
        new_state[newk] = state_dict[k].cpu()
    return new_state


def num_of_trainable_params(model):
    """
    Provide number of trainable parameters (i.e. those requiring gradient computation) for input network.
    Args:
        model: PyTorch Network
    Returns:
        int, number of parameters.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Contrastive(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super(Contrastive, self).__init__()
        self.temperature = temperature

    def forward(self, logits, global_features, labels):
        bsize = global_features.size(0)
        logits = torch.mm(global_features, global_features.t()).fill_diagonal_(0, wrap=False)
        logits = torch.exp(logits/self.temperature)

        masks = labels.view(-1, 1).repeat(1, bsize).eq_(labels.clone())
        masks = masks.fill_diagonal_(0, wrap=False)

        p = torch.mul(logits, masks).sum(dim=-1)
        Z = logits.sum(dim=-1)

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))
        loss = -prob_masked.log().sum()/bsize 
        return loss


class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyWithLogits, self).__init__()

    def forward(self, logits, global_features, labels):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
