from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Experiment
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom_logger import VisdomLogger
from copy import deepcopy

from utils.metrics import *


###################################################################

def train_rerank(model: nn.Module,
        loader: DataLoader,
        class_loss: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch: int,
        ex: Experiment = None) -> None:

    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, points, ptmask, _, _) in enumerate(pbar):
        batch, points, ptmask = map(to_device, (batch, points, ptmask))

        ##################################################
        ## extract features
        _, g, l = model(batch, points)

        g_anchors   = g[0::3]
        g_positives = g[1::3]
        g_negatives = g[2::3]

        l_anchors   = l[0::3]
        l_positives = l[1::3]
        l_negatives = l[2::3]

        p_anchors   = points[0::3]
        p_positives = points[1::3]
        p_negatives = points[2::3]

        m_anchors   = ptmask[0::3]
        m_positives = ptmask[1::3]
        m_negatives = ptmask[2::3]

        p_logits, _, _ = model(None, None, True, 
                src_global=g_anchors,   src_local=l_anchors,   src_mask=m_anchors,   src_positions=p_anchors,   
                tgt_global=g_positives, tgt_local=l_positives, tgt_mask=m_positives, tgt_positions=p_positives)
        n_logits, _, _ = model(None, None, True, 
                src_global=g_anchors,   src_local=l_anchors,   src_mask=m_anchors,   src_positions=p_anchors, 
                tgt_global=g_negatives, tgt_local=l_negatives, tgt_mask=m_negatives, tgt_positions=p_negatives)
        logits = torch.cat([p_logits, n_logits], 0)

        bsize = logits.size(0)
        labels = logits.new_ones(logits.size())
        labels[(bsize//2):] = 0
        loss = class_loss(logits, None, labels).mean()
        acc = ((torch.sigmoid(logits) > 0.5).long() == labels.long()).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss)
        train_accs.append(acc)

        if not (i + 1) % 20:
            step = epoch + i / loader_length
            print('step/loss/accu/lr:', step, train_losses.last_avg.item(), train_accs.last_avg.item(), scheduler.get_last_lr()[0])

    scheduler.step()

    if ex is not None:
        for i, (loss, acc) in enumerate(zip(train_losses.values_list, train_accs.values_list)):
            step = epoch + i / loader_length
            ex.log_scalar('train.loss', loss, step=step)
            ex.log_scalar('train.acc', acc, step=step)


###################################################################


def evaluate_global(model: nn.Module,
        query_loader: DataLoader,
        gallery_loader: Optional[DataLoader],
        recall_ks: List[int]):
    
    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    all_query_features, all_query_labels = [], []

    with torch.no_grad():
        for batch, points, _, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, points, labels = map(to_device, (batch, points, labels))
            features = model(batch, points)[1]
            all_query_labels.append(labels)
            all_query_features.append(features)
        all_query_labels = torch.cat(all_query_labels, 0)
        all_query_features = torch.cat(all_query_features, 0)
        recall_function = partial(
            recall_at_ks, query_features=all_query_features, query_labels=all_query_labels, ks=recall_ks,
            gallery_features=None, gallery_labels=None
        )
        recalls_cosine, nn_dists, nn_inds = recall_function(cosine=True)
    
    return recalls_cosine, nn_dists, nn_inds


def evaluate_rerank(model: nn.Module, 
        cache_nn_inds: torch.Tensor,
        query_loader: DataLoader,
        gallery_loader: Optional[DataLoader],
        recall_ks: List[int]):

    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    global_desc, local_desc, positions, masks, label_inds = [], [], [], [], [] 

    with torch.no_grad():
        for batch, points, ptmask, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, points, ptmask, labels = map(to_device, (batch, points, ptmask, labels))
            _, gf, lf = model(batch, points)
            label_inds.append(labels.cpu())
            global_desc.append(gf.cpu())
            local_desc.append(lf.cpu())
            positions.append(points.cpu())
            masks.append(ptmask.cpu())
        label_inds  = torch.cat(label_inds, 0)
        global_desc = torch.cat(global_desc, 0)
        local_desc  = torch.cat(local_desc, 0)
        positions   = torch.cat(positions, 0)
        masks       = torch.cat(masks, 0)
        recall_function = partial(
            recall_at_ks_rerank, 
            global_desc=global_desc, local_desc=local_desc, 
            positions=positions, masks=masks, labels=label_inds,
            ks=recall_ks, matcher=model, cache_nn_inds=cache_nn_inds,
        )
        recalls_rerank, nn_dists, nn_inds = recall_function()
    return recalls_rerank, nn_dists, nn_inds
