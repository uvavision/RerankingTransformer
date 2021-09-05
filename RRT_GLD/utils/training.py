from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sacred import Experiment
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
# from visdom_logger import VisdomLogger
from utils.metrics import *


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        class_loss: nn.Module,
        optimizer: Optimizer,
        # the second entry indicates if the scheduler should step per iteration or epoch
        scheduler: (_LRScheduler, bool), 
        max_norm: float,
        epoch: int,
        # callback: VisdomLogger,
        freq: int,
        ex: Experiment = None) -> None:
    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)
    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, entry in enumerate(pbar):
        global_feats, local_feats, local_mask, scales, positions, _, _ = entry
        global_feats, local_feats, local_mask, scales, positions = map(to_device, (global_feats, local_feats, local_mask, scales, positions))
        
        p_logits = model(global_feats[0::3], local_feats[0::3], local_mask[0::3], scales[0::3], positions[0::3],
            global_feats[1::3], local_feats[1::3], local_mask[1::3], scales[1::3], positions[1::3])
        n_logits = model(global_feats[0::3], local_feats[0::3], local_mask[0::3], scales[0::3], positions[0::3],
            global_feats[2::3], local_feats[2::3], local_mask[2::3], scales[2::3], positions[2::3])

        logits = torch.cat([p_logits, n_logits], 0)
        bsize = logits.size(0)
        # assert (bsize % 2 == 0)
        labels = logits.new_ones(logits.size()).float()
        labels[(bsize//2):] = 0
        loss = class_loss(logits, labels).mean()
        acc = ((torch.sigmoid(logits) > 0.5).long() == labels.long()).float().mean()

        ##############################################
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if scheduler[-1]:
            scheduler[0].step()

        train_losses.append(loss)
        train_accs.append(acc)


        if not (i + 1) % freq:
            step = epoch + i / loader_length
            print('step/loss/accu/lr:', step, train_losses.last_avg.item(), train_accs.last_avg.item(), scheduler[0].get_last_lr()[0])
            

    if not scheduler[-1]:
        scheduler[0].step()

    if ex is not None:
        for i, (loss, acc) in enumerate(zip(train_losses.values_list, train_accs.values_list)):
            step = epoch + i / loader_length
            ex.log_scalar('train.loss', loss, step=step)
            ex.log_scalar('train.acc', acc, step=step)


def evaluate(
        model: nn.Module,
        cache_nn_inds: torch.Tensor,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        recall: List[int]):
    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)

    query_global, query_local, query_mask, query_scales, query_positions, query_names = [], [], [], [], [], []
    gallery_global, gallery_local, gallery_mask, gallery_scales, gallery_positions, gallery_names = [], [], [], [], [], []

    with torch.no_grad():
        for entry in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            q_global, q_local, q_mask, q_scales, q_positions, _, q_names = entry
            query_global.append(q_global.cpu())
            query_local.append(q_local.cpu())
            query_mask.append(q_mask.cpu())
            query_scales.append(q_scales.cpu())
            query_positions.append(q_positions.cpu())
            query_names.extend(list(q_names))

        query_global    = torch.cat(query_global, 0)
        query_local     = torch.cat(query_local, 0)
        query_mask      = torch.cat(query_mask, 0)
        query_scales    = torch.cat(query_scales, 0)
        query_positions = torch.cat(query_positions, 0)

        for entry in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
            g_global, g_local, g_mask, g_scales, g_positions, _, g_names = entry
            gallery_global.append(g_global.cpu())
            gallery_local.append(g_local.cpu())
            gallery_mask.append(g_mask.cpu())
            gallery_scales.append(g_scales.cpu())
            gallery_positions.append(g_positions.cpu())
            gallery_names.extend(list(g_names))
            
        gallery_global    = torch.cat(gallery_global, 0)
        gallery_local     = torch.cat(gallery_local, 0)
        gallery_mask      = torch.cat(gallery_mask, 0)
        gallery_scales    = torch.cat(gallery_scales, 0)
        gallery_positions = torch.cat(gallery_positions, 0)

        torch.cuda.empty_cache()
        evaluate_function = partial(mean_average_precision_revisited_rerank, model=model, cache_nn_inds=cache_nn_inds,
            query_global=query_global, query_local=query_local, query_mask=query_mask, query_scales=query_scales, query_positions=query_positions, 
            gallery_global=gallery_global, gallery_local=gallery_local, gallery_mask=gallery_mask, gallery_scales=gallery_scales, gallery_positions=gallery_positions, 
            ks=recall, 
            gnd=query_loader.dataset.gnd_data,
        )
        metrics = evaluate_function()
    return metrics 


def evaluate_time(
        model: nn.Module,
        cache_nn_inds: torch.Tensor,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        recall: List[int]):
    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)

    query_global, query_local, query_mask, query_scales, query_positions, query_names = [], [], [], [], [], []
    gallery_global, gallery_local, gallery_mask, gallery_scales, gallery_positions, gallery_names = [], [], [], [], [], []

    with torch.no_grad():
        for entry in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            q_global, q_local, q_mask, q_scales, q_positions, _, q_names = entry
            query_global.append(q_global.cpu())
            query_local.append(q_local.cpu())
            query_mask.append(q_mask.cpu())
            query_scales.append(q_scales.cpu())
            query_positions.append(q_positions.cpu())
            query_names.extend(list(q_names))

        query_global    = torch.cat(query_global, 0)
        query_local     = torch.cat(query_local, 0)
        query_mask      = torch.cat(query_mask, 0)
        query_scales    = torch.cat(query_scales, 0)
        query_positions = torch.cat(query_positions, 0)

        for entry in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
            g_global, g_local, g_mask, g_scales, g_positions, _, g_names = entry
            gallery_global.append(g_global.cpu())
            gallery_local.append(g_local.cpu())
            gallery_mask.append(g_mask.cpu())
            gallery_scales.append(g_scales.cpu())
            gallery_positions.append(g_positions.cpu())
            gallery_names.extend(list(g_names))
            
        gallery_global    = torch.cat(gallery_global, 0)
        gallery_local     = torch.cat(gallery_local, 0)
        gallery_mask      = torch.cat(gallery_mask, 0)
        gallery_scales    = torch.cat(gallery_scales, 0)
        gallery_positions = torch.cat(gallery_positions, 0)

        torch.cuda.empty_cache()
        evaluate_function = partial(mean_average_precision_revisited_rerank_time, model=model, cache_nn_inds=cache_nn_inds,
            query_global=query_global, query_local=query_local, query_mask=query_mask, query_scales=query_scales, query_positions=query_positions, 
            gallery_global=gallery_global, gallery_local=gallery_local, gallery_mask=gallery_mask, gallery_scales=gallery_scales, gallery_positions=gallery_positions, 
            ks=recall, 
            gnd=query_loader.dataset.gnd_data,
        )
        metrics = evaluate_function()
    return metrics 
