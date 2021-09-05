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


def train_global(model: nn.Module,
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
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))

        ##################################################
        ## extract features
        logits, features, _ = model(batch)
        loss = class_loss(logits, features, labels).mean()
        acc = (logits.detach().argmax(1) == labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss)
        train_accs.append(acc)
    
    scheduler.step()

    if ex is not None:
        for i, (loss, acc) in enumerate(zip(train_losses.values_list, train_accs.values_list)):
            step = epoch + i / loader_length
            ex.log_scalar('train.loss', loss, step=step)
            ex.log_scalar('train.acc', acc, step=step)


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
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))

        ##################################################
        ## extract features
        l = model(batch)[2]
        anchors   = l[0::3]
        positives = l[1::3]
        negatives = l[2::3]
        p_logits, _, _ = model(None, True, src_global=None, src_local=anchors, tgt_global=None, tgt_local=positives)
        n_logits, _, _ = model(None, True, src_global=None, src_local=anchors, tgt_global=None, tgt_local=negatives)
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
    all_gallery_features, all_gallery_labels = None, None

    with torch.no_grad():
        for batch, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, labels = map(to_device, (batch, labels))
            features = model(batch)[1]
            all_query_labels.append(labels)
            all_query_features.append(features)
        all_query_labels = torch.cat(all_query_labels, 0)
        all_query_features = torch.cat(all_query_features, 0)

        if gallery_loader is not None:
            all_gallery_features, all_gallery_labels = [], []
            for batch, labels, _ in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
                batch, labels = map(to_device, (batch, labels))
                features = model(batch)[1]
                all_gallery_labels.append(labels)
                all_gallery_features.append(features)
            all_gallery_labels = torch.cat(all_gallery_labels, 0)
            all_gallery_features = torch.cat(all_gallery_features, 0)

        recall_function = partial(
            recall_at_ks, query_features=all_query_features, query_labels=all_query_labels, ks=recall_ks,
            gallery_features=all_gallery_features, gallery_labels=all_gallery_labels
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
    all_query_features, all_query_labels = [], []
    all_gallery_features, all_gallery_labels = None, None

    with torch.no_grad():
        for batch, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, labels = map(to_device, (batch, labels))
            features = model(batch)[2]
            all_query_labels.append(labels)
            all_query_features.append(features.cpu())
        all_query_features = torch.cat(all_query_features, 0)
        all_query_labels = torch.cat(all_query_labels, 0)

        if gallery_loader is not None:
            all_gallery_features = []
            all_gallery_labels = []
            for batch, labels, _ in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
                batch, labels = map(to_device, (batch, labels))
                features = model(batch)[-1]
                all_gallery_labels.append(labels.cpu())
                all_gallery_features.append(features.cpu())

            all_gallery_labels = torch.cat(all_gallery_labels, 0)
            all_gallery_features = torch.cat(all_gallery_features, 0)

        recall_function = partial(
                recall_at_ks_rerank, 
                query_features=all_query_features.cpu(), query_labels=all_query_labels.cpu(), ks=recall_ks,
                matcher=model, cache_nn_inds=cache_nn_inds,
                gallery_features=all_gallery_features, gallery_labels=all_gallery_labels
            )
        recalls_rerank, nn_dists, nn_inds = recall_function()
    return recalls_rerank, nn_dists, nn_inds
