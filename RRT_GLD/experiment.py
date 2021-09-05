import os, math
import os.path as osp
from copy import deepcopy
from functools import partial
from pprint import pprint

import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from torch.optim import SGD, Adam, AdamW, lr_scheduler
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import state_dict_to_cpu, num_of_trainable_params
from utils import pickle_load
from utils import BinaryCrossEntropyWithLogits
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train_one_epoch, evaluate

ex = sacred.Experiment('RRT Training', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    epochs = 15
    lr = 0.0001
    momentum = 0.
    nesterov = False
    weight_decay = 5e-4
    optim = 'adamw'
    scheduler = 'multistep'
    max_norm = 0.0
    seed = 0

    visdom_port = None
    visdom_freq = 100
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('outputs', 'temp')

    no_bias_decay = False
    loss = 'bce'
    scheduler_tau = [16, 18]
    scheduler_gamma = 0.1

    resume = None


@ex.capture
def get_optimizer_scheduler(parameters, optim, loader_length, epochs, lr, momentum, nesterov, weight_decay, scheduler, scheduler_tau, scheduler_gamma, lr_step=None):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True if nesterov and momentum else False)
    elif optim == 'adam':
        optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay) 
    else:
        optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    
    if epochs == 0:
        scheduler = None
        update_per_iteration = None
    elif scheduler == 'cos':
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * loader_length, eta_min=0.000005)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000001)
        update_per_iteration = False
    elif scheduler == 'warmcos':
        # warm_cosine = lambda i: min((i + 1) / 3, (1 + math.cos(math.pi * i / (epochs * loader_length))) / 2)
        warm_cosine = lambda i: min((i + 1) / 3, (1 + math.cos(math.pi * i / epochs)) / 2)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_cosine)
        update_per_iteration = False
    elif scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_tau, gamma=scheduler_gamma)
        update_per_iteration = False
    elif scheduler == 'warmstep':
        warm_step = lambda i: min((i + 1) / 100, 1) * 0.1 ** (i // (lr_step * loader_length))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_step)
        update_per_iteration = True
    else:
        scheduler = lr_scheduler.StepLR(optimizer, epochs * loader_length)
        update_per_iteration = True

    return optimizer, (scheduler, update_per_iteration)


@ex.capture
def get_loss(loss):
    if loss == 'bce':
        return BinaryCrossEntropyWithLogits()
    else:
        raise Exception('Unsupported loss {}'.format(loss))


@ex.automain
def main(epochs, cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, no_bias_decay, max_norm, resume):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    # callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed+1)
    model = get_model()
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state'], strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    class_loss = get_loss()
    nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()

    torch.manual_seed(seed+2)
    model.to(device)
    model = nn.DataParallel(model)
    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(parameters=parameters, loader_length=len(loaders.train))
    if resume is not None and checkpoint.get('optim', None) is not None:
        optimizer.load_state_dict(checkpoint['optim'])
        del checkpoint

    torch.manual_seed(seed+3)
    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    result = eval_function()
    # if callback is not None:
    #     callback.scalars(['l2', 'cosine'], 0, [metrics.recall['l2'][1], metrics.recall['cosine'][1]],
    #                      title='Val Recall@1')
    pprint(result)
    best_val = (0, result, deepcopy(model.state_dict()))

    # saving
    save_name = osp.join(temp_dir, '{}_{}.pt'.format(ex.current_run.config['model']['name'],
                                                         ex.current_run.config['dataset']['name']))
    os.makedirs(temp_dir, exist_ok=True)
    torch.manual_seed(seed+4)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        torch.cuda.empty_cache()
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, epoch=epoch, callback=callback, freq=visdom_freq, ex=ex)
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, callback=callback, freq=visdom_freq, ex=None)
        train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, freq=visdom_freq, ex=None)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        torch.cuda.empty_cache()
        result = eval_function()
        print('Validation [{:03d}]'.format(epoch)), pprint(result)
        ex.log_scalar('val.M_map', result['M_map'], step=epoch + 1)
        ex.log_scalar('val.H_map', result['H_map'], step=epoch + 1)

        if (result['M_map'] + result['H_map']) >= (best_val[1]['M_map'] + best_val[1]['H_map']):
            print('New best model in epoch %d.'%epoch)
            best_val = (epoch + 1, result, deepcopy(model.state_dict()))
            torch.save({'state': state_dict_to_cpu(best_val[2]), 'optim': optimizer.state_dict()}, save_name)

    # logging
    ex.info['metrics'] = best_val[1]
    ex.add_artifact(save_name)

    # if callback is not None:
    #     save_name = os.path.join(temp_dir, 'visdom_data.pt')
    #     callback.save(save_name)
    #     ex.add_artifact(save_name)

    return best_val[1]
