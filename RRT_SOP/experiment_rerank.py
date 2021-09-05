import os
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
from torch.optim import AdamW, lr_scheduler

from models.ingredient import model_ingredient, get_model
from utils import pickle_load, pickle_save
from utils import state_dict_to_cpu, BinaryCrossEntropyWithLogits, num_of_trainable_params
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train_rerank, evaluate_rerank

ex = sacred.Experiment('Rerank (train)', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    epochs = 100
    lr = 0.0001
    weight_decay = 4e-4
    scheduler_tau = [60, 80]
    scheduler_gamma = 0.1
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('logs', 'temp')
    no_bias_decay = False
    resume = None
    cache_nn_inds = 'rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl'
    seed = 459858808


@ex.capture
def get_optimizer_scheduler(parameters, lr, weight_decay, scheduler_tau, scheduler_gamma):
    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_tau, gamma=scheduler_gamma)
    return optimizer, scheduler


@ex.capture
def get_loss():
    return BinaryCrossEntropyWithLogits()


@ex.automain
def main(epochs, cpu, cudnn_flag, temp_dir, seed, no_bias_decay, resume, cache_nn_inds):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model = get_model(num_classes=loaders.num_classes)
    if resume is not None:
        state_dict = torch.load(resume, map_location=torch.device('cpu'))
        if 'state' in state_dict:
            state_dict = state_dict['state']
        model.load_state_dict(state_dict, strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    class_loss = get_loss()

    # Rerank the top-15 only during training to save time
    cache_nn_inds = pickle_load(cache_nn_inds)[:, :15]
    cache_nn_inds = torch.from_numpy(cache_nn_inds)

    model.to(device)
    # if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(parameters=parameters)
    # if resume is not None:
    #     state_dict = torch.load(resume, map_location=torch.device('cpu'))
    #     if 'optim' in state_dict:
    #         optimizer.load_state_dict(state_dict['optim'])
    #     if 'scheduler' in state_dict:
    #         scheduler.load_state_dict(state_dict['scheduler'])

    # setup partial function to simplify call
    eval_function = partial(evaluate_rerank, model=model, cache_nn_inds=cache_nn_inds,
        recall_ks=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    metrics = eval_function()[0]
    pprint(metrics)
    best_val = (0, metrics, deepcopy(model.state_dict()))

    torch.manual_seed(seed)
    # saving
    save_name = osp.join(temp_dir, 
            '{}_{}.pt'.format(
                        ex.current_run.config['model']['arch'],
                        ex.current_run.config['dataset']['name']
                    )
            )
    os.makedirs(temp_dir, exist_ok=True)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        train_rerank(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, epoch=epoch, ex=ex)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        metrics = eval_function()[0]
        print('Validation [{:03d}]'.format(epoch)), pprint(metrics)
        ex.log_scalar('val.recall@1', metrics[1], step=epoch + 1)

        # save model dict if the chosen validation metric is better
        if metrics[1] >= best_val[1][1]:
            best_val = (epoch + 1, metrics, deepcopy(model.state_dict()))
            torch.save(
                {
                    'state': state_dict_to_cpu(best_val[2]),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_name)

    # logging
    ex.info['recall'] = best_val[1]
    ex.add_artifact(save_name)

    return best_val[1][1]