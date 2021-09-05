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
from torch.optim import SGD, lr_scheduler

from models.ingredient import model_ingredient, get_model
from utils import state_dict_to_cpu, Contrastive, num_of_trainable_params
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train_global, evaluate_global

ex = sacred.Experiment('Global (train)', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    epochs = 100
    lr = 0.001
    momentum = 0.9
    nesterov = True
    weight_decay = 4e-4
    scheduler_tau = [60, 80]
    scheduler_gamma = 0.1

    cpu = False  
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('logs', 'temp')

    no_bias_decay = False
    temperature = 0.05
    seed = 364086339
    

@ex.capture
def get_optimizer_scheduler(parameters, lr, momentum, nesterov, weight_decay, scheduler_tau, scheduler_gamma):
    optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True if nesterov and momentum else False)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_tau, gamma=scheduler_gamma)
    return optimizer, scheduler


@ex.capture
def get_loss(temperature):
    return Contrastive(temperature=temperature)


@ex.automain
def main(epochs, cpu, cudnn_flag, temp_dir, seed, no_bias_decay):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model = get_model(num_classes=loaders.num_classes)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    class_loss = get_loss()

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

    # setup partial function to simplify call
    eval_function = partial(evaluate_global, model=model, recall_ks=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

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

        train_global(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, epoch=epoch, ex=ex)

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