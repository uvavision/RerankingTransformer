import os, math
import os.path as osp
from copy import deepcopy
from functools import partial
from pprint import pprint

import sacred
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds

import torch
import torch.nn as nn
from torch.backends import cudnn

from utils import pickle_load, pickle_save
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from models.ingredient import model_ingredient, get_model
from utils.training import evaluate_global


ex = sacred.Experiment('Global (eval)', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    cpu = False 
    cudnn_flag = 'benchmark'
    temp_dir = os.path.join('outputs', 'temp')
    seed = 0
    resume = None
    query_set = 'test'


@ex.automain
def main(cpu, cudnn_flag, temp_dir, seed, resume, query_set):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()
    model = get_model(num_classes=loaders.num_classes)
    state_dict = torch.load(resume, map_location=torch.device('cpu'))
    if 'state' in state_dict:
        state_dict = state_dict['state']
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model = nn.DataParallel(model)
    model.eval()

    # setup partial function to simplify call
    if query_set == 'test':
        query_loader=loaders.query
    else:
        query_loader=loaders.query_train
    eval_function = partial(evaluate_global, model=model, recall_ks=recall_ks, query_loader=query_loader, gallery_loader=loaders.gallery)
        

    # setup best validation logger
    result, nn_dists, nn_inds = eval_function()
    pprint(result)
    pickle_save(osp.join(temp_dir, 'nn_dists.pkl'), nn_dists)
    pickle_save(osp.join(temp_dir, 'nn_inds.pkl'), nn_inds)
