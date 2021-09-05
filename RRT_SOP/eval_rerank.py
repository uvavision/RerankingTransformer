import os, math
from copy import deepcopy
from functools import partial
from pprint import pprint
import os.path as osp

import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler
from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load, pickle_save
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import evaluate_rerank

ex = sacred.Experiment('Rerank (eval)', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = os.path.join('logs', 'temp')
    seed = 0
    resume = None
    cache_nn_inds = 'caches/sop/nn_inds_test.pkl'


@ex.automain
def main(cpu, cudnn_flag, temp_dir, seed, resume, cache_nn_inds):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model = get_model(num_classes=loaders.num_classes)
    state_dict = torch.load(resume, map_location=torch.device('cpu'))
    if 'state' in state_dict:
        state_dict = state_dict['state']
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    # if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.eval()

    cache_nn_inds = pickle_load(cache_nn_inds)
    cache_nn_inds = torch.from_numpy(cache_nn_inds)

    # setup partial function to simplify call
    eval_function = partial(evaluate_rerank, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall_ks=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    result, nn_dists, nn_inds = eval_function()
    pprint(result)

    pickle_save(osp.join(temp_dir, 'nn_dists.pkl'), nn_dists)
    pickle_save(osp.join(temp_dir, 'nn_inds.pkl'), nn_inds)
