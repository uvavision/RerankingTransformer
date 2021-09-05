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
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate

ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    visdom_port = None
    visdom_freq = 20
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('logs', 'temp')
    resume = None
    seed = 0


@ex.automain
def main(cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, resume):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    # device = torch.device('cpu')
    # callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model = get_model()
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state'], strict=True)

    model.to(device)
    # if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)
    model.eval()

    nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    metrics = eval_function()
    pprint(metrics)
    best_val = (0, metrics, deepcopy(model.state_dict()))

    return best_val[1]
