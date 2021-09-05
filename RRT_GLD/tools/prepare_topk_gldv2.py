import _init_paths
import os.path as osp
import numpy as np
from tqdm import tqdm
from utils import pickle_save, pickle_load
from utils.data.delf import datum_io

import sacred
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from numpy import linalg as LA


ex = sacred.Experiment('Prepare Top-K (GLD)')
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    data_dir = osp.join('data', 'gldv2')
    feature_name = 'r50_gldv2'
    

@ex.automain
def main(data_dir, feature_name):
    with open(osp.join(data_dir, 'train.txt')) as fid:
        lines = fid.read().splitlines()

    desc = []
    for i in tqdm(range(len(lines))):
        name = osp.splitext(osp.basename(lines[i].split(',')[0]))[0]
        path = osp.join(data_dir, 'delg_'+feature_name, name+'.delg_global')
        x = datum_io.ReadFromFile(path)
        x = x/LA.norm(x, axis=-1)
        desc.append(x)
    desc = np.stack(desc, axis=0)

    x_step = 2000
    y_step = 2000
    n = len(desc)
    nn_inds = []
    for i in tqdm(range(0, n, x_step)):
        tmp_pkl_name = osp.join(data_dir, feature_name+'_nn_%09d.pkl'%i)
        if osp.exists(tmp_pkl_name):
            inds = pickle_load(tmp_pkl_name)
            nn_inds.append(inds)
            continue
        xs = desc[i:min(i+x_step, n)]
        sims = []
        for j in range(0, n, y_step):
            ys = desc[j:min(j+y_step, n)]
            ds = np.matmul(xs, ys.T)
            sims.append(ds)
        sims = np.concatenate(sims, axis=-1)
        inds = np.argsort(-sims, axis=-1)[:, 1:101]
        pickle_save(tmp_pkl_name, inds)
        nn_inds.append(inds)
    nn_inds = np.concatenate(nn_inds, 0)
    print(nn_inds.shape)
    output_path = osp.join(data_dir, 'nn_inds_%s.pkl'%feature_name)
    pickle_save(output_path, nn_inds)
