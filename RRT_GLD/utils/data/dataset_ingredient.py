import os.path as osp
from copy import deepcopy
from sacred import Ingredient
from typing import NamedTuple, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from .dataset import FeatureDataset
from .utils import pickle_load, TripletSampler


data_ingredient = Ingredient('dataset')


@data_ingredient.config
def config():
    name = None
    desc_name = None
    train_data_dir = None
    test_data_dir  = None
    train_txt = None
    test_txt  = None
    test_gnd_file = None
    
    batch_size      = 36
    test_batch_size = 36
    max_sequence_len = 500
    sampler = 'random'

    num_workers = 8  # number of workers used ot load the data
    pin_memory  = True  # use the pin_memory option of DataLoader 
    recalls = [1, 5, 10]
    ###############################################
    ## Negative sampling
    num_candidates = 100


@data_ingredient.named_config
def roxford_r50_gldv1():
    name = 'roxford_r50_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def roxford_r50_gldv2():
    name = 'roxford_r50_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'random'


@data_ingredient.named_config
def rparis_r50_gldv1():
    name = 'rparis_r50_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def rparis_r50_gldv2():
    name = 'rparis_r50_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'random'


@data_ingredient.named_config
def roxford_r101_gldv1():
    name = 'roxford_r101_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def roxford_r101_gldv2():
    name = 'roxford_r101_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv2'
    sampler = 'random'


@data_ingredient.named_config
def rparis_r101_gldv1():
    name = 'rparis_r101_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r101_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def rparis_r101_gldv2():
    name = 'rparis_r101_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r101_gldv2'
    sampler = 'random'


@data_ingredient.named_config
def gldv2_roxford_r50_gldv1():
    name = 'gldv2_roxford_r50_gldv1'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'triplet'


@data_ingredient.named_config
def gldv2_roxford_r101_gldv1():
    name = 'gldv2_roxford_r101_gldv1'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv1'
    sampler = 'triplet'


@data_ingredient.named_config
def gldv2_roxford_r50_gldv2():
    name = 'gldv2_roxford_r50_gldv2'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'triplet'


@data_ingredient.named_config
def gldv2_roxford_r101_gldv2():
    name = 'gldv2_roxford_r101_gldv2'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv2'
    sampler = 'triplet'


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None


@data_ingredient.capture
def get_sets(desc_name, 
        train_data_dir, test_data_dir, 
        train_txt, test_txt, test_gnd_file, 
        max_sequence_len):
    ####################################################################################################################################
    train_lines     = read_file(osp.join(train_data_dir, train_txt))
    train_samples   = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in train_lines]
    train_set       = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len)
    query_train_set = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len)
    ####################################################################################################################################
    test_gnd_data = None if test_gnd_file is None else pickle_load(osp.join(test_data_dir, test_gnd_file))
    query_lines   = read_file(osp.join(test_data_dir, test_txt[0]))
    gallery_lines = read_file(osp.join(test_data_dir, test_txt[1]))
    query_samples   = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in query_lines]
    gallery_samples = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in gallery_lines]
    gallery_set = FeatureDataset(test_data_dir, gallery_samples, desc_name, max_sequence_len)
    query_set   = FeatureDataset(test_data_dir, query_samples,   desc_name, max_sequence_len, gnd_data=test_gnd_data)
        
    return (train_set, query_train_set), (query_set, gallery_set)


@data_ingredient.capture
def get_loaders(desc_name, train_data_dir, 
    batch_size, test_batch_size, 
    num_workers, pin_memory, 
    sampler, recalls,
    num_candidates=100):

    (train_set, query_train_set), (query_set, gallery_set) = get_sets()

    if sampler == 'random':
        train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
    elif sampler == 'triplet':
        train_nn_inds = osp.join(train_data_dir, 'nn_inds_%s.pkl'%desc_name)
        train_sampler = TripletSampler(train_set.targets, batch_size, train_nn_inds, num_candidates)
    else:
        raise ValueError('Invalid choice of sampler ({}).'.format(sampler))
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
        
    query_loader   = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
