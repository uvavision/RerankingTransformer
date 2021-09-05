import os.path as osp
from typing import NamedTuple, Optional

from sacred import Ingredient
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torchvision import transforms

from .image_dataset import ImageDataset
from .utils import RandomReplacedIdentitySampler, TripletSampler, pickle_load

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def config():
    name = 'sop_super'
    data_path = 'data/Stanford_Online_Products'
    train_file = 'train.txt'
    test_file = 'test.txt'
    image_size = 320
    max_desc = 500
    batch_size = 96
    test_batch_size = 96
    num_workers = 8  
    pin_memory = True
    recalls = [1, 10, 100, 1000]

    superpoint_file      = 'rrt_sop_caches/superpoint_sop_%d_%d.pkl'%(image_size, max_desc)
    train_cache_nn_inds  = 'rrt_sop_caches/rrt_superpoint_sop_nn_inds_train.pkl'
    test_cache_nn_inds   = 'rrt_sop_caches/rrt_superpoint_sop_nn_inds_test.pkl'


@data_ingredient.capture
def get_transforms(image_size):
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


@data_ingredient.capture
def get_sets(name, data_path, train_file, test_file, superpoint_file, max_desc, num_workers):
    transform = get_transforms()
    superpoints = pickle_load(superpoint_file)

    train_lines = read_file(osp.join(data_path, train_file))
    train_samples = [(osp.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in train_lines]
    train_set = ImageDataset(train_samples, superpoints, max_desc, transform=transform)

    test_lines   = read_file(osp.join(data_path, test_file))
    test_samples = [(osp.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in test_lines]
    test_set = ImageDataset(test_samples, superpoints, max_desc, transform=transform)

    return train_set, test_set


class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    gallery: Optional[DataLoader] = None


@data_ingredient.capture
def get_loaders(
        batch_size, test_batch_size, 
        num_workers, pin_memory, recalls,
        train_cache_nn_inds=None,
        test_cache_nn_inds=None):

    train_set, test_set = get_sets()

    if train_cache_nn_inds and osp.exists(train_cache_nn_inds):
        train_sampler = TripletSampler(train_set.targets, batch_size, train_cache_nn_inds)
    else:
        # For evaluation only
        train_sampler = None
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    query_loader = DataLoader(test_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = None

    return MetricLoaders(train=train_loader, query=query_loader, gallery=gallery_loader, num_classes=max(train_set.targets) + 1), recalls
