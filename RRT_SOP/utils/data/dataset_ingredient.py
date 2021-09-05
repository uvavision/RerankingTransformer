import os.path as osp
from typing import NamedTuple, Optional

from sacred import Ingredient
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torchvision import transforms

from .image_dataset import ImageDataset
from .utils import RandomReplacedIdentitySampler, TripletSampler

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def config():
    name = 'sop'
    data_path = 'data/Stanford_Online_Products'
    train_file = 'train.txt'
    test_file = 'test.txt'

    batch_size = 128
    sample_per_id = 2
    assert (batch_size % sample_per_id == 0)
    test_batch_size = 256
    sampler = 'random'

    num_workers = 8  
    pin_memory = True

    crop_size = 224
    recalls = [1, 10, 100, 1000]

    num_identities = batch_size // sample_per_id 
    num_iterations = 59551 // batch_size

    train_cache_nn_inds  = None
    test_cache_nn_inds   = None


@data_ingredient.named_config
def sop_global():
    name = 'sop_global'
    batch_size = 800
    test_batch_size = 800
    sampler = 'random_id'


@data_ingredient.named_config
def sop_rerank():
    name = 'sop_rerank'
    batch_size = 300
    test_batch_size = 600
    sampler = 'triplet'
    recalls = [1, 10, 100]

    train_cache_nn_inds  = 'rrt_sop_caches/rrt_r50_sop_nn_inds_train.pkl'
    test_cache_nn_inds   = 'rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl'


@data_ingredient.capture
def get_transforms(crop_size):
    train_transform, test_transform = [], []
    train_transform.extend([
        transforms.RandomResizedCrop(size=crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])
    test_transform.append(transforms.Resize((256, 256)))
    test_transform.append(transforms.CenterCrop(size=224))
    test_transform.append(transforms.ToTensor())
    return transforms.Compose(train_transform), transforms.Compose(test_transform)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


@data_ingredient.capture
def get_sets(name, data_path, train_file, test_file, num_workers):
    train_transform, test_transform = get_transforms()

    train_lines = read_file(osp.join(data_path, train_file))
    train_samples = [(osp.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in train_lines]
    train_set = ImageDataset(train_samples, transform=train_transform)
    query_train_set = ImageDataset(train_samples, transform=test_transform)

    if isinstance(test_file, list) and len(test_file) == 2:
        query_lines   = read_file(osp.join(data_path, test_file[0]))
        gallery_lines = read_file(osp.join(data_path, test_file[1]))
        query_samples   = [(osp.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in query_lines]
        gallery_samples = [(osp.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in gallery_lines]
        query_set   = ImageDataset(query_samples,   transform=test_transform)
        gallery_set = ImageDataset(gallery_samples, transform=test_transform)
    else:
        query_lines   = read_file(osp.join(data_path, test_file))
        query_samples = [(osp.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in query_lines]
        query_set = ImageDataset(query_samples, transform=test_transform)
        gallery_set = None

    return (train_set, query_train_set), (query_set, gallery_set)


class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None


@data_ingredient.capture
def get_loaders(batch_size, test_batch_size, 
        num_workers, pin_memory, 
        sampler, recalls,
        num_iterations=None, 
        num_identities=None,
        train_cache_nn_inds=None,
        test_cache_nn_inds=None):

    (train_set, query_train_set), (query_set, gallery_set) = get_sets()

    if sampler == 'random':
        train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=True)
    elif sampler == 'triplet':
        if train_cache_nn_inds and osp.exists(train_cache_nn_inds):
            train_sampler = TripletSampler(train_set.targets, batch_size, train_cache_nn_inds)
        else:
            # For evaluation only
            train_sampler = None
    elif sampler == 'random_id':
        train_sampler = RandomReplacedIdentitySampler(train_set.targets, batch_size, 
            num_identities=num_identities, num_iterations=num_iterations)
    else:
        raise ValueError('Invalid choice of sampler ({}).'.format(sampler))
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    query_loader = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = None
    if gallery_set is not None:
        gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=max(train_set.targets) + 1), recalls
