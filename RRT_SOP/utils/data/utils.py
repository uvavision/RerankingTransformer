from random import sample, choices
from typing import Union, List

import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision.transforms import functional as F


###############################################################
import pickle, json


def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)
###############################################################


class RandomReplacedIdentitySampler(Sampler):
    def __init__(self, 
            labels: Union[List[int], torch.Tensor], batch_size: int, 
            num_identities: int, num_iterations: int):
        self.num_identities = num_identities
        self.num_iterations = num_iterations
        self.samples_per_id = batch_size // num_identities
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.counts = torch.bincount(self.labels)
        self.label_indices = [torch.nonzero(self.labels == i, as_tuple=False).squeeze(1).tolist() for i in range(len(self.counts))]

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        possible_ids = [i for i in range(len(self.label_indices)) if len(self.label_indices[i]) >= self.samples_per_id]

        for i in range(self.num_iterations):
            batch = []
            selected_ids = sample(possible_ids, k=self.num_identities)
            for s_id in selected_ids:
                batch.extend(sample(self.label_indices[s_id], k=self.samples_per_id))
            yield batch


class TripletSampler():
    def __init__(self, labels, batch_size, cache_nn_inds):
        self.num_samples = len(labels)
        self.batch_size = batch_size
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.counts = torch.bincount(self.labels)
        self.cache_nn_inds = pickle_load(cache_nn_inds)
        self.label_indices = [torch.nonzero(self.labels == i, as_tuple=False).squeeze(1).tolist() for i in range(len(self.counts))]

    def __iter__(self):
        batch = []
        cands = torch.randperm(self.num_samples).tolist()
        for i in range(self.num_samples):
            idx = cands[i]
            current_label = self.labels[idx]
            topk_inds = self.cache_nn_inds[idx, :100]
            positive_inds = [ni for ni in self.label_indices[current_label] if ni != idx]
            negative_inds = [ni for ni in topk_inds if self.labels[ni] != current_label]
            # if len(positive_inds) == 0:
            #     positive_inds = [ni for ni in self.label_indices[current_label] if ni != idx]
            assert(len(positive_inds) > 0)
            assert(len(negative_inds) > 0)

            batch.append(idx)
            pid = choices(positive_inds)[0]
            nid = choices(negative_inds)[0]
            batch.append(pid) 
            batch.append(nid)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (self.num_samples * 3 + self.batch_size - 1) // self.batch_size
