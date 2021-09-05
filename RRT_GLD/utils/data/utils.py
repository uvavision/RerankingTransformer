import numpy as np
import os.path as osp
import pickle, json, random

import torch


###############################################################
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
def cid2filename(cid, prefix):
    return osp.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


class TripletSampler():
    def __init__(self, labels, batch_size, nn_inds_path, num_candidates):
        self.batch_size     = batch_size
        self.num_candidates = num_candidates
        self.cache_nn_inds  = pickle_load(nn_inds_path)
        self.labels = labels
        assert (len(self.cache_nn_inds) == len(labels))
        #############################################################################
        ## Collect valid tuples
        valids = np.zeros_like(labels)
        for i in range(len(self.cache_nn_inds)):
            nnids = self.cache_nn_inds[i]
            query_label = labels[i]
            index_labels = np.array([labels[j] for j in nnids])
            positives = np.where(index_labels == query_label)[0]
            if len(positives) < 5:
                continue
            valids[i] = 1
        self.valids = np.where(valids > 0)[0]
        self.num_samples = len(self.valids)

    def __iter__(self):
        batch = []
        cands = torch.randperm(self.num_samples).tolist()
        for i in range(len(cands)):
            anchor_idx = self.valids[cands[i]]
            anchor_label = self.labels[anchor_idx]
            nnids = self.cache_nn_inds[anchor_idx]

            positive_inds = [j for j in nnids if self.labels[j] == anchor_label]
            negative_inds = [j for j in nnids if self.labels[j] != anchor_label]
            assert(len(positive_inds) > 0)
            assert(len(negative_inds) > 0)

            random.shuffle(positive_inds)
            random.shuffle(negative_inds)

            batch.append(anchor_idx)
            batch.append(positive_inds[0]) 
            batch.append(negative_inds[0])

            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (self.num_samples * 3 + self.batch_size - 1) // self.batch_size

