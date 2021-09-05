from typing import Dict, List, Optional
import faiss, time
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter:
    """Computes and stores the average and current value on device"""

    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg


@torch.no_grad()
def recall_at_ks(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = False) -> Dict[int, float]:
    """
    Compute the recall between samples at each k. This function uses about 8GB of memory.

    Parameters
    ----------
    query_features : torch.Tensor
        Features for each query sample. shape: (num_queries, num_features)
    query_labels : torch.LongTensor
        Labels corresponding to the query features. shape: (num_queries,)
    ks : List[int]
        Values at which to compute the recall.
    gallery_features : torch.Tensor
        Features for each gallery sample. shape: (num_queries, num_features)
    gallery_labels : torch.LongTensor
        Labels corresponding to the gallery features. shape: (num_queries,)
    cosine : bool
        Use cosine distance between samples instead of euclidean distance.

    Returns
    -------
    recalls : Dict[int, float]
        Values of the recall at each k.

    """
    device = query_features.device 

    offset = 0
    if gallery_features is None and gallery_labels is None:
        offset = 1
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    if cosine:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])
    max_k = max(ks)

    index = faiss.IndexFlatL2(g_f.shape[1])
    # if device == torch.device(type='cpu'):
    #     index = faiss.IndexFlatL2(g_f.shape[1])
    #     # if faiss.get_num_gpus() > 0:
    #     #     index = faiss.index_cpu_to_all_gpus(index)
    # else:
    #     res = faiss.StandardGpuResources()
    #     flat_config = faiss.GpuIndexFlatConfig()
    #     flat_config.device = 0
    #     index = faiss.GpuIndexFlatL2(res, g_f.shape[1], flat_config)
    print('--------------------------------------------')
    print('FAISS initialized')
    index.add(g_f)
    print('--------------------------------------------')
    print('Index features added, start KNN searching')
    closest_dists, closest_indices = index.search(q_f, max_k + offset)
    print('--------------------------------------------')
    closest_dists = closest_dists[:,:(int(max_k) + offset)]
    closest_indices = closest_indices[:,:(int(max_k) + offset)]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (q_l[:, None] == g_l[indices]).any(1).mean()
    return {k: round(v * 100, 2) for k, v in recalls.items()}, closest_dists[:, offset:], closest_indices[:, offset:]


@torch.no_grad()
def recall_at_ks_rerank(
    global_desc, local_desc, positions, masks, labels,
    ks: List[int],
    matcher: nn.Module,
    cache_nn_inds: torch.Tensor) -> Dict[int, float]:

    labels_cpu = labels.cpu().numpy()
    device = next(matcher.parameters()).device
    num_samples, top_k = cache_nn_inds.size()
    top_k = min(top_k, 100)


    bsize = 256
    scores = []
    total_time = 0.0
    ######################################################################
    print('--------------------------------------------')
    print('Start reranking')
    for i in tqdm(range(top_k)):
        k_scores = []
        for j in range(0, num_samples, bsize):
            query_global = global_desc[j:(j+bsize)]
            index_global = global_desc[cache_nn_inds[j:(j+bsize), i]]
            query_local  = local_desc[j:(j+bsize)]
            index_local  = local_desc[cache_nn_inds[j:(j+bsize), i]]
            query_points = positions[j:(j+bsize)]
            index_points = positions[cache_nn_inds[j:(j+bsize), i]]
            query_masks  = masks[j:(j+bsize)]
            index_masks  = masks[cache_nn_inds[j:(j+bsize), i]]
            
            start = time.time()
            current_scores, _, _ = matcher(None, None, True, 
                src_global=query_global.to(device), src_local=query_local.to(device), src_mask=query_masks.to(device), src_positions=query_points.to(device),
                tgt_global=index_global.to(device), tgt_local=index_local.to(device), tgt_mask=index_masks.to(device), tgt_positions=index_points.to(device),
            )
            end = time.time()
            total_time += end-start
            k_scores.append(current_scores.cpu())
        k_scores = torch.cat(k_scores, 0)
        scores.append(k_scores)
    print('time', total_time/num_samples)
    scores = torch.stack(scores, -1)
    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_dists = closest_dists.numpy()    
    # closest_indices = torch.zeros((num_samples, top_k)).long()
    # for i in range(num_samples):
    #     for j in range(top_k):
    #         closest_indices[i, j] = cache_nn_inds[i, indices[i, j]]
    # closest_indices = closest_indices.numpy()
    closest_indices = torch.gather(cache_nn_inds, -1, indices).numpy()   

    max_k = max(ks)
    recalls = {}
    for k in ks:
        indices = closest_indices[:, :k]
        recalls[k] = (labels_cpu[:, None] == labels_cpu[indices]).any(1).mean()
    return {k: round(v * 100, 2) for k, v in recalls.items()}, closest_dists, closest_indices
