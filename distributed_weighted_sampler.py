import numpy as np
from typing import Optional
import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

class DistributedWeightedSampler(Sampler):

    def __init__(self, weights: torch.Tensor, dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, replacement: bool = True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))
        self.weights = weights
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.replacement = replacement #sample can be drown again in that row if True

    def calculate_weights(self, targets):
        class_sample_count = np.array([len(np.where(self.dataset.data.y == t)[0]) for t in np.unique(self.dataset.data.y)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.dataset.data.y])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        return samples_weigth

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # data.data.y == labels
        #targets = self.dataset.data.y
        #targets = targets[self.rank:self.total_size:self.num_replicas]
        #assert len(targets) == self.num_samples
        #weights = self.calculate_weights(targets)
        weights = self.weights[self.rank:self.total_size:self.num_replicas]
        weighted_indices = torch.multinomial(weights, self.num_samples, self.replacement).tolist()

        return iter(weighted_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch