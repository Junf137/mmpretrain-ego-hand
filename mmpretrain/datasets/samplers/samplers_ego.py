from torch.utils.data import Sampler
import torch
import math
from mmpretrain.registry import DATA_SAMPLERS
from mmengine.dist import get_world_size, get_rank


@DATA_SAMPLERS.register_module()
class DistributedWeightedSampler(Sampler):
    """Class-balanced sampler usable with MMEngine dataloaders and DDP."""

    def __init__(self, dataset, ann_file_key="ann_file", replacement=True, seed=None):
        # dataset has .data_list where each item has 'gt_label'
        labels = [int(x["gt_label"]) for x in dataset.data_list]
        num_classes = max(labels) + 1
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
        inv = (1.0 / counts).clamp_(max=1e6)  # safe cap
        weights = inv[torch.tensor(labels)]
        self.weights = (weights / weights.sum()).double().tolist()
        self.dataset = dataset
        self.replacement = replacement
        self.seed = seed

        # per-rank num samples
        world_size = get_world_size()
        n = len(dataset)
        # round up to be divisible by world_size
        self.num_samples = int(math.ceil(n / world_size))
        self.total_size = self.num_samples * world_size
        self.rank = get_rank()

    def __iter__(self):
        # sample indices with replacement to fill total_size
        indices = torch.multinomial(
            input=torch.tensor(self.weights), num_samples=self.total_size, replacement=self.replacement
        ).tolist()
        # subsample for this rank
        offset = self.rank * self.num_samples
        indices = indices[offset : offset + self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
