from typing import Optional
from functools import lru_cache

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, Sampler
import torch.utils.data.distributed as data_dist


class ExampleEchoingSampler(Sampler):
    """Sampler to perform "example echoing": sample the input example multiple times before or after data augmentation.
    Reference: https://arxiv.org/abs/1907.05550

    1) Batch without "example echoing":

    .. code-block:: text

        batch = [d1_aug1, d2_aug2, d3_aug3, ..., dB_augB]

    2) Batch with "example echoing" and `num_echoes=2`:

    .. code-block:: text

        # before data augs
        batch = [d1_aug1, d1_aug2, d2_aug3, d2_aug4, ..., dB/2_augB-1, dB/2_augB]
        # after data augs
        batch = [d1_aug1, d1_aug1, d2_aug2, d2_aug2, ..., dB/2_augB/2, dB/2_augB/2]

    Usage:

    Example echoing before data augmentations:

    .. code-block:: python

        from torch.utils.data import DataLoader
        from ignite.contrib.research import MemoizingDataset, ExampleEchoingSampler

        train_dataset = UserDatasetWithoutAugs(*args, **kwargs)
        train_dataset = MemoizingDataset(train_dataset)  # cache output
        train_dataset = TransformedDataset(train_dataset, transforms=data_augs)

        sampler = ExampleEchoingSampler(num_echoes=3, dataset_length=len(train_dataset))
        train_loader = DataLoader(train_dataset, sampler=sampler, **kwargs)

    Example echoing after data augmentations:

    .. code-block:: python

        from torch.utils.data import DataLoader
        from ignite.contrib.research import MemoizingDataset, ExampleEchoingSampler

        train_dataset = DatasetWithAugs(*args, **kwargs)
        train_dataset = MemoizingDataset(train_dataset)  # cache output

        sampler = ExampleEchoingSampler(num_echoes=3, dataset_length=len(train_dataset))
        train_loader = DataLoader(train_dataset, sampler=sampler, **kwargs)

    """

    def __init__(
        self, num_echoes: int = 3, base_sampler: Optional[Sampler] = None, dataset_length: Optional[int] = None
    ):
        if (base_sampler is None and dataset_length is None) or (
            base_sampler is not None and dataset_length is not None
        ):
            raise ValueError("One of the arguments only should be defined: base_sampler or dataset_length")
        super(ExampleEchoingSampler, self).__init__([])

        if base_sampler is None:
            data_source = list(range(dataset_length))
            base_sampler = RandomSampler(data_source)

        self.num_echoes = num_echoes
        self.base_sampler = base_sampler
        self.sample_indices = None

    def setup_sample_indices(self):
        self.sample_indices = []
        for i in self.base_sampler:
            self.sample_indices.extend([i for _ in range(self.num_echoes)])

    def __iter__(self):
        self.setup_sample_indices()
        return iter(self.sample_indices)

    def __len__(self):
        return len(self.base_sampler) * self.num_echoes


class MemoizingDataset(Dataset):
    """Helper dataset wrapper to cache the output of `__getitem__` method.
    Input dataset should not be `torch.utils.data.IterableDataset`.
    """

    lru_cache_maxsize = 128

    def __init__(self, dataset: torch.utils.data.Dataset):
        if hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__"):
            raise TypeError("Input argument dataset should not be an iterable dataset")
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=lru_cache_maxsize)
    def __getitem__(self, index: int):
        return self.dataset[index]


# Waiting until https://github.com/pytorch/pytorch/issues/23430 to be closed
class DistributedProxySampler(data_dist.DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)
