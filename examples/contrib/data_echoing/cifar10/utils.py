import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data.distributed as data_dist

from torchvision import models
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip

from ignite.contrib.research import ExampleEchoingSampler, MemoizingDataset

import fastresnet


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    elif name in fastresnet.__dict__:
        fn = fastresnet.__dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn()


def get_train_test_loaders(path, batch_size, num_workers, distributed=False, pin_memory=True):

    train_transform = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if not os.path.exists(path):
        os.makedirs(path)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    train_ds = datasets.CIFAR10(root=path, train=True, download=download, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)
    train_eval_ds = datasets.CIFAR10(root=path, train=True, download=False, transform=test_transform)

    train_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)

    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size * 2, sampler=test_sampler,
                                   num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader, train_eval_loader


def get_train_test_loaders_with_example_echoing(path, batch_size, num_workers, 
                                                num_echoes=3, echoing_before_dataaug=True,
                                                distributed=False, pin_memory=True):

    train_transform = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if not os.path.exists(path):
        os.makedirs(path)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    train_ds = datasets.CIFAR10(root=path, train=True, download=download, transform=train_transform)    
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)
    train_eval_ds = datasets.CIFAR10(root=path, train=True, download=False, transform=test_transform)

    if not echoing_before_dataaug:
        train_ds = MemoizingDataset(train_ds)

    train_sampler = ExampleEchoingSampler(num_echoes=num_echoes, dataset_length=len(train_ds))
    test_sampler = None
    if distributed:
        train_sampler = DistributedProxySampler(train_sampler)
        test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)

    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size * 2, sampler=test_sampler,
                                   num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader, train_eval_loader


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
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)
