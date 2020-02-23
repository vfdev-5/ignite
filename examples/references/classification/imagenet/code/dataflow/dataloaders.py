from typing import Callable, Optional, Tuple, Union

import numpy as np
import cv2


from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.data.dataset import Subset
import torch.utils.data.distributed as data_dist

from dataflow.datasets import get_train_dataset, get_val_dataset, TransformedDataset


def get_dataloader(
    dataset: Dataset, transforms: Optional[Callable] = None, limit_num_samples: Optional[int] = None, **kwargs
):
    if transforms is not None:
        dataset = TransformedDataset(dataset, transform_fn=transforms)

    if limit_num_samples is not None:
        np.random.seed(limit_num_samples)
        indices = np.random.permutation(len(dataset))[:limit_num_samples]
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, **kwargs)


def get_train_val_loaders(
    root_path: str,
    train_transforms: Callable,
    val_transforms: Callable,
    batch_size: int = 16,
    num_workers: int = 8,
    val_batch_size: Optional[int] = None,
    pin_memory: bool = True,
    train_sampler: Optional[Union[Sampler, str]] = None,
    limit_train_num_samples: Optional[int] = None,
    limit_val_num_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = get_train_dataset(root_path)
    val_ds = get_val_dataset(root_path)

    # random samples for evaluation on training dataset
    if len(val_ds) < len(train_ds):
        np.random.seed(len(val_ds))
        train_eval_indices = np.random.permutation(len(train_ds))[: len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    val_sampler = None
    if isinstance(train_sampler, str):
        assert train_sampler == "distributed"
        train_sampler = data_dist.DistributedSampler(train_ds)
        val_sampler = data_dist.DistributedSampler(val_ds, shuffle=False)

    train_loader = get_dataloader(
        train_ds,
        transforms=train_transforms,
        limit_num_samples=limit_train_num_samples,
        shuffle=train_sampler is None,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size
    val_loader = get_dataloader(
        val_ds,
        transforms=val_transforms,
        limit_num_samples=limit_val_num_samples,
        shuffle=False,
        sampler=val_sampler,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    train_eval_loader = get_dataloader(
        train_eval_ds,
        transforms=val_transforms,
        shuffle=False,
        sampler=val_sampler,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_eval_loader
