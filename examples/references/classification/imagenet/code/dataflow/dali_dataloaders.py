from typing import Callable, Optional, Tuple, Union, Sequence

import random
from pathlib import Path

import numpy as np

import torch.distributed as dist
from torch.utils.data import Subset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torchvision.datasets import ImageNet

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


from dataflow.dataloaders import opencv_loader


def get_train_val_loaders(
    root_path: str,
    train_transforms: Sequence,
    val_transforms: Sequence,
    batch_size: int = 16,
    num_workers: int = 8,
    val_batch_size: Optional[int] = None,
    device_id: int = 0,
):

    assert Path(root_path).exists()

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    train_ds = ImageNet(
        root_path, split="train", loader=opencv_loader
    )
    val_ds = ImageNet(
        root_path, split="val", loader=opencv_loader
    )

    # random samples for evaluation on training dataset
    if len(val_ds) < len(train_ds):
        np.random.seed(len(val_ds))
        train_eval_indices = np.random.permutation(len(train_ds))[: len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    train_pipeline = ClassificationPipeline(
        dataset=train_ds,
        transforms=train_transforms,
        batch_size=batch_size,
        device_id=device_id,
        num_threads=num_workers
    )
    train_pipeline.build()
    train_loader = DALIClassificationIterator(train_pipeline, size=train_pipeline.size // batch_size, auto_reset=True)

    val_pipeline = ClassificationPipeline(
        dataset=val_ds,
        transforms=val_transforms,
        batch_size=val_batch_size,
        device_id=device_id,
        num_threads=num_workers,
        shuffle=False,
    )
    val_pipeline.build()
    val_loader = DALIClassificationIterator(
        val_pipeline,
        size=val_pipeline.size // val_batch_size,
        fill_last_batch=False,
        auto_reset=True)

    train_eval_pipeline = ClassificationPipeline(
        dataset=train_eval_ds,
        transforms=val_transforms,
        batch_size=val_batch_size,
        device_id=device_id,
        num_threads=num_workers,
        shuffle=False,
    )
    train_eval_pipeline.build()
    train_eval_loader = DALIClassificationIterator(
        train_eval_pipeline,
        size=train_eval_pipeline.size // val_batch_size,
        fill_last_batch=False,
        auto_reset=True)

    return AsTorchDataLoader(train_loader), AsTorchDataLoader(val_loader), AsTorchDataLoader(train_eval_loader)


class TorchDatasetReader:

    def __init__(self, dataset, batch_size, shuffle=False, shard_id=0, num_shards=1):

        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.total_size = len(dataset)
        if num_shards > 1:
            indices = list(range(len(dataset)))
            indices = indices[self.len * shard_id // num_shards: self.total_size * (shard_id + 1) // num_shards]
            dataset = Subset(dataset, indices)

        self.dataset = dataset
        self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def __iter__(self):

        while True:
            x = []
            y = []
            for idx in self.sampler:
                dp = self.dataset[idx]
                x.append(np.asarray(dp[0]))
                y.append(np.array(dp[1], dtype='long'))
                if len(x) == self.batch_size:
                    yield x, y
                    x = []
                    y = []

    @property
    def size(self):
        return len(self.dataset)


class ClassificationPipeline(Pipeline):

    def __init__(self, dataset, transforms, batch_size=16, device_id=0, num_threads=4, shuffle=True):
        super(
            ClassificationPipeline,
            self).__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id)
        num_shards = dist.get_world_size() if dist.is_initialized() else 1

        self.transforms = transforms

        self.image = ops.ExternalSource()
        self.target = ops.ExternalSource()
        extdata = TorchDatasetReader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            shard_id=device_id,
            num_shards=num_shards)
        self.external_data = extdata
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.batch_x = self.image()
        self.batch_y = self.target()

        batch_x = self.batch_x.gpu()
        for op in self.transforms:
            batch_x = op(batch_x)

        return batch_x, self.batch_y.gpu()

    @property
    def size(self):
        return self.external_data.size

    def iter_setup(self):
        try:
            x, y = next(self.iterator)
            self.feed_input(self.batch_x, x)
            self.feed_input(self.batch_y, y)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class AsTorchDataLoader:

    def __init__(self, dali_iterator):
        self.iterator = dali_iterator

    def __len__(self):
        return self.iterator._size

    def __iter__(self):
        return iter(self.iterator)


def dali_random_args(op, **kwargs):

    def wrapper(*args):
        called_kwargs = {k: v() for k, v in kwargs.items()}
        return op(*args, **called_kwargs)

    return wrapper
