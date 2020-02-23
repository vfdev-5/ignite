from typing import Type, Callable

from collections.abc import Mapping

import numpy as np

import cv2

from torch.utils.data import Dataset

from torchvision.datasets import ImageNet


class TransformedDataset(Dataset):
    def __init__(self, ds: Dataset, transform_fn: Callable):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        if isinstance(dp, Mapping):
            return self.transform_fn(**dp)
        return self.transform_fn(dp)


def opencv_loader(path):
    img = cv2.imread(path)
    assert img is not None, "Image at '{}' has a problem".format(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_train_dataset(root_path):
    train_ds = ImageNet(
        root_path, split="train", loader=opencv_loader
    )
    return TransformedDataset(train_ds, lambda dp: {"image": dp[0], "target": dp[1]})


def get_val_dataset(root_path):
    val_ds = ImageNet(
        root_path, split="val", loader=opencv_loader
    )
    return TransformedDataset(val_ds, lambda dp: {"image": dp[0], "target": dp[1]})
