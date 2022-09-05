import os
from pathlib import Path

import torch
from torchvision import datasets, models
from torchvision.transforms import Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, ToTensor
from torchvision.prototype import transforms as TV2


train_transform = Compose(
    [
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


train_transform_v2 = TV2.Compose(
    [
        TV2.Pad(4),
        TV2.RandomCrop(32, fill=128),
        TV2.RandomHorizontalFlip(),
        TV2.ToImageTensor(),
        TV2.ConvertImageDtype(torch.float),
        TV2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform_v2 = TV2.Compose(
    [
        TV2.ToImageTensor(),
        TV2.ConvertImageDtype(torch.float),
        TV2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def get_train_test_datasets(path, use_vision_api_v2=False):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    if use_vision_api_v2:
        print("Using torchvision Transforms API V2")
        train_tfs = train_transform_v2
        test_tfs = test_transform_v2
    else:
        train_tfs = train_transform
        test_tfs = test_transform

    train_ds = datasets.CIFAR10(root=path, train=True, download=download, transform=train_tfs)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_tfs)

    return train_ds, test_ds


def get_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    return fn(num_classes=10)
