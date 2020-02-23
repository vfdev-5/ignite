# Basic training configuration
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

from ignite.contrib.experimental import MemoizingDataset, ExampleEchoingSampler
from ignite.contrib.experimental.data_echoing import DistributedProxySampler

from dataflow.datasets import get_train_dataset
from dataflow.dataloaders import get_train_val_loaders, get_dataloader
from dataflow.transforms import denormalize, prepare_batch_fp32

from efficientnet_pytorch import EfficientNet


# ##############################
# Global configs
# ##############################

seed = 19
device = "cuda"
debug = False

# config to measure time passed to prepare batches and report measured time before the training
benchmark_dataflow = True
benchmark_dataflow_num_iters = 100

fp16_opt_level = "O2"
val_interval = 2

train_crop_size = 224
val_crop_size = 320

batch_size = 128  # batch size per local rank
num_workers = 16  # num_workers per local rank

num_echoes = 3

# ##############################
# Setup Dataflow
# ##############################

assert "DATASET_PATH" in os.environ
data_path = os.environ["DATASET_PATH"]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = A.Compose(
    [
        A.RandomResizedCrop(train_crop_size, train_crop_size, scale=(0.08, 1.0)),
        A.HorizontalFlip(),
        A.CoarseDropout(max_height=64, max_width=64),
        A.HueSaturationValue(),
        A.Normalize(mean=mean, std=std),
        ToTensor(),
    ]
)

val_transforms = A.Compose(
    [
        # https://github.com/facebookresearch/FixRes/blob/b27575208a7c48a3a6e0fa9efb57baa4021d1305/imnet_resnet50_scratch/transforms.py#L76
        A.Resize(int((256 / 224) * val_crop_size), int((256 / 224) * val_crop_size)),
        A.CenterCrop(val_crop_size, val_crop_size),
        A.Normalize(mean=mean, std=std),
        ToTensor(),
    ]
)

train_ds = get_train_dataset(data_path)
train_ds = MemoizingDataset(train_ds)

train_sampler = ExampleEchoingSampler(num_echoes=num_echoes, dataset_length=len(train_ds))
train_sampler = DistributedProxySampler(train_sampler)
train_loader = get_dataloader(
    train_ds,
    transforms=train_transforms,
    limit_num_samples=100 if debug else None,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=train_sampler,
    pin_memory=True,
)

_, val_loader, train_eval_loader = get_train_val_loaders(
    root_path=data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    train_sampler="distributed",
    limit_val_num_samples=100 if debug else None,
)

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

prepare_batch = prepare_batch_fp32

# epoch_length = (dataset_size / num_echoes + batch_size - 1) // batch_size
epoch_length = (len(train_loader.sampler) // num_echoes + batch_size - 1) // batch_size


# ##############################
# Setup Model
# ##############################

model = EfficientNet.from_name('efficientnet-b0')


# ##############################
# Setup Solver
# ##############################

num_epochs = 105

criterion = nn.CrossEntropyLoss()

le = len(train_loader)

base_lr = 0.1 * (batch_size * dist.get_world_size() / 256.0)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
lr_scheduler = lrs.MultiStepLR(optimizer, milestones=[30 * le, 60 * le, 90 * le, 100 * le], gamma=0.1)
