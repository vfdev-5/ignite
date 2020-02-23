# Basic training configuration
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

import nvidia.dali.ops as ops
import nvidia.dali.types as types

from dataflow.dali_dataloaders import get_train_val_loaders, dali_random_args
from dataflow.transforms import denormalize

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


# ##############################
# Setup Dataflow
# ##############################

assert "DATASET_PATH" in os.environ
data_path = os.environ["DATASET_PATH"]

mean = [0.485 * 255,0.456 * 255,0.406 * 255]
std = [0.229 * 255,0.224 * 255,0.225 * 255]


train_transforms = [
    ops.RandomResizedCrop(device="gpu", size=(train_crop_size, train_crop_size)),
    dali_random_args(ops.Flip(device="gpu"), horizontal=ops.CoinFlip()),    
    dali_random_args(ops.Hsv(device="gpu"), hue=ops.Uniform(range=(-10, 10)), saturation=ops.Uniform(range=(-5, 5))),    
    ops.CropMirrorNormalize(device="gpu", mean=mean, std=std, output_dtype=types.FLOAT, output_layout=types.NCHW)
]


val_transforms = [
    ops.Resize(device="gpu", resize_x=int((256 / 224) * val_crop_size), resize_y=int((256 / 224) * val_crop_size)),
    ops.CropMirrorNormalize(device="gpu", crop=(val_crop_size, val_crop_size), mean=mean, std=std, output_dtype=types.FLOAT, output_layout=types.NCHW)
]


train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=batch_size
)


def prepare_batch(batch, *args, **kwargs):
    x = batch[0]['data']
    y = batch[0]['label'].squeeze(dim=-1)
    return x, y

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

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
