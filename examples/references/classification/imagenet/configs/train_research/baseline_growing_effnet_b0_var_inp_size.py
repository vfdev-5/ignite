# Basic training configuration
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import denormalize, prepare_batch_fp32

from models.growing_effnet_b0 import GrowingEffNetB0


# ##############################
# Global configs
# ##############################

seed = 19
device = "cuda"
debug = False

# config to measure time passed to prepare batches and report measured time before the training
benchmark_dataflow = True
benchmark_dataflow_num_iters = 100

start_by_validation = True

fp16_opt_level = "O2"
val_interval = 1

min_train_crop_size = 64
max_train_crop_size = 224
val_crop_size = 320

batch_size = 128  # batch size per local rank
num_workers = 16  # num_workers per local rank


# ##############################
# Setup Dataflow
# ##############################

assert "DATASET_PATH" in os.environ
data_path = os.environ["DATASET_PATH"]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

resized_crop_aug = A.RandomResizedCrop(min_train_crop_size, min_train_crop_size, scale=(0.08, 1.0))
resize_aug = A.Resize(min_train_crop_size, min_train_crop_size)

train_transforms = A.Compose(
    [
        resized_crop_aug,
        A.HorizontalFlip(),
        A.CoarseDropout(max_height=min_train_crop_size // 2, max_width=min_train_crop_size // 2, p=0.3),
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

train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=batch_size,
    pin_memory=True,
    train_sampler="distributed"
)

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

prepare_batch = prepare_batch_fp32

# ##############################
# Setup Model
# ##############################

model = GrowingEffNetB0(in_channels=3, num_classes=1000, reduce=4)


# ##############################
# Setup Solver
# ##############################

num_epochs = 105

criterion = nn.CrossEntropyLoss()

le = len(train_loader)

base_lr = 0.1 * (batch_size * dist.get_world_size() / 256.0)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
lr_scheduler = lrs.MultiStepLR(optimizer, milestones=[30 * le, 60 * le, 90 * le, 100 * le], gamma=0.1)
