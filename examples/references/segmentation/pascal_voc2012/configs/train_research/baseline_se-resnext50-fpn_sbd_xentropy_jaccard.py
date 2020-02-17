# Basic training configuration
import os
from functools import partial
from itertools import chain

import cv2
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

from segmentation_models_pytorch import FPN

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import ignore_mask_boundaries, prepare_batch_fp32, denormalize

from models import SameSizeOutputModelWrapper

from losses import SumOfLosses
from losses.jaccard import SoftmaxJaccardWithLogitsLoss

assert 'DATASET_PATH' in os.environ
data_path = os.environ['DATASET_PATH']

assert 'SBD_DATASET_PATH' in os.environ
sbd_data_path = os.environ['SBD_DATASET_PATH']


debug = False
seed = 12

device = 'cuda'
fp16_opt_level = "O2"

num_classes = 21


batch_size = 24 // dist.get_world_size()
val_batch_size = 24
non_blocking = True
num_workers = 12 // dist.get_world_size()
val_interval = 1
start_by_validation = True
accumulation_steps = 2

val_img_size = 513
train_img_size = 480

# ##############################
# Setup Dataflow
# ##############################

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


train_transforms = A.Compose([
    A.RandomScale(scale_limit=(0.0, 1.5), interpolation=cv2.INTER_LINEAR, p=1.0),
    A.PadIfNeeded(val_img_size, val_img_size, border_mode=cv2.BORDER_CONSTANT),
    A.RandomCrop(train_img_size, train_img_size),

    A.HorizontalFlip(),
    A.Blur(blur_limit=3),

    A.CoarseDropout(max_height=64, max_width=64),

    A.OneOf([
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
    ]),

    A.Normalize(mean=mean, std=std),
    ignore_mask_boundaries,
    ToTensor(),
])

val_transforms = A.Compose([
    A.PadIfNeeded(val_img_size, val_img_size, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=mean, std=std),
    ignore_mask_boundaries,
    ToTensor(),
])


train_loader, val_loader, train_eval_loader = get_train_val_loaders(root_path=data_path,
                                                                    train_transforms=train_transforms,
                                                                    val_transforms=val_transforms,
                                                                    batch_size=batch_size,
                                                                    num_workers=num_workers,
                                                                    val_batch_size=val_batch_size,
                                                                    with_sbd=sbd_data_path,
                                                                    train_sampler='distributed',
                                                                    limit_train_num_samples=100 if debug else None,
                                                                    limit_val_num_samples=100 if debug else None)

prepare_batch = prepare_batch_fp32

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# ##############################
# Setup models
# ##############################

model = FPN(encoder_name='se_resnext50_32x4d', classes=num_classes)
model = SameSizeOutputModelWrapper(model)


# ##############################
# Setup solver
# ##############################

num_epochs = 100

xentropy = nn.CrossEntropyLoss()
jaccard = SoftmaxJaccardWithLogitsLoss()

names = ['cross entropy loss', 'jaccard loss']

criterion = SumOfLosses(losses=[xentropy, jaccard], coeffs=[1.0, 2.0],
                        names=names,
                        total_loss_name="supervised batch loss")

output_names = ["supervised batch loss", ] + names


lr = 0.007
weight_decay = 5e-4
momentum = 0.9
nesterov = False

optimizer = optim.SGD([{'params': model.model.encoder.parameters()},
                       {'params': chain(model.model.decoder.parameters(), model.model.segmentation_head.parameters())}],
                      lr=1.0, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)


le = len(train_loader)


def lambda_lr_scheduler(iteration, lr0, n, a):
    return lr0 * pow((1.0 - 1.0 * iteration / n), a)


lr_scheduler = lrs.LambdaLR(optimizer,
                            lr_lambda=[
                                partial(lambda_lr_scheduler, lr0=lr, n=num_epochs * le, a=0.9),
                                partial(lambda_lr_scheduler, lr0=lr * 10.0, n=num_epochs * le, a=0.9)
                            ])
