# Benchmark mixed precision training on Cifar100

In this folder we provide scripts to benchmark 1) native PyTorch mixed precision module `torch.cuda.amp`, 2) NVidia/Apex package and 
3) [Microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed/) optimization library.

We will train Wide-ResNet model on Cifar100 dataset using Turing enabled GPU and compare training times.

## Requirements

```bash
pip install pytorch-ignite fire
```

## Download dataset

```bash
python -c "from torchvision.datasets.cifar import CIFAR100; CIFAR100(root='/tmp/cifar100/', train=True, download=True)"
```

## FP32 training

```bash
python benchmark_fp32.py /tmp/cifar100/ --batch_size=256 --max_epochs=20
```


## PyTorch native AMP

Recently added `torch.cuda.amp` module to perform automatic mixed precision training instead of using Nvidia/Apex package 
is available in PyTorch >=1.6.0.

```bash
python benchmark_torch_cuda_amp.py /tmp/cifar100/ --batch_size=256 --max_epochs=20
```

## NVidia/Apex AMP

We check 2 optimization levels: "O1" and "O2"

- "O1" optimization level: automatic casts arount Pytorch functions and tensor methods
- "O2" optimization level: fp16 training with fp32 batchnorm and fp32 master weights

```bash
python benchmark_nvidia_apex.py /tmp/cifar100/ --batch_size=256 --max_epochs=20 --opt="O1"
```

and 

```bash
python benchmark_nvidia_apex.py /tmp/cifar100/ --batch_size=256 --max_epochs=20 --opt="O2"
```


## Microsoft/DeepSpeed

DeepSpeed package uses internally NVidia/Apex for AMP and provides other features like "Zero Redundancy Optimizer". 
More details about it can be found at https://www.deepspeed.ai/ .

### Requirements

To install DeepSpeed, please follow its official [installation guide](https://www.deepspeed.ai/getting-started/#installation).

### Run 

```bash
deepspeed benchmark_deepspeed.py /tmp/cifar100/ --batch_size=256 --max_epochs=20
```
