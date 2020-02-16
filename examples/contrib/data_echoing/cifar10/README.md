# Neural Network Training with Data Echoing on CIFAR10

In this example, we explore "data echoing" from https://arxiv.org/abs/1907.05550 which, according to the paper, 
reduces the total computation used by earlier pipeline stages and speeds up training.

## Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`
- [tensorboardx](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`

## Usage:

### Example without any data echoing

Run the example on a single GPU (script will not run without a GPU):
```bash
python main.py
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as
```bash
--params="data_path=/path/to/cifar10/;..."
```

### Example echoing before data augmentation

```bash
python main.py --params="with_example_echoing=True"
```

### Example echoing after data augmentation

```bash
python main.py --params="with_example_echoing=True;echoing_before_dataaug=False"
```
