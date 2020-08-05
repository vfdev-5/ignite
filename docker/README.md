# Convenient docker images

In this folder we provide Dockerfiles to build docker images with PyTorch, Ignite and other convenient libraries:


## PyTorch-Ignite docker image with [Horovod distributed framework](https://horovod.readthedocs.io/en/latest/index.html)

- [hvd-base](hvd/Dockerfile.base) : latest stable PyTorch, latest stable Ignite and Horovod

- [hvd-vision](hvd/Dockerfile.vision) : hvd-base image, OpenCV, Albumentations
    

Note: 

- Horodov is built from master with NCCL GPU operations support and Gloo collective communications.


### How to pull images

Base image:

```bash
docker pull pytorchignite/hvd-base:latest
```

Vision image:

```bash
docker pull pytorchignite/hvd-vision:latest
```


### Manually build image

Base image:

```bash
cd hvd && docker build -t pytorchignite/hvd-base:latest -f Dockerfile.base .
```


