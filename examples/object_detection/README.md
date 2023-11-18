# Object Detection Example with PyTorch-Ignite

In this example, we show how to use _Ignite_ to train an object detection model

- Using VOC2012 or Coco128 datasets
- Validate with COCO mAP
- Visualization
- TensorBoard / ClearML as experiment tracking systems

## Requirements:

- pytorch-ignite: `pip install pytorch-ignite`
- [torchvision](https://github.com/pytorch/vision): `pip install torchvision`
- [albumentations](https://github.com/albumentations-team/albumentations): `pip install albumentations`
- [pycocotools](https://cocodataset.org/): `pip install pycocotools`
- [python-fire](https://github.com/google/python-fire): `pip install fire`
- (Optional)[tensorboard](https://www.tensorflow.org/tensorboard): `pip install tensorboard`
- (Optional)[clearml](https://clear.ml/): `pip install clearml`
- (Optional)[ultralytics/yolov8](https://docs.ultralytics.com/): `pip install ultralytics`

Alternatively, install the all requirements using `pip install -r requirements.txt`.

## Usage:

### Download Pascal VOC dataset

```bash
python main.py download_voc --path=/path/to/dataset/folder
# for example
# python main.py download --path=/data
```

### Download Coco128 dataset

```bash
python main.py download_coco128 --path=/path/to/dataset/folder
# for example
# python main.py download --path=/data/coco128
```

### Single GPU training

We will train RetinaNet with pretrained Resnet50 backbone and FPN (`retinanet_resnet50_fpn_v2`)

- Pascal VOC dataset:
```bash
python main.py train --dataset=voc --data_path=/path/to/folder_with_vocdevkit --output_path=/path/to/output_folder  --batch_size=2 --model=retinanet_resnet50_fpn_v2
# for example, we have /data/VOCdevkit
# python main.py train --dataset=voc --data_path=/data --output_path=/output --model=retinanet_resnet50_fpn
```

```bash
python main.py train --dataset=voc --data_path=/path/to/folder_with_vocdevkit --output_path=/path/to/output_folder --batch_size=2 --model=yolov8n-coco
# for example, we have /data/VOCdevkit
# python main.py train --dataset=voc --data_path=/data --output_path=/output --model=yolov8n-coco
```


- Coco128 dataset:
```bash
python main.py train --dataset=coco128 --data_path=/path/to/coco128 --output_path=/path/to/output_folder --batch_size=8 --model=retinanet_resnet50_fpn_v2
# for example, we have /data/VOCdevkit
# python main.py train --dataset=coco128 --data_path=/data --output_path=/output --model=retinanet_resnet50_fpn
```

```bash
python main.py train --dataset=coco128 --data_path=/path/to/coco128 --output_path=/path/to/output_folder --batch_size=8 --model=yolov8n-coco
# for example, we have /data/coco128
# python main.py train --dataset=coco128 --batch_size=8 --data_path=/data/coco128 --output_path=/output --model=yolov8n-coco
```



### Multiple GPUs training using `torchrun`

We will again train RetinaNet with pretrained Resnet50 backbone and FPN (`retinanet_resnet50_fpn_v2`).
Let's say we have 4 GPUs and we pick total batch size as 16 (4 per device):

```bash
torchrun --nproc_per_node=4 main.py train --backend=nccl --batch_size=16 --data_path=/path/to/folder_with_vocdevkit --output_path=/path/to/output_folder --model=retinanet_resnet50_fpn_v2 --weights_backbone="ResNet50_Weights.IMAGENET1K_V1"
```

### Open TensorBoard


```bash
tensorboard --logdir=/path/to/output_folder
```