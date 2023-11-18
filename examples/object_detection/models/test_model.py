# python models/test_model.py retinanet_resnet50_fpn voc /data/ --data_augs=hflip
# python models/test_model.py retinanet_resnet50_fpn coco128 /data/coco128 --data_augs=hflip
# python models/test_model.py yolov8n coco128 /data/coco128
# python models/test_model.py yolov8n-coco coco128 /data/coco128

import sys
from pathlib import Path
from typing import Optional

import fire

import torch

from ignite.utils import manual_seed


sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from dataflow import get_dataflow

from models import get_model


def test_model(model, dataloader, n=3, mode="train"):
    assert mode in ("train", "eval")

    manual_seed(123)

    dataloader_iter = iter(dataloader)

    if mode == "train":
        model.train()
    else:
        model.eval()

    for j in range(n):
        try:
            images, targets = next(dataloader_iter)
        except Exception as e:
            print(f"Exception in batch {j}")
            raise e

        if mode == "train":
            loss_dict = model(images, targets)
            assert isinstance(loss_dict, dict)
        else:
            assert isinstance(images, list)
            output = model(images)
            assert isinstance(output, list) and len(output) == len(images)
            for pred in output:
                assert isinstance(pred, dict):
                assert "boxes" in pred and isinstance(pred["boxes"], torch.Tensor)
                assert "scores" in pred and isinstance(pred["scores"], torch.Tensor)
                assert "labels" in pred and isinstance(pred["labels"], torch.Tensor)
                # else:
                #     assert isinstance(pred, torch.Tensor), type(pred)
                #     assert pred.ndim == 2 and pred.shape[1] == 6


def main(
    model_name: str,
    dataset_name: str,
    data_path: str,
    data_augs: Optional[str] = None,
):
    data_path = Path(data_path)
    assert data_path.exists()

    config = {
        "dataset": dataset_name,
        "data_path": data_path,
        "data_augs": data_augs,
        "num_workers": 0,
        "batch_size": 8,
        "eval_batch_size": 8,
        "model": model_name,
        "use_coco_weights": True,
        "weights_backbone": None,
        "trainable_backbone_layers": None,
    }

    train_loader, train_eval_loader, val_loader, num_classes = get_dataflow(config)
    config["num_classes"] = num_classes
    model = get_model(config)

    test_model(model, train_loader, n=2, mode="train")
    test_model(model, train_eval_loader, n=2, mode="eval")
    test_model(model, val_loader, n=2, mode="eval")


if __name__ == "__main__":
    fire.Fire(main)
