# python dataflow/test_dataflow.py coco128 /data/coco128
# python dataflow/test_dataflow.py voc /data/ --data_augs=hflip

import sys
from pathlib import Path
from typing import Optional

import fire

import torch

from ignite.utils import manual_seed

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from dataflow import get_dataflow


def test_dataloader(dataloader, n=50):
    manual_seed(123)

    dataloader_iter = iter(dataloader)

    for j in range(n):
        try:
            imgs, targets = next(dataloader_iter)
        except Exception as e:
            print(f"Exception in batch {j}")
            raise e

        assert isinstance(imgs, list) and len(imgs) == dataloader.batch_size, (j, imgs)
        assert isinstance(targets, list) and len(targets) == dataloader.batch_size, (j, targets)

        for i, (img, target) in enumerate(zip(imgs, targets)):
            assert img.shape[0] == 3 and img.shape[1] > 50 and img.shape[2] > 50, (j, i, img.shape)
            assert img.dtype == torch.float32, (j, i, img.shape)

            for key in ["boxes", "labels", "image_id"]:
                assert key in target, (j, i, target.keys())

            boxes = target["boxes"]
            assert len(boxes.shape) == 2 and boxes.shape[1] == 4, (j, i, boxes.shape, boxes.dtype)
            assert boxes.shape[0] >= 0, (j, i, boxes.shape, boxes.dtype)
            assert boxes.dtype == torch.float32, (j, i, boxes.shape, boxes.dtype)

            labels = target["labels"]
            assert len(labels.shape) == 1 and labels.shape[0] >= 0, (j, i, labels.shape, labels.dtype)
            assert labels.dtype == torch.long, (j, i, labels.shape, labels.dtype)

            image_id = target["image_id"]
            assert isinstance(image_id, str), (j, i, image_id)


def main(
    dataset_name: str,
    data_path: str,
    data_augs: Optional[str] = None,
):
    data_path = Path(data_path)
    assert data_path.exists()

    assert dataset_name in ("voc", "coco128")

    config = {
        "dataset": dataset_name,
        "data_path": data_path,
        "data_augs": data_augs,
        "num_workers": 0,
        "batch_size": 8,
        "eval_batch_size": 8,
    }
    train_loader, train_eval_loader, val_loader, num_classes = get_dataflow(config)

    if dataset_name == "voc":
        expected_num_classes = 20
        n = 100
    elif dataset_name == "coco128":
        expected_num_classes = 80
        n = 5
    else:
        expected_num_classes = None
        n = None

    assert num_classes == expected_num_classes

    test_dataloader(train_loader, n=n)
    test_dataloader(train_eval_loader, n=n)
    test_dataloader(val_loader, n=n)


if __name__ == "__main__":
    fire.Fire(main)
