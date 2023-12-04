"""
Evaluation code is mainly taken credit from torchvision coco_eval.
reference: https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py
"""
import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import ignite.distributed as idist
from ignite.metrics import Metric


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


class CocoEvaluator:
    def __init__(self, coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.coco_eval = COCOeval(coco_gt, iouType="bbox")
        self.reset()

    def reset(self):
        self.img_ids = []
        self.eval_imgs = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        results = self.prepare(predictions)
        with redirect_stdout(io.StringIO()):
            coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
        self.coco_eval.cocoDt = coco_dt
        self.coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(self.coco_eval)
        self.eval_imgs.append(eval_imgs)

    def synchronize_between_processes(self):
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        create_common_coco_eval(self.coco_eval, self.img_ids, self.eval_imgs)

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self):
        self.coco_eval.summarize()

    def prepare(self, predictions):
        return self.prepare_for_coco_detection(predictions)

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = idist.all_gather(img_ids)
    all_eval_imgs = idist.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


class CocoMetric(Metric):
    def __init__(self, coco_api, *args, **kwargs):
        self.coco_evaluator = CocoEvaluator(coco_api)
        super().__init__(*args, **kwargs)

    def update(self, output):
        y_pred, y = output[0], output[1]
        res = {target["image_id"]: output for target, output in zip(y, y_pred)}
        self.coco_evaluator.update(res)

    def reset(self):
        self.coco_evaluator.reset()

    @torch.no_grad()
    def iteration_completed(self, engine) -> None:
        self.update(engine.state.output)

    def compute(self):
        # sync between processes
        self.coco_evaluator.synchronize_between_processes()

        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        return self.coco_evaluator.coco_eval.stats[0]


def test_metric(lrank, data_path, max_batches=None, expected_mean_ap=None):
    from dataflow.voc import default_od_collate_fn, get_test_transform, VOCDataset
    from torch.utils.data import Subset

    config = {
        "data_path": data_path,
        "data_augs": "hflip",
        "batch_size": 8,
        "num_workers": 4,
    }

    val_transform = get_test_transform(config)
    test_dataset = VOCDataset(config["data_path"], image_set="val", download=False, transforms=val_transform)
    test_dataset = Subset(test_dataset, indices=range(max_batches * config["batch_size"]))
    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        collate_fn=default_od_collate_fn,
    )

    metric = CocoMetric(convert_to_coco_api(test_dataset))

    if max_batches is None:
        max_batches = len(test_loader)

    ws = idist.get_world_size()
    r = idist.get_rank()

    def cond(j):
        return (ws * j + r) % 3 == 0

    for i, batch in enumerate(test_loader):
        if i >= max_batches:
            break

        # y_true is a list of dicts with keys: ['boxes', 'labels', 'image_id', 'area', 'iscrowd']
        y_true = batch[1]
        # y_pred is a list of dicts with keys: 'boxes', 'scores', 'labels'

        # For each process we set up the predictions as the GT with a certain rule
        # We need a global index = ws * data_index + rank
        # Single proc:
        # bs = 8
        # proc 0:
        #   y_true: [y0, y1, y2, y3, y4, y5, y6, y7]
        #
        #   y_pred: [y0, 0, 0, y3, 0, 0, y6, 0]

        # Two procs:
        # Data is distributed as [a, b, c, d] -> [a, c], [b, d]
        # bs = 4 per proc
        # proc 0:
        #   y_true: [y0, y2, y4, y6]
        #   y_pred: [y0, 0,  0, y6]
        # proc 1:
        #   y_true: [y1, y3, y5, y7]
        #   y_pred: [0,  y3,  0,  0]

        y_pred = [
            {
                "boxes": y["boxes"]
                if cond(j)
                else torch.tensor([], device=y["boxes"].device, dtype=y["boxes"].dtype).reshape(0, 4),
                "scores": 0.78 * torch.ones_like(y["labels"], dtype=torch.float32)
                if cond(j)
                else torch.tensor([], device=y["labels"].device, dtype=torch.float32),
                "labels": y["labels"]
                if cond(j)
                else torch.tensor([], device=y["labels"].device, dtype=y["labels"].dtype),
            }
            for j, y in enumerate(y_true)
        ]
        metric.update((y_pred, y_true))

    value = metric.compute()
    import time

    time.sleep(0.1 * lrank)
    print(lrank, "mAP:", value)
    if expected_mean_ap is not None:
        assert value == expected_mean_ap

    return value


if __name__ == "__main__":
    # Metric tests in DDP config
    import sys
    from pathlib import Path

    assert len(sys.argv) == 2, "Usage: python mean_ap.py /data"
    data_path = Path(sys.argv[1])
    assert data_path.exists()

    print("Expected (single process)")
    expected_mean_ap = test_metric(0, data_path, 10)

    print("Ouput (two processes)")
    idist.spawn("gloo", test_metric, (data_path, 10, expected_mean_ap), nproc_per_node=2)
