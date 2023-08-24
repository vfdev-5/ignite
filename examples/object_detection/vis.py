from typing import Any, Dict, List

import torch

from dataflow import Dataset

from torchvision.utils import draw_bounding_boxes


def draw_predictions_targets_on_image(
    image: torch.Tensor, target: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]
) -> List[torch.Tensor]:
    if image.device.type == "cuda":
        image = image.cpu()
    if pred["boxes"].device.type == "cuda":
        pred["boxes"] = pred["boxes"].cpu()
    if target["boxes"].device.type == "cuda":
        target["boxes"] = target["boxes"].cpu()

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    pred_labels = [Dataset.label_to_name[label.item()] for label in pred["labels"]]
    pred_boxes = pred["boxes"].long()
    image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    target_labels = [Dataset.label_to_name[label.item()] for label in target["labels"]]
    target_boxes = target["boxes"].long()
    return draw_bounding_boxes(image, target_boxes, target_labels, colors="green")


def draw_predictions(engine, tb_logger, tag=""):
    output = engine.state.output
    batch = engine.state.batch
    iteration = engine.state.iteration

    for i, (image, target, pred) in enumerate(zip(batch[0], batch[1], output[0])):
        out_image = draw_predictions_targets_on_image(image, target, pred)
        tb_logger.writer.add_image(tag, out_image, global_step=iteration + i)
