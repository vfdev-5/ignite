import torch

from torchvision.utils import draw_bounding_boxes

from dataflow import Dataset


def draw_predictions(engine, tb_logger):
    output = engine.state.output
    batch = engine.state.batch
    for image, target, pred in zip(batch[0], output[1], output[0]):
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        pred_labels = [Dataset.label_to_name[label.item()] for label in pred["labels"]]
        pred_boxes = pred["boxes"].long()
        image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

        target_labels = [Dataset.label_to_name[label.item()] for label in target["labels"]]
        target_boxes = target["boxes"].long()
        image = draw_bounding_boxes(image, target_boxes, target_labels, colors="green")

    tb_logger.writer.