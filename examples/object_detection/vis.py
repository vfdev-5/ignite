from typing import Dict, List, Optional

import torch

from torchvision.utils import draw_bounding_boxes


def draw_predictions_targets_on_image(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    label_to_name: Dict[int, str],
    score_threshold: Optional[float] = None,
) -> List[torch.Tensor]:
    image = image.cpu()

    if score_threshold is not None:
        m = pred["scores"] > score_threshold
        pred["boxes"] = pred["boxes"][m, :]
        pred["labels"] = pred["labels"][m]
        pred["scores"] = pred["scores"][m]

    pred["boxes"] = pred["boxes"].to(dtype=torch.long, device="cpu")
    pred["labels"] = pred["labels"].to(dtype=torch.long, device="cpu")
    pred["scores"] = pred["scores"].cpu()
    target["boxes"] = target["boxes"].to(dtype=torch.long, device="cpu")

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    pred_labels = [
        f"{label_to_name[label.item()]}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])
    ]
    image = draw_bounding_boxes(image, pred["boxes"], pred_labels, colors="red")

    target_labels = [label_to_name[label.item()] for label in target["labels"]]
    return draw_bounding_boxes(image, target["boxes"], target_labels, colors="green")


def draw_predictions(engine, tb_logger, label_to_name, tag="", score_threshold=0.5):
    output = engine.state.output
    batch = engine.state.batch
    iteration = engine.state.iteration

    for i, (image, target, pred) in enumerate(zip(batch[0], batch[1], output[0])):
        out_image = draw_predictions_targets_on_image(
            image, target, pred, label_to_name=label_to_name, score_threshold=score_threshold
        )
        tb_logger.writer.add_image(tag, out_image, global_step=iteration + i)
