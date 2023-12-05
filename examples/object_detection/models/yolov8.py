import torch
import torch.nn as nn

try:
    from ultralytics.engine.trainer import DEFAULT_CFG, get_cfg
    from ultralytics.nn.tasks import DetectionModel, torch_safe_load
    from ultralytics.utils.ops import non_max_suppression

    has_ultralytics = True
except ModuleNotFoundError:
    has_ultralytics = False


class YoloDetectionModelWrapper(nn.Module):
    """Yolo DetectionModel wrapper to adapt input and output to
    match torchvision detection models interface:

    input: images and optional targets
    output: if training: dict of losses otherwise list of predictions
    """

    def __init__(self, yolo_detection_model) -> None:
        super().__init__()
        self.model = yolo_detection_model

    def forward(self, batch, _=None):
        print(type(batch), len(batch))
        if self.training:
            # loss is a sum 3 losses: box, cls, dfl
            loss, _ = self.model(batch)
            return {"total_loss": loss}
        else:
            assert isinstance(batch, list) and all([isinstance(i, torch.Tensor) for i in batch])
            images = torch.stack(batch, dim=0)
            output = self.model(images)
            output = non_max_suppression(
                output,
                self.model.args.conf,
                self.model.args.iou,
                labels=[],
                multi_label=True,
                agnostic=self.model.args.single_cls,
                max_det=self.model.args.max_det,
            )
            return [
                {
                    "boxes": pred[..., :4],
                    "scores": pred[..., 4],
                    "labels": pred[..., 5],
                }
                for pred in output
            ]


def get_ultralytics_yolo_model(model_name, config):
    if not has_ultralytics:
        raise RuntimeError("To use Yolo models, please install ultralytics:\n\tpip install ultralytics")

    assert any([f"yolov8{size}" in model_name for size in "nsmlx"])

    use_coco_weights = "-coco" in model_name
    if use_coco_weights:
        model_name = model_name.replace("-coco", "")

    num_classes = config["num_classes"]
    conf = config.get("confidence", 0.001)
    if config["use_pt_weights"]:
        from ultralytics import YOLO

        cfg = f"{model_name}.pt"
        model = YOLO(cfg).model
        yaml_file = model.yaml_file
    else:
        cfg = f"{model_name}.yaml"
        model = DetectionModel(cfg=cfg, nc=num_classes)
        yaml_file = model.yaml["yaml_file"]

    if use_coco_weights:
        print(f"Use MSCoco weights for {model_name} model")
        chkpt, _ = torch_safe_load(f"{model_name}.pt")
        coco_model = chkpt["model"]
        coco_model_state_dict = coco_model.state_dict()

        if coco_model_state_dict["model.22.cv3.2.2.weight"].shape[0] != num_classes:
            coco_model_state_dict = {k: v for k, v in coco_model_state_dict.items() if "model.22.cv3" not in k}

        model.load_state_dict(coco_model_state_dict, strict=False)

    # This is needed in order to v8DetectionLoss work
    overrides = {
        "model": yaml_file,
        "task": "detect",
        "conf": conf,
    }
    model.args = get_cfg(DEFAULT_CFG, overrides)
    return YoloDetectionModelWrapper(model)
