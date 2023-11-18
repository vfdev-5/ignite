import warnings

import torchvision

from models.yolov8 import get_ultralytics_yolo_model


def get_model(config):
    model_name = config["model"]
    if model_name.startswith("yolov8"):
        if config["weights_backbone"] == "auto":
            config["weights_backbone"] = None
        assert config["weights_backbone"] is None, "Can't specify weights_backbone for Yolo models"
        return get_ultralytics_yolo_model(model_name, config)
    else:
        return get_torchvision_model(model_name, config)


def get_torchvision_model(model_name, config):
    num_classes = config["num_classes"] + 1
    skip_resize = config["data_augs"] in ["fixedsize"]

    if config["weights_backbone"] == "auto":
        if "resnet50" in model_name:
            config["weights_backbone"] = "ResNet50_Weights.IMAGENET1K_V1"
        else:
            warnings.warn(f"Failed to auto provide pretrained backbone for model: '{model_name}'")
            config["weights_backbone"] = None

    kwargs = {"trainable_backbone_layers": None}
    if skip_resize:
        kwargs["_skip_resize"] = True

    model = torchvision.models.get_model(
        model_name,
        weights_backbone=config["weights_backbone"],
        num_classes=num_classes,
        **kwargs,
    )
    return model
