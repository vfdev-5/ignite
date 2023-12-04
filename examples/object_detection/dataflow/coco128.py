
import torch
from torch.utils.data import Dataset, Subset

import ignite.distributed as idist

try:
    from ultralytics.data import YOLODataset

    has_ultralytics = True
except ModuleNotFoundError:
    has_ultralytics = False


class COCO128Dataset(Dataset):
    """Proxy dataset over ultralytics YOLODataset
    - getitem returns a tuple of (image, target)
    """

    label_to_name = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }

    classes = list(label_to_name.values())
    name_to_label = {v: k for k, v in enumerate(classes)}

    def __init__(self, data_path, mode, buffer_size=64, download=False, return_yolo_dict=False):
        if not has_ultralytics:
            raise RuntimeError("To use Yolo models, please install ultralytics:\n\tpip install ultralytics")

        assert mode in ("train", "val")

        self.return_yolo_dict = return_yolo_dict

        data = {
            "train": "images/train2017",
            "val": "images/train2017",
            "test": None,
            "nc": len(COCO128Dataset.classes),
            "names": COCO128Dataset.classes,
        }

        kwargs = dict(
            imgsz=640,
            batch_size=buffer_size,
            prefix="",
            data=data,
        )
        if mode == "train":
            kwargs.update(dict(augment=True, pad=0.0))
        elif mode == "val":
            kwargs.update(dict(augment=False, pad=0.5, fraction=1.0))

        self.dataset = YOLODataset(data_path, **kwargs)

    @staticmethod
    def collate_fn(list_datapoints):
        batch = [d for d, _ in list_datapoints]
        nones = [n for _, n in list_datapoints]
        return YOLODataset.collate_fn(batch), nones

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        datapoint = self.dataset[index]
        datapoint["img"] = datapoint["img"].float() / 255

        if self.return_yolo_dict:
            # returns a structure compatible to (images, targets)
            return datapoint, 0

        image = datapoint["img"]
        boxes_ncxcywh = datapoint["bboxes"]  # tensor in normalized cxcywh
        size = datapoint["resized_shape"]
        boxes_cxywh = boxes_ncxcywh * torch.tensor([*size, *size])
        boxes_xyxy = boxes_cxywh.clone()
        boxes_xyxy[..., :2] -= 0.5 * boxes_xyxy[..., 2:]
        boxes_xyxy[..., 2:] += boxes_xyxy[..., :2]

        target = {
            "boxes": boxes_xyxy,  # [N, 4]
            "labels": datapoint["cls"].view(-1).long(),  # [N, ]
            "image_id": datapoint["im_file"],
            "iscrowd": torch.tensor([False] * len(datapoint["bboxes"])),
            "area": (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]),
        }
        return image, target


def default_od_collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(mode, config, train_eval_size=None, return_yolo_dict=False):
    assert mode in ["train", "train_eval", "eval"]

    if mode in ["eval", "train_eval"]:
        mode = "val" if mode == "eval" else "train"

    dataset = COCO128Dataset(
        config["data_path"],
        mode=mode,
        buffer_size=config["batch_size"],
        return_yolo_dict=return_yolo_dict,
    )

    if mode == "train_eval" and train_eval_size is not None:
        g = torch.Generator().manual_seed(len(dataset))
        train_eval_indices = torch.randperm(len(dataset), generator=g)[:train_eval_size]
        dataset = Subset(dataset, train_eval_indices)

    collate_fn = dataset.collate_fn if return_yolo_dict else default_od_collate_fn

    if config["eval_batch_size"] is None:
        config["eval_batch_size"] = 2 * config["batch_size"]

    data_loader = idist.auto_dataloader(
        dataset,
        batch_size=config["batch_size"] if mode == "train" else config["eval_batch_size"],
        num_workers=config["num_workers"],
        shuffle=True if mode == "train" else False,
        drop_last=True if mode == "train" else False,
        collate_fn=collate_fn,
    )

    return data_loader, len(dataset.classes)


def get_dataflow(config):
    if not has_ultralytics:
        raise RuntimeError("To use Yolo models, please install ultralytics:\n\tpip install ultralytics")

    return_yolo_dict = "yolo" in config.get("model", "")

    train_loader, _ = get_dataloader("train", config, return_yolo_dict=return_yolo_dict)
    val_loader, _ = get_dataloader("eval", config, return_yolo_dict=False)
    train_eval_loader, _ = get_dataloader("train_eval", config, train_eval_size=len(val_loader), return_yolo_dict=False)

    num_classes = len(COCO128Dataset.classes)

    return train_loader, train_eval_loader, val_loader, num_classes
