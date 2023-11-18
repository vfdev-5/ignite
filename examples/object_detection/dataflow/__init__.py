from dataflow import coco128, voc


def get_dataflow(config):
    dataset_name = config["dataset"]
    if dataset_name == "voc":
        return voc.get_dataflow(config)
    elif dataset_name == "coco128":
        return coco128.get_dataflow(config)
    raise ValueError(f"Unknown dataset name '{dataset_name}'")


def get_dataloader(config):
    dataset_name = config["dataset"]
    if dataset_name == "voc":
        return voc.get_dataloader(config)
    elif dataset_name == "coco128":
        return coco128.get_dataloader(config)
    raise ValueError(f"Unknown dataset name '{dataset_name}'")
