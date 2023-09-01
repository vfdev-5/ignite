import datetime
from pathlib import Path
from typing import Any, Optional

import fire
import torch
import torchvision

from dataflow import Dataset, get_dataflow, get_dataloader
from mean_ap import CocoMetric, convert_to_coco_api
from torch.cuda.amp import autocast, GradScaler
from utils import FBResearchLogger, save_config

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, create_lr_scheduler_with_warmup, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed, setup_logger


def training(local_rank, config):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)

    output_path = config["output_path"]
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    output_path = Path(output_path) / "train" / folder_name
    config["output_path"] = output_path.as_posix()
    if rank == 0 and not output_path.exists():
        output_path.mkdir(parents=True)

    logger = setup_logger(name="VOCDetection-Training", filepath=output_path / "logs.txt")
    log_basic_info(logger, config)
    logger.info(f"Output path: {config['output_path']}")

    if rank == 0:
        if config["with_clearml"]:
            from clearml import Task

            task = Task.init("CIFAR10-Training", task_name=output_path.stem)
            task.connect_configuration(config)
            # Log hyper parameters
            hyper_params = [
                "model",
                "batch_size",
                "momentum",
                "weight_decay",
                "num_epochs",
                "learning_rate",
            ]
            task.connect({k: config[k] for k in hyper_params})
        else:
            save_config(config, output_path / "args.yaml")

    # Setup dataflow, model, optimizer, criterion
    train_loader, train_eval_loader, val_loader, num_classes = get_dataflow(config)

    config["num_classes"] = num_classes
    if config["epoch_length"] is None:
        config["epoch_length"] = len(train_loader)
    model, optimizer, lr_scheduler = initialize(config)

    trainer = create_trainer(model, optimizer, lr_scheduler, train_loader, config, logger)

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_evaluator(
        model, metrics={"mAP": CocoMetric(convert_to_coco_api(val_loader.dataset))}, config=config
    )
    FBResearchLogger(logger).attach(evaluator, "Test", every=config["log_every_iters"])
    train_evaluator = create_evaluator(
        model, metrics={"mAP": CocoMetric(convert_to_coco_api(train_eval_loader.dataset))}, config=config
    )
    FBResearchLogger(logger).attach(train_evaluator, "Train Eval", every=config["log_every_iters"])

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_eval_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED, run_validation)

    if config["start_by_validation"]:
        trainer.add_event_handler(Events.STARTED, run_validation)

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        evaluators = {"training": train_evaluator, "test": evaluator}
        tb_logger = common.setup_tb_logging(output_path, trainer, optimizer, evaluators=evaluators)

        from vis import draw_predictions

        train_evaluator.add_event_handler(
            Events.ITERATION_COMPLETED(every=len(val_loader) // 3), draw_predictions, tb_logger, "Train Eval"
        )

        evaluator.add_event_handler(
            Events.ITERATION_COMPLETED(every=len(val_loader) // 3), draw_predictions, tb_logger, "Validation"
        )

    # Store 2 best models by validation accuracy starting from num_epochs / 2:
    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="test_mAP",
        score_function=Checkpoint.get_default_score_fn("mAP"),
    )
    evaluator.add_event_handler(
        Events.COMPLETED(lambda *_: trainer.state.epoch > config["num_epochs"] // 3), best_model_handler
    )

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"], epoch_length=config["epoch_length"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


def initialize(config):
    kwargs = {"trainable_backbone_layers": config["trainable_backbone_layers"]}
    if config["data_augs"] in ["fixedsize"]:
        kwargs["_skip_resize"] = True

    model = torchvision.models.get_model(
        config["model"],
        weights_backbone=config["weights_backbone"],
        num_classes=config["num_classes"] + 1,
        **kwargs,
    )
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model, sync_bn=config["sync_bn"])

    if config["with_torch_compile"]:
        model = torch.compile(model)

    opt_name = config["optim"]
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD and AdamW are supported.")
    optimizer = idist.auto_optim(optimizer)

    num_epochs = config["num_epochs"]
    le = config["epoch_length"]
    milestones = [int(num_epochs * 0.8 * le), int(num_epochs * 0.9 * le)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    lr_scheduler = create_lr_scheduler_with_warmup(
        lr_scheduler,
        warmup_start_value=1e-5,
        warmup_duration=min(1000, le - 1),
    )

    return model, optimizer, lr_scheduler


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"Epoch[{epoch}] Evaluation complete. Time taken: {datetime.timedelta(seconds=int(elapsed))}\n - {tag} metrics:\n {metrics_output}"
    )


def log_basic_info(logger, config):
    logger.info(f"Train {config['model']} on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Torchvision version: {torchvision.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def create_trainer(model, optimizer, lr_scheduler, train_loader, config, logger):
    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    with_amp = config["with_amp"]
    scaler = GradScaler(enabled=with_amp)

    grad_accumulation_steps = config["grad_accumulation_steps"]

    def train_step(engine, batch):
        images, targets = batch[0], batch[1]

        if isinstance(images, torch.Tensor):
            images = images.to(device, non_blocking=True)
        else:
            images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        model.train()

        if (engine.state.iteration - 1) % grad_accumulation_steps == 0:
            optimizer.zero_grad()

        with autocast(enabled=with_amp):
            loss_dict = model(images, targets)
            losses = sum(loss / grad_accumulation_steps for loss in loss_dict.values())

        scaler.scale(losses).backward()
        if engine.state.iteration % grad_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        output = {
            "batch_loss": losses.item() * grad_accumulation_steps,
        }
        output.update({k: v.item() for k, v in loss_dict.items()})
        return output

    trainer = Engine(train_step)

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    if with_amp:
        to_save["amp_scaler"] = scaler

    metric_names = ["batch_loss"]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_loader.sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
    )

    FBResearchLogger(logger, show_output=True).attach(
        trainer, name="Train", every=config["log_every_iters"], optimizer=optimizer
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, metrics, config, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, batch):
        model.eval()
        images, targets = batch[0], batch[1]

        if isinstance(images, torch.Tensor):
            images = images.to(device, non_blocking=True)
        else:
            images = list(image.to(device) for image in images)

        with autocast(enabled=with_amp):
            output = model(images)
        return output, targets

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def get_save_handler(config):
    if config["with_clearml"]:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_path"])

    return DiskSaver(config["output_path"], require_empty=False)


def main_train(
    seed: int = 543,
    data_path: str = "/data",  # path to VOCdevkit, e.g. /data/VOCdevkit -> /data
    output_path: str = "/tmp/output-voc-detection",
    model: str = "retinanet_resnet50_fpn",
    weights_backbone: Optional[str] = None,  # backbone weights enum name to load, e.g. ResNet50_Weights.IMAGENET1K_V1
    sync_bn: bool = False,
    num_epochs: int = 26,
    optim: str = "adamw",
    learning_rate: float = 0.0001,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    batch_size: int = 4,  # total batch size for training, nb images per gpu is batch_size / nb_gpus
    eval_batch_size: Optional[
        int
    ] = None,  # total batch size for evaluation, nb images per gpu is batch_size / nb_gpus. By default, eval_batch_size = 2 * batch_size
    grad_accumulation_steps: int = 1,  # nb of gradients accumulation steps
    num_workers: int = 12,
    epoch_length: Optional[int] = None,
    trainable_backbone_layers: Optional[int] = None,
    data_augs: str = "hflip",  # object detection data aug presents: hflip and fixedsize
    validate_every: int = 3,
    start_by_validation: bool = False,
    checkpoint_every: int = 1000,
    backend: Optional[str] = None,
    resume_from: Optional[str] = None,
    log_every_iters: int = 50,
    nproc_per_node: Optional[int] = None,
    with_clearml: bool = False,
    with_amp: bool = True,
    with_torch_compile: bool = False,
    **spawn_kwargs: Any,
):
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)


def evaluation(local_rank, config):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)

    output_path = config["output_path"]
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    output_path = Path(output_path) / "eval" / folder_name
    config["output_path"] = output_path.as_posix()
    if rank == 0 and not output_path.exists():
        output_path.mkdir(parents=True)

    logger = setup_logger(name="VOCDetection-Evaluation", filepath=output_path / "logs.txt")
    log_basic_info(logger, config)
    logger.info(f"Output path: {config['output_path']}")

    if rank == 0:
        save_config(config, output_path / "args.yaml")

    # Setup dataflow, model, optimizer, criterion
    val_loader = get_dataloader("eval", config)

    config["num_classes"] = len(Dataset.classes)
    kwargs = {}
    if config["data_augs"] in ["fixedsize"]:
        kwargs["_skip_resize"] = True

    model = torchvision.models.get_model(
        config["model"],
        num_classes=config["num_classes"] + 1,
        **kwargs,
    )
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model, sync_bn=config["sync_bn"])

    weights_path = Path(config["weights_path"])
    assert weights_path.exists(), f"Weights path '{weights_path.as_posix()}' is not found"
    logger.info(f"Load model weights from file: {weights_path.as_posix()}")
    checkpoint = torch.load(weights_path.as_posix(), map_location="cpu")
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)

    evaluator = create_evaluator(
        model, metrics={"mAP": CocoMetric(convert_to_coco_api(val_loader.dataset))}, config=config
    )
    FBResearchLogger(logger).attach(evaluator, "Test", every=config["log_every_iters"])

    try:
        state = evaluator.run(val_loader)
    except Exception as e:
        logger.exception("")
        raise e

    log_metrics(logger, 0, state.times["COMPLETED"], "Test", state.metrics)


def main_evaluate(
    weights_path: str,
    data_path: str = "/data",  # path to VOCdevkit, e.g. /data/VOCdevkit -> /data
    output_path: str = "/tmp/output-voc-detection",
    model: str = "retinanet_resnet50_fpn",
    sync_bn: bool = False,
    eval_batch_size: int = 4,  # total batch size, nb images per gpu is batch_size / nb_gpus
    num_workers: int = 4,
    backend: Optional[str] = None,
    nproc_per_node: Optional[int] = None,
    with_amp: bool = True,
    **spawn_kwargs: Any,
):
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(evaluation, config)


def download_dataset(path):
    from dataflow import Dataset

    _ = Dataset(path, image_set="train", download=True, transforms=None)
    _ = Dataset(path, image_set="val", download=True, transforms=None)


if __name__ == "__main__":
    fire.Fire(
        {
            "train": main_train,
            "eval": main_evaluate,
            "download": download_dataset,
        }
    )
