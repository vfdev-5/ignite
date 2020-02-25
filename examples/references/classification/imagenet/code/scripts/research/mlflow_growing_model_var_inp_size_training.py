# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method
from pathlib import Path

import torch
import torch.distributed as dist

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import mlflow

import ignite
from ignite.engine import Engine, Events, _prepare_batch, create_supervised_evaluator
from ignite.metrics import Accuracy, TopKCategoricalAccuracy

from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import common

from py_config_runner.utils import set_seed
from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config

from utils.handlers import predictions_gt_images_handler, DataflowBenchmark


def training(config, local_rank=None, with_mlflow_logging=False, with_plx_logging=False):

    if not getattr(config, "use_fp16", True):
        raise RuntimeError("This training script uses by default fp16 AMP")

    set_seed(config.seed + local_rank)
    torch.cuda.set_device(local_rank)
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    train_loader = config.train_loader
    train_sampler = getattr(train_loader, "sampler", None)

    train_eval_loader = config.train_eval_loader
    val_loader = config.val_loader

    original_model = config.model.to(device)
    print("Original model numel:", sum([p.numel() for p in original_model.parameters()]))
    assert hasattr(original_model, "grow")
    original_optimizer = config.optimizer

    def _setup_model_optimizer(original_model, original_optimizer):
        model, optimizer = amp.initialize(
            original_model, original_optimizer, opt_level=getattr(
                config, "fp16_opt_level", "O2"), num_losses=1)
        model = DDP(model, delay_allreduce=True)
        return model, optimizer

    model, optimizer = _setup_model_optimizer(original_model, original_optimizer)
    criterion = config.criterion.to(device)

    prepare_batch = getattr(config, "prepare_batch", _prepare_batch)
    non_blocking = getattr(config, "non_blocking", True)

    # Setup trainer
    accumulation_steps = getattr(config, "accumulation_steps", 1)
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    def train_update_function(engine, batch):

        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        y_pred = model_output_transform(y_pred)
        loss = criterion(y_pred, y) / accumulation_steps

        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return {"supervised batch loss": loss.item()}

    trainer = Engine(train_update_function)

    lr_scheduler = config.lr_scheduler
    to_save = {"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler, "trainer": trainer, "amp": amp}
    common.setup_common_training_handlers(
        trainer,
        train_sampler,
        to_save=to_save,
        save_every_iters=1000,
        output_path=config.output_path.as_posix(),
        lr_scheduler=lr_scheduler,
        with_gpu_stats=True,
        output_names=["supervised batch loss"],
        with_pbars=True,
        with_pbar_on_iters=with_mlflow_logging,
        log_every_iters=1,
    )

    if getattr(config, "benchmark_dataflow", False):
        benchmark_dataflow_num_iters = getattr(config, "benchmark_dataflow_num_iters", 1000)
        DataflowBenchmark(benchmark_dataflow_num_iters, prepare_batch=prepare_batch, device=device).attach(
            trainer, train_loader
        )

    # Schedule training image size
    assert hasattr(config, "train_transforms")
    assert hasattr(config, "resized_crop_aug")
    resize_aug = config.train_transforms.transforms[0]
    resized_crop_aug = config.resized_crop_aug

    assert hasattr(config, "max_train_crop_size")
    assert hasattr(resize_aug, "height")
    assert hasattr(resize_aug, "width")
    assert hasattr(resized_crop_aug, "height")
    assert hasattr(resized_crop_aug, "width")

    max_train_crop_size = getattr(config, "max_train_crop_size")
    switch_to_resized_crop = False

    @trainer.on(Events.EPOCH_COMPLETED)
    def schedule_training_image_size(_):
        nonlocal switch_to_resized_crop, resize_aug

        resize_aug.height = min(resize_aug.height + 10, max_train_crop_size)
        resize_aug.width = min(resize_aug.width + 10, max_train_crop_size)

        if resize_aug.width >= 128 and switch_to_resized_crop:
            # replace resize by resized crop
            resized_crop_aug.height = resize_aug.height
            resized_crop_aug.width = resize_aug.width
            config.train_transforms.transforms.transforms[0] = resized_crop_aug
            resize_aug = config.train_transforms.transforms[0]
            switch_to_resized_crop = False

    # Grow model
    max_grow = 10
    # @trainer.on(Events.EPOCH_STARTED(every=3))
    @trainer.on(Events.EPOCH_STARTED)
    def grow_model(_):
        nonlocal model, optimizer, max_grow
        if max_grow > 0:
            original_model.grow(device)
            original_optimizer.param_groups[0]['params'] = list(original_model.parameters())
            model, optimizer = _setup_model_optimizer(original_model, original_optimizer)
            print("Grown model numel:", sum([p.numel() for p in original_model.parameters()]))
            max_grow -= 1

    # Setup evaluators
    val_metrics = {"Accuracy": Accuracy(device=device), "Top-5 Accuracy": TopKCategoricalAccuracy(k=5, device=device)}

    if hasattr(config, "val_metrics") and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    evaluator_args = dict(
        model=model,
        metrics=val_metrics,
        device=device,
        non_blocking=non_blocking,
        prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred: (model_output_transform(y_pred), y),
    )
    train_evaluator = create_supervised_evaluator(**evaluator_args)
    evaluator = create_supervised_evaluator(**evaluator_args)

    if dist.get_rank() == 0 and with_mlflow_logging:
        ProgressBar(persist=False, desc="Train Evaluation").attach(train_evaluator)
        ProgressBar(persist=False, desc="Val Evaluation").attach(evaluator)

    def run_validation(_):
        train_evaluator.run(train_eval_loader)
        evaluator.run(val_loader)

    if getattr(config, "start_by_validation", False):
        trainer.add_event_handler(Events.STARTED, run_validation)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=getattr(config, "val_interval", 1)), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)

    score_metric_name = "Accuracy"

    if hasattr(config, "es_patience"):
        common.add_early_stopping_by_val_score(config.es_patience, evaluator, trainer, metric_name=score_metric_name)

    if dist.get_rank() == 0:

        tb_logger = common.setup_tb_logging(
            config.output_path.as_posix(),
            trainer,
            optimizer,
            evaluators={"training": train_evaluator, "validation": evaluator},
        )
        if with_mlflow_logging:
            common.setup_mlflow_logging(
                trainer, optimizer, evaluators={"training": train_evaluator, "validation": evaluator}
            )

        if with_plx_logging:
            common.setup_plx_logging(
                trainer, optimizer, evaluators={"training": train_evaluator, "validation": evaluator}
            )

        common.save_best_model_by_val_score(
            config.output_path.as_posix(), evaluator, model, metric_name=score_metric_name, trainer=trainer
        )

        # Log train/val predictions:
        tb_logger.attach(
            evaluator,
            log_handler=predictions_gt_images_handler(
                img_denormalize_fn=config.img_denormalize, n_images=15, another_engine=trainer, prefix_tag="validation"
            ),
            event_name=Events.ITERATION_COMPLETED(once=len(val_loader) // 2),
        )

        tb_logger.attach(
            train_evaluator,
            log_handler=predictions_gt_images_handler(
                img_denormalize_fn=config.img_denormalize, n_images=15, another_engine=trainer, prefix_tag="training"
            ),
            event_name=Events.ITERATION_COMPLETED(once=len(train_eval_loader) // 2),
        )

        if resize_aug is not None:
            @trainer.on(Events.EPOCH_STARTED)
            def log_train_image_size(_):
                tb_logger.writer.add_scalar("training/image_size", resize_aug.width, global_step=trainer.state.epoch)

    trainer.run(train_loader, max_epochs=config.num_epochs, epoch_length=getattr(config, "epoch_length", None))


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    dist.init_process_group("nccl", init_method="env://")

    # As we passed config with option --manual_config_load
    assert hasattr(config, "setup"), (
        "We need to manually setup the configuration, please set --manual_config_load " "to py_config_runner"
    )

    config = config.setup()

    assert_config(config, TRAINVAL_CONFIG)
    # The following attributes are automatically added by py_config_runner
    assert hasattr(config, "config_filepath") and isinstance(config.config_filepath, Path)
    assert hasattr(config, "script_filepath") and isinstance(config.script_filepath, Path)

    # dump python files to reproduce the run
    mlflow.log_artifact(config.config_filepath.as_posix())
    mlflow.log_artifact(config.script_filepath.as_posix())

    output_path = mlflow.get_artifact_uri()
    config.output_path = Path(output_path)

    if dist.get_rank() == 0:
        mlflow.log_params({"pytorch version": torch.__version__, "ignite version": ignite.__version__})
        mlflow.log_params(get_params(config, TRAINVAL_CONFIG))

    try:
        training(config, local_rank=local_rank, with_mlflow_logging=True, with_plx_logging=False)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        mlflow.log_param("Run Status", "FAILED")
        dist.destroy_process_group()
        raise e

    mlflow.log_param("Run Status", "OK")
    dist.destroy_process_group()
