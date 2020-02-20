import os
from functools import partial

import numpy as np
import torch

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, RunningAverage

import pytest


def test_wrong_input_args():
    with pytest.raises(TypeError, match=r"Argument src should be a Metric or None."):
        RunningAverage(src=[12, 34])

    with pytest.raises(ValueError, match=r"Argument alpha should be a float between"):
        RunningAverage(alpha=-1.0)

    with pytest.raises(ValueError, match=r"Argument output_transform should be None if src is a Metric"):
        RunningAverage(Accuracy(), output_transform=lambda x: x[0])

    with pytest.raises(ValueError, match=r"Argument output_transform should not be None if src corresponds"):
        RunningAverage()

    with pytest.raises(ValueError, match=r"Argument device should be None if src is a Metric"):
        RunningAverage(Accuracy(), device="cpu")


def test_integration():

    n_iters = 100
    batch_size = 10
    n_classes = 10
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_iters, batch_size, n_classes))
    loss_values = iter(range(n_iters))

    def update_fn(engine, batch):
        loss_value = next(loss_values)
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        return loss_value, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    trainer = Engine(update_fn)
    alpha = 0.98

    acc_metric = RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]]), alpha=alpha)
    acc_metric.attach(trainer, "running_avg_accuracy")

    avg_output = RunningAverage(output_transform=lambda x: x[0], alpha=alpha)
    avg_output.attach(trainer, "running_avg_output")

    running_avg_acc = [None]

    @trainer.on(Events.ITERATION_COMPLETED)
    def manual_running_avg_acc(engine):
        _, y_pred, y = engine.state.output
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y).view(-1)
        num_correct = torch.sum(correct).item()
        num_examples = correct.shape[0]
        batch_acc = num_correct * 1.0 / num_examples
        if running_avg_acc[0] is None:
            running_avg_acc[0] = batch_acc
        else:
            running_avg_acc[0] = running_avg_acc[0] * alpha + (1.0 - alpha) * batch_acc
        engine.state.running_avg_acc = running_avg_acc[0]

    @trainer.on(Events.EPOCH_STARTED)
    def running_avg_output_init(engine):
        engine.state.running_avg_output = None

    @trainer.on(Events.ITERATION_COMPLETED)
    def running_avg_output_update(engine):
        if engine.state.running_avg_output is None:
            engine.state.running_avg_output = engine.state.output[0]
        else:
            engine.state.running_avg_output = (
                engine.state.running_avg_output * alpha + (1.0 - alpha) * engine.state.output[0]
            )

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_acc_values(engine):
        assert engine.state.running_avg_acc == engine.state.metrics["running_avg_accuracy"], "{} vs {}".format(
            engine.state.running_avg_acc, engine.state.metrics["running_avg_accuracy"]
        )

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_output_values(engine):
        assert engine.state.running_avg_output == engine.state.metrics["running_avg_output"], "{} vs {}".format(
            engine.state.running_avg_output, engine.state.metrics["running_avg_output"]
        )

    np.random.seed(10)
    running_avg_acc = [None]
    n_iters = 10
    batch_size = 10
    n_classes = 10
    data = list(range(n_iters))
    loss_values = iter(range(n_iters))
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_iters, batch_size, n_classes))
    trainer.run(data, max_epochs=1)

    running_avg_acc = [None]
    n_iters = 10
    batch_size = 10
    n_classes = 10
    data = list(range(n_iters))
    loss_values = iter(range(n_iters))
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_iters, batch_size, n_classes))
    trainer.run(data, max_epochs=1)


def test_epoch_unbound():

    n_iters = 10
    n_epochs = 3
    batch_size = 10
    n_classes = 10
    data = list(range(n_iters))
    loss_values = iter(range(n_epochs * n_iters))
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_epochs * n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_epochs * n_iters, batch_size, n_classes))

    def update_fn(engine, batch):
        loss_value = next(loss_values)
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        return loss_value, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    trainer = Engine(update_fn)
    alpha = 0.98

    acc_metric = RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]]), alpha=alpha, epoch_bound=False)
    acc_metric.attach(trainer, "running_avg_accuracy")

    avg_output = RunningAverage(output_transform=lambda x: x[0], alpha=alpha, epoch_bound=False)
    avg_output.attach(trainer, "running_avg_output")

    running_avg_acc = [None]

    @trainer.on(Events.STARTED)
    def running_avg_output_init(engine):
        engine.state.running_avg_output = None

    @trainer.on(Events.ITERATION_COMPLETED, running_avg_acc)
    def manual_running_avg_acc(engine, running_avg_acc):
        _, y_pred, y = engine.state.output
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y).view(-1)
        num_correct = torch.sum(correct).item()
        num_examples = correct.shape[0]
        batch_acc = num_correct * 1.0 / num_examples
        if running_avg_acc[0] is None:
            running_avg_acc[0] = batch_acc
        else:
            running_avg_acc[0] = running_avg_acc[0] * alpha + (1.0 - alpha) * batch_acc
        engine.state.running_avg_acc = running_avg_acc[0]

    @trainer.on(Events.ITERATION_COMPLETED)
    def running_avg_output_update(engine):
        if engine.state.running_avg_output is None:
            engine.state.running_avg_output = engine.state.output[0]
        else:
            engine.state.running_avg_output = (
                engine.state.running_avg_output * alpha + (1.0 - alpha) * engine.state.output[0]
            )

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_acc_values(engine):
        assert engine.state.running_avg_acc == engine.state.metrics["running_avg_accuracy"], "{} vs {}".format(
            engine.state.running_avg_acc, engine.state.metrics["running_avg_accuracy"]
        )

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_output_values(engine):
        assert engine.state.running_avg_output == engine.state.metrics["running_avg_output"], "{} vs {}".format(
            engine.state.running_avg_output, engine.state.metrics["running_avg_output"]
        )

    trainer.run(data, max_epochs=3)


def test_multiple_attach():
    n_iters = 100
    errD_values = iter(np.random.rand(n_iters))
    errG_values = iter(np.random.rand(n_iters))
    D_x_values = iter(np.random.rand(n_iters))
    D_G_z1 = iter(np.random.rand(n_iters))
    D_G_z2 = iter(np.random.rand(n_iters))

    def update_fn(engine, batch):
        return {
            "errD": next(errD_values),
            "errG": next(errG_values),
            "D_x": next(D_x_values),
            "D_G_z1": next(D_G_z1),
            "D_G_z2": next(D_G_z2),
        }

    trainer = Engine(update_fn)
    alpha = 0.98

    # attach running average
    monitoring_metrics = ["errD", "errG", "D_x", "D_G_z1", "D_G_z2"]
    for metric in monitoring_metrics:
        foo = partial(lambda x, metric: x[metric], metric=metric)
        RunningAverage(alpha=alpha, output_transform=foo).attach(trainer, metric)

    @trainer.on(Events.ITERATION_COMPLETED)
    def check_values(engine):

        values = []
        for metric in monitoring_metrics:
            values.append(engine.state.metrics[metric])

        values = set(values)
        assert len(values) == len(monitoring_metrics)

    data = list(range(n_iters))
    trainer.run(data)


def _test_distrib_on_output(device):
    import torch.distributed as dist

    rank = dist.get_rank()
    n_iters = 10
    n_epochs = 3
    batch_size = 10

    # Data per rank
    data = list(range(n_iters))
    k = n_epochs * batch_size * n_iters
    all_loss_values = torch.arange(0, k * dist.get_world_size(), dtype=torch.float64).to(device)
    loss_values = iter(all_loss_values[k * rank : k * (rank + 1)])

    def update_fn(engine, batch):
        loss_value = next(loss_values)
        return loss_value.item()

    trainer = Engine(update_fn)
    alpha = 0.98

    avg_output = RunningAverage(output_transform=lambda x: x, alpha=alpha, epoch_bound=False, device=device)
    avg_output.attach(trainer, "running_avg_output")

    @trainer.on(Events.STARTED)
    def running_avg_output_init(engine):
        engine.state.running_avg_output = None

    @trainer.on(Events.ITERATION_COMPLETED)
    def running_avg_output_update(engine):
        i = engine.state.iteration - 1
        o = sum([all_loss_values[i + j * k] for j in range(dist.get_world_size())]).item()
        if engine.state.running_avg_output is None:
            engine.state.running_avg_output = o
        else:
            engine.state.running_avg_output = engine.state.running_avg_output * alpha + (1.0 - alpha) * o

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_output_values(engine):
        assert engine.state.running_avg_output == pytest.approx(
            engine.state.metrics["running_avg_output"]
        ), "{}: {} vs {}".format(
            engine.state.iteration, engine.state.running_avg_output, engine.state.metrics["running_avg_output"]
        )

    trainer.run(data, max_epochs=3)


def _test_distrib_on_metric(device):
    import torch.distributed as dist

    rank = dist.get_rank()
    n_iters = 10
    n_epochs = 3
    batch_size = 10
    n_classes = 10

    data = list(range(n_iters))
    np.random.seed(12)
    all_y_true_batch_values = np.random.randint(
        0, n_classes, size=(dist.get_world_size(), n_epochs * n_iters, batch_size)
    )
    all_y_pred_batch_values = np.random.rand(dist.get_world_size(), n_epochs * n_iters, batch_size, n_classes)

    y_true_batch_values = iter(all_y_true_batch_values[rank, ...])
    y_pred_batch_values = iter(all_y_pred_batch_values[rank, ...])

    def update_fn(engine, batch):
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    trainer = Engine(update_fn)
    alpha = 0.98

    acc_metric = RunningAverage(
        Accuracy(output_transform=lambda x: [x[0], x[1]], device=device), alpha=alpha, epoch_bound=False
    )
    acc_metric.attach(trainer, "running_avg_accuracy")

    running_avg_acc = [None]
    true_acc_metric = Accuracy(device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def manual_running_avg_acc(engine):
        i = engine.state.iteration - 1

        true_acc_metric.reset()
        for j in range(dist.get_world_size()):
            output = (
                torch.from_numpy(all_y_pred_batch_values[j, i, :, :]),
                torch.from_numpy(all_y_true_batch_values[j, i, :]),
            )
            true_acc_metric.update(output)

        batch_acc = true_acc_metric._num_correct * 1.0 / true_acc_metric._num_examples

        if running_avg_acc[0] is None:
            running_avg_acc[0] = batch_acc
        else:
            running_avg_acc[0] = running_avg_acc[0] * alpha + (1.0 - alpha) * batch_acc
        engine.state.running_avg_acc = running_avg_acc[0]

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_acc_values(engine):
        assert engine.state.running_avg_acc == engine.state.metrics["running_avg_accuracy"], "{} vs {}".format(
            engine.state.running_avg_acc, engine.state.metrics["running_avg_accuracy"]
        )

    trainer.run(data, max_epochs=3)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = "cuda:{}".format(local_rank)
    _test_distrib_on_output(device)
    _test_distrib_on_metric(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_on_output(device)
    _test_distrib_on_metric(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_on_output(device)
    _test_distrib_on_metric(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl["local_rank"])
    _test_distrib_on_output(device)
    _test_distrib_on_metric(device)
