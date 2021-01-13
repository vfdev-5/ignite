import collections.abc as collections
import logging
import random
from typing import Any, Callable, Optional, Tuple, Type, Union, cast, Dict

import torch

__all__ = ["convert_tensor", "apply_to_tensor", "apply_to_type", "to_onehot", "setup_logger", "manual_seed", "log_metrics"]


def convert_tensor(
    input_: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Move tensors to relevant device."""

    def _func(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, non_blocking=non_blocking) if device is not None else tensor

    return apply_to_tensor(input_, _func)


def apply_to_tensor(
    input_: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes], func: Callable
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on a tensor or mapping, or sequence of tensors.
    """
    return apply_to_type(input_, torch.Tensor, func)


def apply_to_type(
    input_: Union[Any, collections.Sequence, collections.Mapping, str, bytes],
    input_type: Union[Type, Tuple[Type[Any], Any]],
    func: Callable,
) -> Union[Any, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on a object of `input_type` or mapping, or sequence of objects of `input_type`.
    """
    if isinstance(input_, input_type):
        return func(input_)
    if isinstance(input_, (str, bytes)):
        return input_
    if isinstance(input_, collections.Mapping):
        return cast(Callable, type(input_))(
            {k: apply_to_type(sample, input_type, func) for k, sample in input_.items()}
        )
    if isinstance(input_, tuple) and hasattr(input_, "_fields"):  # namedtuple
        return cast(Callable, type(input_))(*(apply_to_type(sample, input_type, func) for sample in input_))
    if isinstance(input_, collections.Sequence):
        return cast(Callable, type(input_))([apply_to_type(sample, input_type, func) for sample in input_])
    raise TypeError((f"input must contain {input_type}, dicts or lists; found {type(input_)}"))


def to_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a tensor of indices of any shape `(N, ...)` to a
    tensor of one-hot indicators of shape `(N, num_classes, ...) and of type uint8. Output's device is equal to the
    input's device`.
    """
    onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:], dtype=torch.uint8, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    filepath: Optional[str] = None,
    distributed_rank: Optional[int] = None,
) -> logging.Logger:
    """Setups logger: name, level, format etc.

    Args:
        name (str, optional): new name for the logger. If None, the standard logger is used.
        level (int): logging level, e.g. CRITICAL, ERROR, WARNING, INFO, DEBUG
        format (str): logging format. By default, `%(asctime)s %(name)s %(levelname)s: %(message)s`
        filepath (str, optional): Optional logging file path. If not None, logs are written to the file.
        distributed_rank (int, optional): Optional, rank in distributed configuration to avoid logger setup for workers.
            If None, distributed_rank is initialized to the rank of process.

    Returns:
        logging.Logger

    For example, to improve logs readability when training with a trainer and evaluator:

    .. code-block:: python

        from ignite.utils import setup_logger

        trainer = ...
        evaluator = ...

        trainer.logger = setup_logger("trainer")
        evaluator.logger = setup_logger("evaluator")

        trainer.run(data, max_epochs=10)

        # Logs will look like
        # 2020-01-21 12:46:07,356 trainer INFO: Engine run starting with max_epochs=5.
        # 2020-01-21 12:46:07,358 trainer INFO: Epoch[1] Complete. Time taken: 00:5:23
        # 2020-01-21 12:46:07,358 evaluator INFO: Engine run starting with max_epochs=1.
        # 2020-01-21 12:46:07,358 evaluator INFO: Epoch[1] Complete. Time taken: 00:01:02
        # ...

    """
    logger = logging.getLogger(name)

    # don't propagate to ancestors
    # the problem here is to attach handlers to loggers
    # should we provide a default configuration less open ?
    if name is not None:
        logger.propagate = False

    # Remove previous handlers
    if logger.hasHandlers():
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter(format)

    if distributed_rank is None:
        import ignite.distributed as idist

        distributed_rank = idist.get_rank()

    if distributed_rank > 0:
        logger.addHandler(logging.NullHandler())
    else:
        logger.setLevel(level)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if filepath is not None:
            fh = logging.FileHandler(filepath)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def manual_seed(seed: int) -> None:
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).

    Args:
        seed (int): Random state seed

    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def log_metrics(logger: logging.Logger, epoch: int, elapsed: float, tag: str, metrics: Dict[str, Any]) -> None:
    """Helper method to quickly log computed metrics with a logger

    Args:
        logger (Logger): logger to use
        epoch (int): epoch value to log
        elapsed (float): elapsed time to compute metrics
        tag (str): 

    Example of usage:

    .. code-block:: python

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation():
            epoch = trainer.state.epoch
            state = train_evaluator.run(train_eval_loader)
            log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
            state = evaluator.run(val_loader)
            log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    .. code-block:: text

        TODO: OUTPUT

    """
    def log_tensor(t: Any) -> None:

        if not isinstance(t, torch.Tensor):
            return f"{t}"

        print_opts = torch._tensor_str.PRINT_OPTS
        torch.set_printoptions(precision=3, sci_mode=False)
        output = None

        if t.ndim < 2:
            output = f"{t.cpu().tolist()}"
        elif t.ndim == 2:
            output = f"\n{t.cpu().numpy()}"
        else:
            output = f"{t.cpu()}"

        torch.set_printoptions(precision=print_opts.precision, sci_mode=print_opts.sci_mode)
        return output

    metrics_vals = "\n".join([f"\t{k}: {log_tensor(v)}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {int(elapsed)} - {tag} metrics:\n\n {metrics_vals}\n")
