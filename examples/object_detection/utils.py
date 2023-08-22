import datetime

import torch

from ignite.engine import Engine, Events
from ignite.handlers import Timer


MB = 1024.0 * 1024.0


class FBResearchLogger:
    def __init__(self, logger, delimiter="  "):
        self.delimiter = delimiter
        self.logger = logger
        self.iter_timer = None
        self.data_timer = None

    def attach(self, engine: Engine, name: str, every: int = 1):
        engine.add_event_handler(Events.EPOCH_STARTED, self.log_epoch_started, engine, name)
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=every), self.log_every, engine)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch_completed, engine)

        self.iter_timer = Timer(average=True)
        self.iter_timer.attach(
            engine,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED,
        )
        self.data_timer = Timer(average=True)
        self.data_timer.attach(
            engine,
            start=Events.EPOCH_STARTED,
            resume=Events.GET_BATCH_STARTED,
            pause=Events.GET_BATCH_COMPLETED,
            step=Events.GET_BATCH_COMPLETED,
        )

    def log_every(self, engine):
        cuda_max_mem = ""
        if torch.cuda.is_available():
            cuda_max_mem = f"GPU Max Mem: {torch.cuda.max_memory_allocated() / MB:.0f} MB"

        current_iter = engine.state.iteration % (engine.state.epoch_length + 1)
        iter_avg_time = self.iter_timer.value()

        eta_seconds = iter_avg_time * (engine.state.epoch_length - current_iter)

        meters = [f"{k}: {v:.4f}" for k, v in engine.state.metrics.items()]

        msg = self.delimiter.join(
            [
                f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}]",
                f"[{current_iter}/{engine.state.epoch_length}]:",
                f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}",
            ]
            + meters
            + [
                f"Iter time: {iter_avg_time:.4f} s",
                f"Data prep time: {self.data_timer.value():.4f} s",
                cuda_max_mem,
            ]
        )
        self.logger.info(msg)

    def log_epoch_started(self, engine, name):
        msg = f"{name}: start epoch [{engine.state.epoch}/{engine.state.max_epochs}]"
        self.logger.info(msg)

    def log_epoch_completed(self, engine):
        epoch_time = engine.state.times[Events.EPOCH_COMPLETED.name]
        msg = self.delimiter.join(
            [
                f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}]",
                f"Total time: {datetime.timedelta(seconds=int(epoch_time))}",
                f"({epoch_time / engine.state.epoch_length:.4f} s / it)",
            ]
        )
        self.logger.info(msg)
