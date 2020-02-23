import torch
import torch.distributed as dist

from ignite.engine import Events, Engine

from dataflow.vis import make_grid


def predictions_gt_images_handler(img_denormalize_fn, n_images=None, another_engine=None, prefix_tag=None):
    def wrapper(engine, logger, event_name):
        batch = engine.state.batch
        output = engine.state.output
        x, y = batch['image'], batch['target']
        y_pred = output[0]

        if y.shape == y_pred.shape and y.ndim == 4:
            # Case of y of shape (B, C, H, W)
            y = torch.argmax(y, dim=1)

        y_pred = torch.argmax(y_pred, dim=1).byte()

        if n_images is not None:
            x = x[:n_images, ...]
            y = y[:n_images, ...]
            y_pred = y_pred[:n_images, ...]

        grid_pred_gt = make_grid(x, y_pred, img_denormalize_fn, batch_gt=y)

        state = engine.state if another_engine is None else another_engine.state
        global_step = state.get_event_attrib_value(event_name)

        tag = "predictions_with_gt"
        if prefix_tag is not None:
            tag = "{}: {}".format(prefix_tag, tag)
        logger.writer.add_image(tag=tag, img_tensor=grid_pred_gt, global_step=global_step, dataformats="HWC")

    return wrapper


class DataflowBenchmark:
    def __init__(self, num_iters=100, prepare_batch=None, device="cuda"):

        from ignite.handlers import Timer

        def upload_to_gpu(engine, batch):
            if prepare_batch is not None:
                x, y = prepare_batch(batch, device=device, non_blocking=False)

        self.num_iters = num_iters
        self.benchmark_dataflow = Engine(upload_to_gpu)

        @self.benchmark_dataflow.on(Events.ITERATION_COMPLETED(once=num_iters))
        def stop_benchmark_dataflow(engine):
            engine.terminate()

        if dist.is_available() and dist.get_rank() == 0:

            @self.benchmark_dataflow.on(Events.ITERATION_COMPLETED(every=num_iters // 100))
            def show_progress_benchmark_dataflow(engine):
                print(".", end=" ")

        self.timer = Timer(average=False)
        self.timer.attach(
            self.benchmark_dataflow,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED,
        )

    def attach(self, trainer, train_loader):

        from torch.utils.data import DataLoader

        @trainer.on(Events.STARTED)
        def run_benchmark(_):

            rank = dist.get_rank() if dist.is_initialized() else 0

            if rank == 0:
                print("-" * 50)
                print(" - Dataflow benchmark")

            self.benchmark_dataflow.run(train_loader)
            t = self.timer.value()

            if rank == 0:
                print(" ")
                print(" Total time ({} iterations) : {:.5f} seconds".format(self.num_iters, t))
                print(" time per iteration         : {} seconds".format(t / self.num_iters))

                if isinstance(train_loader, DataLoader):
                    num_images = train_loader.batch_size * self.num_iters
                    print(" number of images / s       : {}".format(num_images / t))

                print("-" * 50)
