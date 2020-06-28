import fire

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from torchvision.models import wide_resnet50_2

import ignite.distributed as idist
from ignite.engine import Events, Engine, create_supervised_evaluator, convert_tensor
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Timer
from ignite.contrib.handlers import ProgressBar

from deepspeed import DeepSpeedLight

from utils import get_train_eval_loaders


class Args:
    pass


def main(dataset_path, batch_size=256, max_epochs=10):
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
    torch.backends.cudnn.benchmark = True

    device = "cuda"

    train_loader, test_loader, eval_train_loader = get_train_eval_loaders(dataset_path, batch_size=batch_size)

    model = wide_resnet50_2(num_classes=100).to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss().to(device)

    idist.set_local_rank(0)

    ds_args = Args()
    ds_args.local_rank = 0
    ds_args.deepspeed_config = None
    ds_config_params = {"train_batch_size": batch_size, "steps_per_print": len(train_loader)}
    model_engine = DeepSpeedLight(ds_args, model, optimizer=optimizer, config_params=ds_config_params)
    model_engine.tput_timer.steps_per_output = len(train_loader)

    def train_step(engine, batch):
        x = convert_tensor(batch[0], device, non_blocking=True)
        y = convert_tensor(batch[1], device, non_blocking=True)

        optimizer.zero_grad()

        y_pred = model_engine(x)
        loss = criterion(y_pred, y)

        model_engine.backward(loss)
        model_engine.step()

        return loss.item()

    trainer = Engine(train_step)
    timer = Timer(average=True)
    timer.attach(trainer, step=Events.EPOCH_COMPLETED)
    ProgressBar(persist=True).attach(trainer, output_transform=lambda out: {"batch loss": out})

    metrics = {"Accuracy": Accuracy(), "Loss": Loss(criterion)}

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def log_metrics(engine, title):
        for name in metrics:
            print("\t{} {}: {:.2f}".format(title, name, engine.state.metrics[name]))

    @trainer.on(Events.COMPLETED)
    def run_validation(_):
        print("- Mean elapsed time for 1 epoch: {}".format(timer.value()))
        print("- Metrics:")
        with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "Train"):
            evaluator.run(eval_train_loader)

        with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "Test"):
            evaluator.run(test_loader)

    trainer.run(train_loader, max_epochs=max_epochs)


if __name__ == "__main__":
    fire.Fire(main)
