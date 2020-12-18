from enum import Enum
from typing import Callable, Sequence, Union

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


__all__ = ["MeanAveragePrecision"]


class GeomMatchPolicy(Enum):
    PASCAL_VOC = 0
    MS_COCO = 1


class MeanAveragePrecision(Metric):
    """Calculates Mean Average Precision metric for detection task.

    """

    def __init__(
        self,
        num_classes: int,
        match_policy: GeomMatchPolicy = GeomMatchPolicy.PASCAL_VOC,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if num_classes < 2:
            raise ValueError("Argument num_classes should be larger then one, but given {}".format(num_classes))

        if match_policy not in GeomMatchPolicy:
            raise ValueError(
                "Argument match_policy should be one of GeomMatchPolicy, but given {}".format(match_policy)
            )

        self.match_policy = match_policy
        self.num_classes = num_classes
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0
        # self.

    @reinit__is_reduced
    def update(self, output: List[torch.Tensor]) -> None:

        pass

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
        return 0.0

