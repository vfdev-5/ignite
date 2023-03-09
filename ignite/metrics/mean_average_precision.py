from collections import defaultdict
from typing import Callable, cast, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

import ignite.distributed as idist
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class MeanAveragePrecision(Metric):
    def __init__(
        self,
        iou_thresholds: Optional[Union[List[float], torch.Tensor]] = None,
        rec_thresholds: Optional[Union[List[float], torch.Tensor]] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        r"""Calculate the mean average precision of overall categories.

        If this metric is used with :class:`~ignite.engine.engine.Engine`, then output of evaluator should be
        a tuple of 2 lists of tensors, e.g ``([pred1, pred2, pred3, ...], [gt1, gt2, gt3, ...])``.
        First list is a batch of predictions and the second list is a batch of targets.
        Lists should have the same length.

        The shape of the ground truth is (N, 5) where N stands for the number of ground truth boxes and 5 is
        (x1, y1, x2, y2, class_number).
        The shape of the prediction is (M, 6) where M stands for the number of predicted boxes and 6 is
        (x1, y1, x2, y2, confidence, class_number).

        Args:
            iou_thresholds: list of IoU thresholds to be considered for computing Mean Average Precision.
                Values should be between 0 and 1. List is sorted internally. Default is that of the COCO
                official evaluation metric.
            output_transform: a callable that is used to transform the
                :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
                By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
            device: specifies which device updates are accumulated on. Setting the
                metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
                non-blocking. By default, CPU.
        """
        try:
            from torchvision.ops import box_iou

            self.box_iou = box_iou
        except ImportError:
            raise ModuleNotFoundError("This module requires torchvision to be installed.")

        if iou_thresholds is None:
            iou_thresholds = torch.linspace(0.5, 0.95, 10).tolist()

        self.iou_thresholds = self._setup_thresholds(iou_thresholds, "iou_thresholds")

        if rec_thresholds is None:
            rec_thresholds = torch.linspace(0, 1, 101, device=device, dtype=torch.double)

        self.rec_thresholds = self._setup_thresholds(rec_thresholds, "rec_thresholds")

        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    def _setup_thresholds(self, thresholds: Union[List[float], torch.Tensor], tag: str) -> torch.Tensor:
        if isinstance(thresholds, list):
            thresholds = torch.tensor(thresholds)

        if isinstance(thresholds, torch.Tensor):
            if thresholds.ndim != 1:
                raise ValueError(
                    f"{tag} should be a one-dimensional tensor or a list of floats"
                    f", given a {thresholds.ndim}-dimensional tensor."
                )
            thresholds = thresholds.sort().values
        else:
            raise TypeError(f"{tag} should be a list of floats or a tensor, given {type(thresholds)}.")

        if thresholds.min() < 0 or thresholds.max() > 1:
            raise ValueError(f"{tag} values should be between 0 and 1, given {thresholds}")

        return thresholds

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_categories: int = 0
        self._tp: Dict[int, List[torch.BoolTensor]] = defaultdict(lambda: [])
        self._num_gt: Dict[int, int] = defaultdict(lambda: 0)
        self._scores: Dict[int, List[torch.Tensor]] = defaultdict(lambda: [])

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Metric update function using predictions and targets corresponding to a single image.

        Args:
            output: a tuple of 2 tensors in which the first one is the prediction and the second is the ground truth.
                The shape of the ground truth is (N, 5) where N stands for the number of ground truth boxes and 5 is
                (x1, y1, x2, y2, class_number). The shape of the prediction is (M, 6) where M stands for the
                number of predicted boxes and 6 is (x1, y1, x2, y2, confidence, class_number).
        """
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndim == 3 and y_pred.shape[0] == 1:
            y_pred = y_pred.squeeze(0)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)

        if y.ndim != 2 or y.shape[1] != 5:
            raise ValueError(f"Provided y with a wrong shape, expected (N, 5), got {y.shape}")
        if y_pred.ndim != 2 or y_pred.shape[1] != 6:
            raise ValueError(f"Provided y_pred with a wrong shape, expected (M, 6), got {y_pred.shape}")
        categories = torch.cat((y[:, 4], y_pred[:, 5])).int().unique().tolist()
        self._num_categories = max(self._num_categories, max(categories, default=-1) + 1)
        iou = self.box_iou(y_pred[:, :4], y[:, :4])

        for category in categories:
            class_index_gt = y[:, 4] == category
            num_category_gt = class_index_gt.sum()
            self._num_gt[category] += num_category_gt

            class_index_dt = y_pred[:, 5] == category
            if not class_index_dt.any():
                continue

            category_scores = y_pred[class_index_dt, 4]
            self._scores[category].append(category_scores.to(self._device))

            category_tp = torch.zeros(
                (len(self.iou_thresholds), class_index_dt.sum().item()), dtype=torch.bool, device=self._device
            )
            if class_index_gt.any():
                class_iou = iou[:, class_index_gt][class_index_dt, :]
                category_maximum_iou = class_iou.max()
                category_pred_idx_sorted_by_decreasing_score = torch.argsort(
                    category_scores, stable=True, descending=True
                ).tolist()
                for thres_idx, iou_thres in enumerate(self.iou_thresholds):
                    if iou_thres <= category_maximum_iou:
                        matched_gt_indices = set()
                        for pred_idx in category_pred_idx_sorted_by_decreasing_score:
                            match_iou, match_idx = -1.0, -1
                            for gt_idx in range(num_category_gt):
                                if (class_iou[pred_idx][gt_idx] < iou_thres) or (gt_idx in matched_gt_indices):
                                    continue
                                if class_iou[pred_idx][gt_idx] >= match_iou:
                                    match_iou = class_iou[pred_idx][gt_idx]
                                    match_idx = gt_idx
                            if match_idx != -1:
                                matched_gt_indices.add(match_idx)
                                category_tp[thres_idx][pred_idx] = True
                    else:
                        break

            self._tp[category].append(cast(torch.BoolTensor, category_tp))

    @sync_all_reduce("_num_categories:MAX")
    def compute(self) -> float:
        # `gloo` does not support `gather` on GPU. Do we need
        #  to take an action regarding that?
        num_gt = torch.tensor([self._num_gt[cat_id] for cat_id in range(self._num_categories)], device=self._device)
        num_gt = cast(torch.Tensor, idist.all_reduce(num_gt))

        num_thresholds = len(self.iou_thresholds)
        _tp = {
            cat_id: torch.cat(cast(List[torch.Tensor], self._tp[cat_id]), dim=1)
            if len(self._tp[cat_id]) != 0
            else torch.empty((num_thresholds, 0), dtype=torch.bool, device=self._device)
            for cat_id in range(self._num_categories)
        }
        _scores = {
            cat_id: torch.cat(self._scores[cat_id], dim=0)
            if len(self._scores[cat_id]) != 0
            else torch.tensor([], dtype=torch.float, device=self._device)
            for cat_id in range(self._num_categories)
        }

        num_predictions = torch.tensor(
            [_tp[cat_idx].shape[1] if cat_idx in _tp else 0 for cat_idx in range(self._num_categories)],
            device=self._device,
        )
        world_size = idist.get_world_size()
        if world_size > 1:
            ranks_num_preds = torch.stack(
                cast(torch.Tensor, idist.all_gather(num_predictions)).split(split_size=self._num_categories)
            )
            max_num_predictions = ranks_num_preds.amax(dim=0)
        else:
            max_num_predictions = num_predictions

        recall_thresh_repeated_iou_thresh_times = self.rec_thresholds.repeat((num_thresholds, 1))
        average_precision = torch.tensor(0.0, device=self._device, dtype=torch.double)
        num_present_categories = self._num_categories
        for category_idx in range(self._num_categories):

            if num_gt[category_idx] == 0:
                num_present_categories -= 1
                continue
            max_num_preds_in_cat = max_num_predictions[category_idx]
            if max_num_preds_in_cat == 0:
                continue

            if world_size > 1:
                ranks_tp = cast(
                    torch.Tensor,
                    idist.all_gather(
                        F.pad(
                            _tp[category_idx],
                            (
                                0,  # type: ignore[arg-type]
                                max_num_preds_in_cat - num_predictions[category_idx],
                            ),
                        ).to(torch.uint8),
                    ),
                )
                ranks_scores = cast(
                    torch.Tensor,
                    idist.all_gather(
                        F.pad(
                            _scores[category_idx],
                            (
                                0,  # type: ignore[arg-type]
                                max_num_preds_in_cat - num_predictions[category_idx],
                            ),
                        )
                    ),
                )

                ranks_tp_unpadded = [
                    ranks_tp[
                        r * num_thresholds : (r + 1) * num_thresholds,
                        : ranks_num_preds[r, category_idx],  # type: ignore[misc]
                    ].to(torch.bool)
                    for r in range(world_size)
                ]
                tp = torch.cat(ranks_tp_unpadded, dim=1)

                ranks_scores_unpadded = [
                    ranks_scores[
                        r * max_num_preds_in_cat : r * max_num_preds_in_cat  # type: ignore[misc]
                        + ranks_num_preds[r, category_idx]
                    ]
                    for r in range(world_size)
                ]
                scores = torch.cat(ranks_scores_unpadded, dim=0)
            else:
                tp = _tp[category_idx]
                scores = _scores[category_idx]

            tp = tp[:, torch.argsort(scores, stable=True, descending=True)]

            tp_summation = tp.cumsum(dim=1).double()
            fp_summation = (~tp).cumsum(dim=1).double()
            recall = tp_summation / num_gt[category_idx]
            precision = tp_summation / (fp_summation + tp_summation + torch.finfo(torch.double).eps)

            recall_thresh_indices = torch.searchsorted(recall, recall_thresh_repeated_iou_thresh_times)
            for t in range(num_thresholds):
                for r_idx in recall_thresh_indices[t]:
                    if r_idx == recall.shape[1]:
                        break
                    # Interpolated precision. Please refer to PASCAL VOC paper section 4.2
                    # for more information. MS COCO is like PASCAL in this regard.
                    average_precision += precision[t][r_idx:].max()
        if num_present_categories == 0:
            mAP = torch.tensor(-1.0, device=self._device)
        else:
            mAP = average_precision / (num_present_categories * len(self.rec_thresholds) * num_thresholds)

        return mAP.item()
