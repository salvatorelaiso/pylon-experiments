from typing import Optional

import torch
from pyxdameraulevenshtein import (
    damerau_levenshtein_distance,
    normalized_damerau_levenshtein_distance,
)
from torchmetrics import Metric


class DamerauLevenshteinDistance(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, input_logits: bool = True):
        super().__init__()
        self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")
        self.input_logits = input_logits

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if self.input_logits:
            predictions = torch.argmax(predictions, dim=-1)

        assert predictions.shape == targets.shape

        # Convert single tensors to batched tensors
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        elif predictions.ndim > 2:
            raise ValueError("Only 1D and 2D tensors are supported")

        for i in range(predictions.shape[0]):
            self.distance += damerau_levenshtein_distance(
                predictions[i].int().tolist(),
                targets[i].int().tolist(),
            )
            self.total_length += 1

    def compute(self):
        return self.distance / self.total_length

    def __iter__(self):
        super().__iter__()


class NormalizedDamerauLevenshteinDistance(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, input_logits: bool = True):
        super().__init__()
        self.add_state("norm_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")
        self.input_logits = input_logits

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if self.input_logits:
            predictions = torch.argmax(predictions, dim=-1)

        assert predictions.shape == targets.shape

        # Convert single tensors to batched tensors
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        elif predictions.ndim > 2:
            raise ValueError("Only 1D and 2D tensors are supported")

        for i in range(predictions.shape[0]):
            self.norm_distance += normalized_damerau_levenshtein_distance(
                predictions[i].int().tolist(),
                targets[i].int().tolist(),
            )
            self.total_length += 1

    def compute(self):
        return self.norm_distance / self.total_length

    def __iter__(self):
        super().__iter__()


class WeightedNormalizedDamerauLevenshteinDistance(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, input_logits: bool = True):
        super().__init__()
        self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")
        self.input_logits = input_logits

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if self.input_logits:
            predictions = torch.argmax(predictions, dim=-1)

        assert predictions.shape == targets.shape

        # Convert single tensors to batched tensors
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        elif predictions.ndim > 2:
            raise ValueError("Only 1D and 2D tensors are supported")

        for i in range(predictions.shape[0]):
            self.distance += damerau_levenshtein_distance(
                predictions[i].int().tolist(),
                targets[i].int().tolist(),
            )
            self.total_length += max(len(predictions[i]), len(targets[i]))

    def compute(self):
        return self.distance / self.total_length

    def __iter__(self):
        super().__iter__()
