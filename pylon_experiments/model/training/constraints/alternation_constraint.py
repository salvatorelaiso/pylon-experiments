from dataclasses import dataclass

import numpy as np
import torch
from pylon.constraint import constraint
from pylon.sampling_solver import WeightedSamplingSolver

from .constraint import Constraint


@dataclass(frozen=True, kw_only=True)
class AlternationLengthConstraintSettings:
    activity_a: int
    activity_b: int
    max_alternation_length: int
    padding_value: int = 0


class AlternationConstraint(Constraint):
    def __init__(
        self, settings: AlternationLengthConstraintSettings, num_samples: int = 100
    ):
        self.__settings = settings
        self.__constraint = constraint(
            AlternationConstraint.alternation_length_constraint_fn,
            WeightedSamplingSolver(num_samples=num_samples),
        )

    def __call__(self, logits: torch.Tensor, *, lengths: list[int]) -> torch.Tensor:
        return self.__constraint(
            logits.cpu(),
            lengths=lengths,
            activity_a=self.__settings.activity_a,
            activity_b=self.__settings.activity_b,
            max_alternation_length=self.__settings.max_alternation_length,
            padding_value=self.__settings.padding_value,
        )

    def __repr__(self):
        return f"AlternationConstraint(activity_a={self.__settings.activity_a}, activity_b={self.__settings.activity_b}, max_alternation_length={self.__settings.max_alternation_length})"

    def __str__(self):
        return self.__repr__()

    def short_str(self):
        return f"max-alt_{self.__settings.activity_a}_{self.__settings.activity_b}={self.__settings.max_alternation_length}"

    @staticmethod
    def __alternation_length(
        batch: torch.Tensor, activity_a: int, activity_b: int
    ) -> np.ndarray:
        a_batch = (batch == activity_a).numpy()
        b_batch = (batch == activity_b).numpy()

        a_even_batch = a_batch.copy()
        a_even_batch[:, 1::2] = False  # Set to False all the odd columns
        a_odd_batch = a_batch.copy()
        a_odd_batch[:, ::2] = False  # Set to False all the even columns

        b_even_batch = b_batch.copy()
        b_even_batch[:, 1::2] = False  # Set to False all the odd columns
        b_odd_batch = b_batch.copy()
        b_odd_batch[:, ::2] = False  # Set to False all the even columns

        a_b_batch = np.logical_or(a_even_batch, b_odd_batch)
        b_a_batch = np.logical_or(a_odd_batch, b_even_batch)

        # https://stackoverflow.com/a/57437919/11270758
        def get_max_length(batch):
            # Add columns of zeros to the left and right.
            padded = np.pad(batch, [(0, 0), (1, 1)], mode="constant")

            # Get indices in each row where transitions between 0's and 1's occur.
            diffs = np.diff(padded)
            rows, wheres = np.where(diffs)

            # Compute the length of each patch of 1's.
            rows, lengths = rows[::2], np.diff(wheres)[::2]

            # Compute the maximal length for each row.
            rows, split_at = np.unique(rows, return_index=True)
            maxima = np.maximum.reduceat(lengths, split_at)

            # Store the computed maxima
            maxima_batch = np.zeros(batch.shape[0], dtype=np.int32)
            maxima_batch[rows] = maxima

            return maxima_batch

        a_b_maxima = get_max_length(a_b_batch)
        b_a_maxima = get_max_length(b_a_batch)

        output = np.maximum(a_b_maxima, b_a_maxima)

        return output

    @staticmethod
    def alternation_length_constraint_fn(
        sampled_predictions: torch.Tensor,
        kwargs: dict,
    ) -> torch.BoolTensor:
        # Overwrite the predictions from padded positions
        # in order to not interfere with the constraint.
        for i, length in enumerate(kwargs["lengths"]):
            sampled_predictions[i, length:] = kwargs["padding_value"]
        alternation_lengths = AlternationConstraint.__alternation_length(
            sampled_predictions, kwargs["activity_a"], kwargs["activity_b"]
        )

        return torch.from_numpy(alternation_lengths <= kwargs["max_alternation_length"])
