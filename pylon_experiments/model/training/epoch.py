import os
import pathlib
from itertools import zip_longest
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchmetrics
from declare4pylon.constraint import DeclareConstraint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from pylon_experiments.data.trace_dataset import TraceDataset
from pylon_experiments.model.model import NextActivityPredictor


def train_with_constraints_epoch(
    *,
    epoch: int,
    epochs: int,
    model: NextActivityPredictor,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    constraints: list[DeclareConstraint],
    constraints_multipliers: list[DeclareConstraint],
    metrics: dict[str, torchmetrics.Metric],
    device: torch.device,
    dataloader: DataLoader,
    mode: Literal["train", "val", "test"] = "train",
    ignore_task_loss: bool = False,
    constraints_multiplier: float | None = None,
):
    loader = tqdm(
        dataloader,
        unit="batch",
        desc=f"Epoch {epoch+1}/{epochs} - {mode}",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )
    train = mode == "train"
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_samples = 0

    constraint_losses = {str(constraint): 0 for constraint in constraints}

    for traces, lengths in loader:
        max_length = lengths[0]

        if not ignore_task_loss:
            for (
                prefixes,
                next_activity,
                prefixes_lengths,
            ) in TraceDataset.incremental_length_prefixes(
                traces=traces, max_length=max_length
            ):
                prefixes, next_activity = prefixes.to(device), next_activity.to(device)

                pred = model(prefixes, prefixes_lengths)
                loss = criterion(pred, next_activity)
                total_loss += loss.item()
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for metric in metrics.values():
                    metric.update(pred, next_activity)

        # Constraint loss
        if constraints:
            batch_logits = []
            for (
                prefixes,
                next_activity,
                prefixes_lengths,
            ) in TraceDataset.incremental_length_prefixes(
                traces=traces, max_length=max_length
            ):
                prefixes = prefixes.to(device)
                y_pred = model(prefixes, prefixes_lengths)
                batch_logits.append(y_pred)

            padded_batch_logits = pad_sequence(
                batch_logits, batch_first=True, padding_value=-torch.inf
            )

            padded_batch_logits[:, :, 0] = torch.where(
                padded_batch_logits[:, :, 0] == -torch.inf,
                1,
                padded_batch_logits[:, :, 0],
            )

            constraint_losses_tensors = {}
            for constraint, multiplier in zip(constraints, constraints_multipliers):
                constraint_loss = (
                    constraint(
                        padded_batch_logits,
                    )
                    * multiplier
                )
                constraint_losses[str(constraint)] += constraint_loss.item()
                constraint_losses_tensors[str(constraint)] = constraint_loss

            if train:
                optimizer.zero_grad()
                total_constraint_loss = (
                    sum(constraint_losses_tensors.values()) * constraints_multiplier
                )
                total_constraint_loss.backward()
                optimizer.step()

        total_samples += traces.size(0)

        loader.set_postfix(
            loss=total_loss / total_samples,
            **{name: metric.compute().item() for name, metric in metrics.items()},
            **{k: v / total_samples for k, v in constraint_losses.items()},
        )

    epoch_results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    constraints_results = {k: v / total_samples for k, v in constraint_losses.items()}
    return {"loss": total_loss / total_samples, **epoch_results, **constraints_results}


def test(
    *,
    model: NextActivityPredictor,
    criterion: torch.nn.Module,
    metrics: dict[str, torchmetrics.Metric],
    device: torch.device,
    loader: DataLoader,
    output_path: str | os.PathLike | pathlib.Path,
):
    loader = tqdm(
        loader,
        unit="batch",
        desc=f"Testing",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )
    model.eval()

    total_loss = 0
    total_samples = 0

    test_prefixes = []
    test_predictions = []
    test_next_activities = []

    predicted_traces: list[torch.Tensor] = []

    for traces, lengths in loader:
        max_length = lengths[0]

        batch_logits = []

        for (
            prefixes,
            next_activity,
            prefixes_lengths,
        ) in TraceDataset.incremental_length_prefixes(
            traces=traces, max_length=max_length
        ):
            prefixes, next_activity = prefixes.to(device), next_activity.to(device)

            pred = model(prefixes, prefixes_lengths)
            loss = criterion(pred, next_activity)
            total_loss += loss.item()

            y_pred = model(prefixes, prefixes_lengths)

            batch_logits.append(y_pred)

            test_prefixes.append(prefixes.cpu().numpy())
            test_predictions.append(y_pred.argmax(dim=-1).cpu().numpy())
            test_next_activities.append(next_activity.cpu().numpy())

            for metric in metrics.values():
                metric.update(pred, next_activity)

        padded_batch_logits = pad_sequence(
            batch_logits, batch_first=True, padding_value=-torch.inf
        )

        padded_batch_logits[:, :, 0] = torch.where(
            padded_batch_logits[:, :, 0] == -torch.inf,
            1,
            padded_batch_logits[:, :, 0],
        )

        predicted_traces.append(padded_batch_logits.argmax(dim=-1).T)

        total_samples += traces.size(0)

        loader.set_postfix(
            loss=total_loss / total_samples,
            **{name: metric.compute().item() for name, metric in metrics.items()},
        )

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "next_activity.csv", "w") as f:
        f.write("correct,prefix,y_hat,y\n")
        for (
            batch_test_prefixes,
            batch_test_predictions,
            batch_test_next_activities,
        ) in zip(test_prefixes, test_predictions, test_next_activities):
            for prefix, prediction, next_activity in zip(
                batch_test_prefixes, batch_test_predictions, batch_test_next_activities
            ):
                f.write(
                    f"{int(prediction == next_activity)},{np.array2string(prefix, max_line_width=np.inf)},{prediction},{next_activity}\n"
                )

    with open(output_path / "generated_traces.csv", "w") as f:
        f.write("generated_trace\n")
        for batch_predicted_traces in predicted_traces:
            for predicted_trace in batch_predicted_traces:
                f.write(
                    np.array2string(
                        np.trim_zeros(predicted_trace.cpu().numpy()),
                        max_line_width=np.inf,
                    )
                    + "\n"
                )

    epoch_results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    return {"loss": total_loss / total_samples, **epoch_results}


def test_generation(
    *,
    model: NextActivityPredictor,
    loader: DataLoader,
    device: torch.device,
    output: str | os.PathLike | pathlib.Path,
    fixed_length: int | None = None,
):
    MAX_PREDICTIONS_LENGTH = 1000

    loader = tqdm(
        loader,
        unit="batch",
        desc=f"Testing",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )
    model.eval()

    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for traces, lengths in loader:
        max_length = lengths[0] if fixed_length is None else fixed_length + 1
        min_length = 1 if fixed_length is None else fixed_length
        for (
            prefixes,
            _,
            prefixes_lengths,
            traces,
        ) in TraceDataset.incremental_length_prefixes(
            traces=traces,
            max_length=max_length,
            min_length=min_length,
            with_traces=True,
        ):
            prefixes = prefixes.to(device)

            for prefix, prefix_length, trace in zip(prefixes, prefixes_lengths, traces):
                suffix = []

                x = prefix.unsqueeze(0)
                x_length = prefix_length.unsqueeze(0)

                for __ in range(MAX_PREDICTIONS_LENGTH):
                    logits = model(x, x_length)
                    y_hat = logits.argmax(dim=-1)
                    suffix.append(y_hat)
                    if (
                        y_hat == model.vocab.activity2idx["<eos>"]
                        or y_hat == model.vocab.activity2idx["<pad>"]
                    ):
                        break
                    x = torch.cat([x, y_hat.unsqueeze(0)], dim=1)
                    x_length += 1

                # Remove padding (if any) from the trace
                zero_indices = (trace == 0).nonzero(as_tuple=True)[0]
                if len(zero_indices) > 0:  # If there is at least one zero
                    first_zero_index = zero_indices[0]
                    trace = trace[:first_zero_index]  # Keep only the left part

                records.append((prefix, torch.cat(suffix), trace))

    output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(
        records, columns=["prefix", "generated_suffix", "test_trace"]
    )
    results = results.map(lambda x: x.cpu().numpy())
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    results.to_csv(output, index=False)
