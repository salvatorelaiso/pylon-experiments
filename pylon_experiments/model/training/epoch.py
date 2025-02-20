from itertools import zip_longest
from typing import Literal

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
    constraints,
    metrics: dict[str, torchmetrics.Metric],
    device: torch.device,
    dataloader: DataLoader,
    mode: Literal["train", "val", "test"] = "train",
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
            for constraint in constraints:
                constraint_loss = constraint(
                    padded_batch_logits,
                )
                constraint_losses[str(constraint)] += constraint_loss.item()
                constraint_losses_tensors[str(constraint)] = constraint_loss

            if train:
                optimizer.zero_grad()
                total_constraint_loss = sum(constraint_losses_tensors.values())
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
