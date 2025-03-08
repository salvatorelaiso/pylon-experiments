import torch
import torchmetrics
from declare4pylon.constraint import DeclareConstraint
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader

from pylon_experiments.model.model import NextActivityPredictor
from pylon_experiments.model.training.epoch import train_with_constraints_epoch


class Config:
    arbitrary_types_allowed = True


@dataclass(frozen=True, kw_only=True, config=Config)
class Args:
    epochs: int = 100
    learning_rate: float = 0.001

    train_loader = DataLoader
    val_loader = DataLoader
    test_loader = DataLoader
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    model: NextActivityPredictor
    device: torch.device


@dataclass(frozen=True, kw_only=True, config=Config)
class TrainOutput:
    history: dict
    best_loss_model: torch.nn.Module
    best_acc_model: torch.nn.Module
    best_loss_epoch: int
    best_acc_epoch: int


def train(
    *,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    constraints: list[DeclareConstraint],
    constraints_multipliers: list[float],
    constraints_multiplier: float,
    metrics: dict[str, torchmetrics.Metric],
    model: NextActivityPredictor,
    ignore_task_loss: bool,
    device: torch.device,
):
    history = {"train": {}, "val": {}}

    history["train"]["loss"] = []
    history["train"].update({name: [] for name in metrics.keys()})

    history["val"]["loss"] = []
    history["val"].update({name: [] for name in metrics.keys()})

    if constraints:
        history["train"].update({str(constraint): [] for constraint in constraints})
        history["val"].update({str(constraint): [] for constraint in constraints})

    best_val_loss = float("inf")
    best_val_loss_model = None
    best_val_loss_epoch = None

    best_val_acc = float("-inf")
    best_val_acc_model = None
    best_val_acc_epoch = None

    for epoch in range(epochs):
        train_results = train_with_constraints_epoch(
            epoch=epoch,
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            constraints=constraints,
            constraints_multipliers=constraints_multipliers,
            constraints_multiplier=constraints_multiplier,
            metrics=metrics,
            device=device,
            dataloader=train_loader,
            ignore_task_loss=ignore_task_loss,
        )

        val_results = train_with_constraints_epoch(
            epoch=epoch,
            epochs=epochs,
            model=model,
            optimizer=None,
            criterion=criterion,
            constraints=constraints,
            constraints_multipliers=constraints_multipliers,
            metrics=metrics,
            device=device,
            dataloader=val_loader,
            mode="val",
        )

        train_loss = train_results["loss"]
        val_loss = val_results["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_model = model
            best_val_loss_epoch = epoch + 1

        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            best_val_acc_model = model
            best_val_acc_epoch = epoch + 1

        history["train"]["loss"].append(train_loss)
        history["train"].update(
            {
                name: history["train"][name] + [train_results[name]]
                for name in metrics.keys()
            }
        )
        history["train"].update(
            {
                str(constraint): history["train"][str(constraint)]
                + [train_results[str(constraint)]]
                for constraint in constraints
            }
        )
        history["val"]["loss"].append(val_loss)
        history["val"].update(
            {
                name: history["val"][name] + [val_results[name]]
                for name in metrics.keys()
            }
        )
        history["val"].update(
            {
                str(constraint): history["val"][str(constraint)]
                + [val_results[str(constraint)]]
                for constraint in constraints
            }
        )

    return TrainOutput(
        history=history,
        best_loss_model=best_val_loss_model,
        best_acc_model=best_val_acc_model,
        best_loss_epoch=best_val_loss_epoch,
        best_acc_epoch=best_val_acc_epoch,
    )
