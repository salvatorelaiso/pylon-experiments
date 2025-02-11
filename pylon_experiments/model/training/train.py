from dataclasses import asdict

import torch
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader

from pylon_experiments.model.training.epoch import test_epoch, train_epoch, val_epoch


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
    model: torch.nn.Module
    device: torch.device

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


def train(
    *,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    model: torch.nn.Module,
    device: torch.device,
):
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            loader=train_loader,
        )
        val_loss = val_epoch(
            epoch=epoch,
            model=model,
            criterion=criterion,
            device=device,
            loader=val_loader,
        )
