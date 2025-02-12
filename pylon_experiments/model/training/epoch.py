import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    *,
    epoch: int,
    epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metrics: dict[str, torchmetrics.Metric],
    device: torch.device,
    dataloader: DataLoader,
):
    loader = tqdm(
        dataloader,
        unit="batch",
        desc=f"Epoch {epoch+1}/{epochs} - train",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )

    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x, lengths)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

        for metric in metrics.values():
            metric.update(y_pred, y)
        loader.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
            **{name: metric.compute().item() for name, metric in metrics.items()},
        )

    epoch_results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    return {"loss": total_loss / total_samples, **epoch_results}


def val_epoch(
    *,
    epoch: int,
    epochs: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    metrics: dict[str, torchmetrics.Metric],
    device: torch.device,
    dataloader: DataLoader,
):
    loader = tqdm(
        dataloader,
        unit="batch",
        desc=f"Epoch {epoch+1}/{epochs} -   val",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )

    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x, lengths)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

            for metric in metrics.values():
                metric.update(y_pred, y)
            loader.set_postfix(
                val_loss=total_loss / total_samples,
                **{name: metric.compute().item() for name, metric in metrics.items()},
            )

    epoch_results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    return {"loss": total_loss / total_samples, **epoch_results}


def test_epoch(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    metrics: dict[str, torchmetrics.Metric],
    device: torch.device,
    dataloader: DataLoader,
):
    loader = tqdm(
        dataloader,
        unit="batch",
        desc=f"Test",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )

    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x, lengths)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

            for metric in metrics.values():
                metric.update(y_pred, y)
            loader.set_postfix(
                val_loss=total_loss / total_samples,
                **{name: metric.compute().item() for name, metric in metrics.items()},
            )

    epoch_results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    return {"loss": total_loss / total_samples, **epoch_results}


def train_with_constraints_epoch(
    *,
    epoch: int,
    epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metrics: dict[str, torchmetrics.Metric],
    constraints,
    device: torch.device,
    dataloader: DataLoader,
):
    loader = tqdm(
        dataloader,
        unit="batch",
        desc=f"Epoch {epoch+1}/{epochs} - train",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )

    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for x, y, lengths in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x, lengths)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

        for metric in metrics:
            metric.update(y_pred, y)
        loader.set_postfix(
            loss=total_loss / total_samples,
            **{name: metric.compute().item() for name, metric in metrics.items()},
        )

    epoch_results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    return {"loss": total_loss / total_samples, **epoch_results}
