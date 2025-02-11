import torch
from torch.utils.data import DataLoader


def train_epoch(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
):
    model.train()
    total_loss = 0
    for i, (x, y, lengths) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x, lengths)
        loss = criterion(y_pred, y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")
    return total_loss / len(loader)


def val_epoch(
    *,
    epoch: int,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
):
    model.eval()
    total_loss = 0
    for i, (x, y, lengths) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x, lengths)
        loss = criterion(y_pred, y)
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch}, batch {i}, val_loss: {loss.item()}")
    return total_loss / len(loader)


def test_epoch(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (x, y, lengths) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x, lengths)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
        return total_loss / len(loader)
