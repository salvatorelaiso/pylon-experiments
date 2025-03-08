import argparse
import csv
import datetime
import pathlib
import random
import time

import numpy as np
import torch
import torchmetrics

from pylon_experiments.args import Args
from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.data.loader import Loader
from pylon_experiments.data.vocab import Vocab
from pylon_experiments.model.model import Args as ModelArgs
from pylon_experiments.model.model import NextActivityPredictor
from pylon_experiments.model.training.constraints.constraint import (
    constraint_from_string,
)
from pylon_experiments.model.training.train import train


def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_offset = 0

    if args.base_model:
        model = torch.load(args.base_model, weights_only=False).to(device)
        vocab: Vocab = model.vocab
        epoch_offset = int(args.base_model.stem.split("_")[-1])
        print(f"Loaded model from {args.base_model} with {epoch_offset} epochs.")
    else:
        model = NextActivityPredictor(args=args.model_args).to(device)
        vocab: Vocab = Vocab.load(args.model_args.vocab_path)
        print("Model:", model)

    loaders = Loader(args=args.loader_args).get_loaders()

    constraints = [constraint_from_string(c, vocab) for c in args.constraints]
    print("Constraints:", constraints)

    while True:
        try:
            run_folder_name = f"{datetime.datetime.now(datetime.UTC).strftime("%Y%m%d.%H%M")}.{'no_constraint' if not constraints else 'constraints'}"
            run_path = (
                pathlib.Path(__file__).parents[1].resolve()
                / "runs"
                / args.loader_args.dataset_path.parent.name
                / args.loader_args.dataset_path.name
                / run_folder_name
            )
            run_path.mkdir(
                parents=True, exist_ok=False
            )  # Exist_ok=False to avoid overwriting existing runs (e.g. when the previous ended in the same minute as the current one)
            break
        except FileExistsError:
            continue

    args.dump_args(run_path / "args.json")
    args.print_args()

    start_time = time.perf_counter()
    output = train(
        epochs=args.epochs,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        criterion=torch.nn.CrossEntropyLoss(),
        constraints=[c[0] for c in constraints],
        constraints_multipliers=[c[1] for c in constraints],
        constraints_multiplier=args.constraints_multiplier,
        metrics={
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=len(vocab)
            ).to(device),
        },
        model=model,
        ignore_task_loss=args.ignore_task_loss,
        device=device,
    )
    end_time = time.perf_counter()
    with open(run_path / "elapsed_time.txt", "w") as file:
        file.write(
            f"{int(end_time - start_time)} seconds ({(end_time - start_time) / 60:.2f} minutes)"
        )

    torch.save(model, run_path / f"model.epoch_{args.epochs}.pth")
    torch.save(
        output.best_acc_model,
        run_path / f"model.epoch_{output.best_acc_epoch}.best_val_acc.pth",
    )
    torch.save(
        output.best_loss_model,
        run_path / f"model.epoch_{output.best_loss_epoch}.best_val_loss.pth",
    )

    # Save the training history in a csv file
    csv_path = run_path / "history.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        train_keys = output.history["train"].keys()
        val_keys = output.history["val"].keys()
        writer.writerow(
            ["epoch"]
            + [f"train_{key}" for key in train_keys]
            + [f"val_{key}" for key in val_keys]
        )
        for epoch in range(args.epochs):
            writer.writerow(
                [epoch + 1 + epoch_offset]
                + [output.history["train"][key][epoch] for key in train_keys]
                + [output.history["val"][key][epoch] for key in val_keys]
            )


def parse_args():
    argparser = argparse.ArgumentParser()

    # main args
    argparser.add_argument(
        "--seed",
        type=int,
        help="Seed for random number generators. (default: 42)",
        default=42,
    )

    argparser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train the model. (default: 100)",
        default=100,
    )

    argparser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for the optimizer. (default: 0.001)",
        default=0.001,
    )

    argparser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate for the LSTM. (default: 0.0)",
        default=0.0,
    )

    # Loader args
    argparser.add_argument(
        "--path",
        type=str,
        help="Path to the folder specific to the dataset. Expected to contain a folder 'datasets' with the files 'train.pkl', 'val.pkl' and 'test.pkl'.",
        required=True,
    )

    # Model args
    argparser.add_argument(
        "--hidden-size",
        type=int,
        help="Size of the hidden state of the LSTM. (default: 128)",
        default=128,
    )
    argparser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers in the LSTM. (default: 1)",
        default=1,
    )
    argparser.add_argument(
        "--embedding-dim",
        type=int,
        help="Size of the embedding layer. (default: 128)",
        default=128,
    )
    argparser.add_argument(
        "--constraints",
        nargs="*",
        default=[],
        type=str,
        help="List of Declare constraints to use during training. (default: [])",
    )
    argparser.add_argument(
        "--constraints-multiplier",
        type=float,
        help="Training multiplier for the constraints loss. (default: 1.0)",
        default=1.0,
    )
    argparser.add_argument(
        "--base-model",
        type=str,
        help="Path to a base model to start training from. (default: None)",
        default=None,
        required=False,
    )
    argparser.add_argument(
        "--ignore-task-loss",
        help="Train the model without the task loss. (default: False)",
        action="store_true",
    )

    args = argparser.parse_args()
    cwd = pathlib.Path.cwd()

    return Args(
        seed=args.seed,
        learning_rate=args.learning_rate,
        constraints_multiplier=args.constraints_multiplier,
        epochs=args.epochs,
        loader_args=LoaderArgs(dataset_path=cwd / args.path),
        model_args=ModelArgs(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            vocab_path=cwd / args.path / ".." / "extracted" / "vocab.pkl",
        ),
        constraints=args.constraints,
        base_model=args.base_model,
        ignore_task_loss=args.ignore_task_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
