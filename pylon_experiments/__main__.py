import argparse
import csv
import datetime
import pathlib
import random

import numpy as np
import torch
import torchmetrics
from declare4pylon.relation.settings import RelationConstraintSettings
from declare4pylon.relation.succession import AlternateSuccessionConstraint
from pylon.sampling_solver import WeightedSamplingSolver

from pylon_experiments.args import Args
from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.data.loader import Loader
from pylon_experiments.data.vocab import Vocab
from pylon_experiments.model.metrics.damerau_levehenstein import (
    DamerauLevenshteinDistance,
)
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

    epoch_offset = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = Loader(args=args.loader_args).get_loaders()
    vocab: Vocab = Vocab.load(args.model_args.vocab_path)
    model = NextActivityPredictor(args=args.model_args).to(device)
    print("Model:", model)

    constraints = [constraint_from_string(c, vocab) for c in args.constraints]
    print(constraints)

    run_folder_name = f"{datetime.datetime.now(datetime.UTC).strftime("%Y%m%d.%H%M")}.{'no_constraint' if not constraints else 'constraints'}"
    run_path = (
        pathlib.Path(__file__).parents[1].resolve()
        / "runs"
        / args.loader_args.dataset_path.name
        / run_folder_name
    )
    run_path.mkdir(parents=True, exist_ok=True)

    args.dump_args(run_path / "args.json")

    output = train(
        epochs=args.epochs,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        criterion=torch.nn.CrossEntropyLoss(),
        constraints=constraints,
        metrics={
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=len(vocab)
            ).to(device),
        },
        model=model,
        device=device,
        epoch_offset=epoch_offset,
    )

    torch.save(model.state_dict(), run_path / f"model.epoch_{args.epochs}.pth")
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

    # Loader args
    argparser.add_argument(
        "--path",
        type=str,
        help="Path to the folder specific to the dataset. Expected to contain a folder 'datasets' with the files 'train.pkl', 'val.pkl' and 'test.pkl'.",
        required=True,
    )

    # Model args
    argparser.add_argument(
        "--hidden_size",
        type=int,
        help="Size of the hidden state of the LSTM. (default: 128)",
        default=128,
    )
    argparser.add_argument(
        "--num_layers",
        type=int,
        help="Number of layers in the LSTM. (default: 1)",
        default=1,
    )
    argparser.add_argument(
        "--embedding_dim",
        type=int,
        help="Size of the embedding layer. (default: 128)",
        default=128,
    )
    argparser.add_argument(
        "--constraints",
        nargs="*",
        default=[],
        type=str,
        help="Constraints",
    )

    args = argparser.parse_args()
    cwd = pathlib.Path.cwd()

    return Args(
        seed=args.seed,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        loader_args=LoaderArgs(dataset_path=cwd / args.path),
        model_args=ModelArgs(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            embedding_dim=args.embedding_dim,
            vocab_path=cwd / args.path / "extracted" / "vocab.pkl",
        ),
        constraints=args.constraints,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
