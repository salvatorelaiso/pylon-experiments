import argparse
import pathlib
import random

import numpy as np
import torch
from pydantic.dataclasses import dataclass

from pylon_experiments.args import Args
from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.data.loader import Loader
from pylon_experiments.model.model import Args as ModelArgs
from pylon_experiments.model.model import NextActivityPredictor
from pylon_experiments.model.training.train import train


def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = Loader(args=args.loader_args).get_loaders()
    model = NextActivityPredictor(args=args.model_args).to(device)

    train(
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        criterion=torch.nn.CrossEntropyLoss(),
        model=model,
        device=device,
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
        "--learning_rate",
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
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
