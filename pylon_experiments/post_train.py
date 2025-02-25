import csv
import json
import pathlib

import pandas as pd
import torch
import torchmetrics
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.data.loader import Loader
from pylon_experiments.model.metrics.damerau_levehenstein import (
    DamerauLevenshteinDistance,
    NormalizedDamerauLevenshteinDistance,
)
from pylon_experiments.model.training.epoch import test


def convert_csv_to_tensorboard(csv_file: pathlib.Path, log_dir: pathlib.Path):
    is_empty = not any(log_dir.iterdir())
    if not is_empty:
        print(
            f"The folder {log_dir} is not empty.\nSkipping the tensorboard summary file generation..."
        )
        return
    writer = SummaryWriter(log_dir)

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, float(value), epoch)

    writer.close()


def plots(csv_file, plots_dir):
    history = pd.read_csv(csv_file)
    keys = [key.removeprefix("train_").removeprefix("val_") for key in history.keys()]
    for key in keys:
        if key == "epoch":
            continue
        plt.plot(
            history["epoch"],
            history["train_" + key],
            label="train_" + key,
        )
        plt.plot(
            history["epoch"],
            history["val_" + key],
            label="val_" + key,
        )
        plt.legend(bbox_to_anchor=(1, 0), loc="lower left")
        plt.savefig(plots_dir / f"{key}.png", bbox_inches="tight")
        plt.clf()


def main():
    history_files = [file for file in pathlib.Path("runs").rglob("history.csv")]

    for history_file in history_files:
        print(f"Converting {history_file} to tensorboard format.")
        log_dir = history_file.parent / "tensorboard"
        log_dir.mkdir(exist_ok=True)
        convert_csv_to_tensorboard(history_file, log_dir)
        plots_dir = history_file.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plots(history_file, plots_dir)

    models = [file for file in pathlib.Path("runs").rglob("*.best_val_acc.pth")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_path in models:
        dataset = model_path.parents[1].name
        print(
            f"Testing model {model_path} trained on dataset {dataset} on test traces."
        )
        # Load the model and test it on the test traces
        model = torch.load(model_path, weights_only=False).to(device)
        print(model)

        test_loader = Loader(
            args=LoaderArgs(dataset_path=pathlib.Path("./data") / dataset)
        ).get_loaders()["test"]

        metrics = {
            "acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=len(model.vocab)
            ).to(device),
            "dld": DamerauLevenshteinDistance(),
            "norm_dld": NormalizedDamerauLevenshteinDistance(),
        }

        output_path = model_path.parent / model_path.stem
        output_path.mkdir(parents=True, exist_ok=True)
        results = test(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            constraints=[],
            metrics=metrics,
            device=device,
            loader=test_loader,
            output_path=output_path,
        )
        # Save the json results
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
