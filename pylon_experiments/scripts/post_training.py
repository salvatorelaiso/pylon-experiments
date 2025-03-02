import csv
import json
import pathlib

import numpy as np
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
from pylon_experiments.model.training.constraints.constraint import (
    constraint_from_string,
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
        dataset = model_path.parents[2].name
        percentage = model_path.parents[1].name
        print(
            f"Testing model {model_path} trained on {percentage} traces of {dataset}."
        )
        # Load the model and test it on the test traces
        model = torch.load(model_path, weights_only=False).to(device)
        print(model)

        test_loader = Loader(
            args=LoaderArgs(dataset_path=pathlib.Path("./data") / dataset / percentage)
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
        epoch_results = test(
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
            json.dump(epoch_results, f, indent=4)

        # Test how many constraints are satisfied
        constraint_file = (
            pathlib.Path("./data/") / dataset / "extracted" / "constraints.txt"
        )

        constraints: dict[str, tuple[any, float]] = {}
        with open(constraint_file) as f:
            lines = f.readlines()
            for line in lines:
                constraint_str, constraint_support = line.split(" - Support: ")

                constraint = constraint_from_string(constraint_str, model.vocab)
                constraint_type_str, remaining = constraint_str.split("[")
                constraint_activities = remaining.removesuffix("]").split(",")
                constraint_activities = [
                    constraint_activity.strip()
                    + "("
                    + str(model.vocab.to_idx(constraint_activity.strip()))
                    + ")"
                    for constraint_activity in constraint_activities
                ]
                constraint_str = (
                    constraint_type_str + "[" + ", ".join(constraint_activities) + "]"
                )

                constraint_support = constraint_support.strip().removesuffix("%")
                constraint_support = float(constraint_support)

                constraints[constraint_str] = (
                    constraint,
                    constraint_support,
                )

        with open(output_path / "predicted_traces.txt") as f:
            keys = [
                f"{constraint_str} {support}%"
                for constraint_str, (constraint, support) in constraints.items()
            ]

            constraints_results = pd.DataFrame(columns=["trace"] + keys)

            lines = f.readlines()
            for line in lines:
                line = line.removeprefix("[").removesuffix("]\n")
                np_trace = np.fromstring(line, dtype=int, sep=" ")
                trace = torch.tensor(np_trace).unsqueeze(0).to(device)

                row = [f"{np_trace.tolist()}"]

                for constraint_str, (constraint, support) in constraints.items():
                    sat = constraint._condition(
                        trace, **constraint._settings.dict()
                    ).item()

                    row.append(sat)

                constraints_results.loc[len(constraints_results)] = row

        # Write which constraints are satisfied
        constraints_results.to_csv(
            output_path / "constraints_satisfactions.csv", index=False
        )
        # Compute the percentage of True values for each column (except the column "trace")
        satisfaction_results = constraints_results.loc[
            :, [c for c in constraints_results.columns if c != "trace"]
        ].mean()
        satisfaction_results = (
            satisfaction_results.to_frame()
            .reset_index()
            .rename(columns={"index": "constraint", 0: "satisfaction"})
        )
        satisfaction_results["support"] = satisfaction_results["constraint"].apply(
            lambda x: float(x.split("] ")[1].removesuffix("%")) / 100
        )
        satisfaction_results["constraint"] = satisfaction_results["constraint"].apply(
            lambda x: x.split("] ")[0]
        )
        satisfaction_results.to_csv(
            output_path / "constraints_satisfaction_rate.csv",
            columns=["constraint", "support", "satisfaction"],
            index=False,
        )


if __name__ == "__main__":
    main()
