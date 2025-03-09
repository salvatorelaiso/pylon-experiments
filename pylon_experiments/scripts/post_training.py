import argparse
import csv
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torchmetrics
from declare4pylon.constraint import DeclareConstraint
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.data.loader import Loader
from pylon_experiments.model.metrics.damerau_levehenstein import (
    DamerauLevenshteinDistance,
    NormalizedDamerauLevenshteinDistance,
    WeightedNormalizedDamerauLevenshteinDistance,
)
from pylon_experiments.model.training.constraints.constraint import (
    constraint_from_string,
)
from pylon_experiments.model.training.epoch import test, test_generation
from pylon_experiments.scripts.args import Args


def convert_to_list(s):
    return list(map(int, s.strip("[]").split()))


def convert_constraint_line(line: str, model: torch.nn.Module):
    constraint_str, constraint_support = line.split(" - Support: ")

    constraint, _ = constraint_from_string(constraint_str, model.vocab)
    constraint_type_str, remaining = constraint_str.split("[")
    constraint_activities = remaining.removesuffix("]").split(",")
    constraint_activities = [
        constraint_activity.strip()
        + "("
        + str(model.vocab.to_idx(constraint_activity.strip()))
        + ")"
        for constraint_activity in constraint_activities
    ]
    constraint_str = constraint_type_str + "[" + ", ".join(constraint_activities) + "]"

    constraint_support = constraint_support.strip().removesuffix("%")
    constraint_support = float(constraint_support) / 100

    return (
        constraint_str,
        constraint,
        constraint_support,
    )


def evaluate_constraints_on_traces(
    *,
    traces_file: pathlib.Path,
    constraints: dict[str, tuple[DeclareConstraint, float]],
    output: pathlib.Path,
):
    traces = pd.read_csv(traces_file)
    traces = traces.map(lambda x: torch.tensor(convert_to_list(x)))

    records = []
    for i in range(len(traces)):
        record = {
            "trace": traces.iloc[i]["generated_trace"].cpu().numpy(),
        }
        for constraint_str, (constraint, _) in constraints.items():
            sat = constraint._condition(
                traces.iloc[i]["generated_trace"].unsqueeze(0),
                **constraint._settings.dict(),
            ).item()
            record[constraint_str] = int(sat)
        records.append(record)

    constraints_results = pd.DataFrame.from_records(records)
    constraints_results.to_csv(output, index=False)


def evaluate_constraints_on_predictions(
    *,
    predictions_file: pathlib.Path,
    constraints: dict[str, tuple[DeclareConstraint, float]],
    output: pathlib.Path,
):
    predictions = pd.read_csv(predictions_file)
    predictions = predictions.map(lambda x: torch.tensor(convert_to_list(x)))

    records = []
    for i in range(len(predictions)):
        record = {
            "prefix": predictions.iloc[i]["prefix"].cpu().numpy(),
            "generated_suffix": predictions.iloc[i]["generated_suffix"].cpu().numpy(),
        }
        for constraint_str, (constraint, _) in constraints.items():
            sat = constraint._condition(
                predictions.iloc[i]["generated_suffix"].unsqueeze(0),
                prefixes=predictions.iloc[i]["prefix"].unsqueeze(0),
                **constraint._settings.dict(),
            ).item()
            record[constraint_str] = int(sat)
        records.append(record)

    constraints_results = pd.DataFrame.from_records(records)
    constraints_results.to_csv(output, index=False)


def evaluate_constraints_satisfaction_rate(
    *,
    results_file: pathlib.Path,
    constraints: dict[str, tuple[DeclareConstraint, float]],
    output: pathlib.Path,
):
    ignore = ["trace", "prefix", "generated_suffix"]
    constraints_results = pd.read_csv(results_file)
    satisfaction_results = constraints_results.loc[
        :, [c for c in constraints_results.columns if c not in ignore]
    ].mean()

    satisfaction_results = (
        satisfaction_results.to_frame()
        .reset_index()
        .rename(columns={"index": "constraint", 0: "satisfaction"})
    )
    satisfaction_results["support"] = satisfaction_results["constraint"].apply(
        lambda x: constraints[x][1]
    )

    satisfaction_results = satisfaction_results.sort_values(
        by="support", ascending=False
    )

    satisfaction_results[["constraint", "support", "satisfaction"]].to_csv(
        output, index=False
    )


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


def main(args: Args):
    args.path

    history_files = [file for file in args.path.rglob("history.csv")]

    # Draw the plots for each history file
    for history_file in history_files:
        print(f"Converting {history_file} to tensorboard format.")
        log_dir = history_file.parent / "tensorboard"
        log_dir.mkdir(exist_ok=True)
        convert_csv_to_tensorboard(history_file, log_dir)
        plots_dir = history_file.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        plots(history_file, plots_dir)

    models = [file for file in args.path.rglob("*.best_val_acc.pth")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_path in models:
        dataset = model_path.parents[2].name
        percentage = model_path.parents[1].name
        model = torch.load(model_path, weights_only=False).to(device)
        test_loader = Loader(
            args=LoaderArgs(dataset_path=pathlib.Path("./data") / dataset / percentage)
        ).get_loaders()["test"]
        print(
            f"Testing model {model_path} trained on {percentage} traces of {dataset}."
        )
        print(model)
        # Read the constraints from the file
        constraints: dict[str, tuple[any, float]] = {}
        constraint_file = (
            pathlib.Path("./data/") / dataset / "extracted" / "constraints.txt"
        )
        with open(constraint_file) as f:
            lines = f.readlines()
            for line in lines:
                constraint_str, constraint, support = convert_constraint_line(
                    line, model
                )
                constraints[constraint_str] = (constraint, support)
        print(constraints)

        # Create the output directory for the test results
        output_path = model_path.parent / model_path.stem
        output_path.mkdir(parents=True, exist_ok=True)

        # Test the model without teacher forcing
        test_generation(
            model=model,
            device=device,
            loader=test_loader,
            output=output_path / "generation" / "predictions.csv",
        )
        evaluate_constraints_on_predictions(
            predictions_file=output_path / "generation" / "predictions.csv",
            constraints=constraints,
            output=output_path / "generation" / "constraints_satisfactions.csv",
        )
        evaluate_constraints_satisfaction_rate(
            results_file=output_path / "generation" / "constraints_satisfactions.csv",
            constraints=constraints,
            output=output_path / "generation" / "constraints_satisfaction_rate.csv",
        )

        # Test the model without teacher forcing with fixed length prefixes
        fixed_length = 6
        test_generation(
            model=model,
            device=device,
            loader=test_loader,
            output=output_path
            / f"fixed_length_prefixes_{fixed_length}"
            / "predictions.csv",
            fixed_length=fixed_length,
        )
        evaluate_constraints_on_predictions(
            predictions_file=output_path
            / f"fixed_length_prefixes_{fixed_length}"
            / "predictions.csv",
            constraints=constraints,
            output=output_path
            / f"fixed_length_prefixes_{fixed_length}"
            / "constraints_satisfactions.csv",
        )
        evaluate_constraints_satisfaction_rate(
            results_file=output_path
            / f"fixed_length_prefixes_{fixed_length}"
            / "constraints_satisfactions.csv",
            constraints=constraints,
            output=output_path
            / f"fixed_length_prefixes_{fixed_length}"
            / "constraints_satisfaction_rate.csv",
        )

        # Evaluate the model with the same approach as the training
        metrics = {
            "acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=len(model.vocab)
            ).to(device),
            "dld": DamerauLevenshteinDistance(),
            "norm_dld": NormalizedDamerauLevenshteinDistance(),
            "weighted_norm_dld": WeightedNormalizedDamerauLevenshteinDistance(),
        }
        epoch_results = test(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            metrics=metrics,
            device=device,
            loader=test_loader,
            output_path=output_path / "step_by_step",
        )
        with open(output_path / "step_by_step" / "results.json", "w") as f:
            json.dump(epoch_results, f, indent=4)

        # Read the predicted traces and evaluate the constraints
        evaluate_constraints_on_traces(
            traces_file=output_path / "step_by_step" / "generated_traces.csv",
            constraints=constraints,
            output=output_path / "step_by_step" / "constraints_satisfactions.csv",
        )
        evaluate_constraints_satisfaction_rate(
            results_file=output_path / "step_by_step" / "constraints_satisfactions.csv",
            constraints=constraints,
            output=output_path / "step_by_step" / "constraints_satisfaction_rate.csv",
        )


def parse_args() -> Args:
    argparser = argparse.ArgumentParser(
        prog="post_training",
        description="Evaluate the models.",
    )
    argparser.add_argument(
        "--path",
        type=str,
        help="Base path where to find the models. (default: 'runs')",
        required=False,
        default="runs",
    )

    args = argparser.parse_args()
    cwd = pathlib.Path.cwd()
    return Args(path=cwd / args.path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
