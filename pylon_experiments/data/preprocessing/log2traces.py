import argparse
import random
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.objects.log.util.sorting import sort_timestamp_log

from pylon_experiments.data.preprocessing.args import Args
from pylon_experiments.data.trace_dataset import TraceDataset
from pylon_experiments.data.vocab import generate_vocab

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def df_to_sorted_event_log(df: pd.DataFrame) -> pm4py.objects.log.obj.EventLog:
    df = df[["case:concept:name", "concept:name", "time:timestamp"]]
    return sort_timestamp_log(df)


def event_log_to_traces_dict(
    log: pm4py.objects.log.obj.EventLog, trace_percentage: float
) -> dict[any, list[str]]:
    """
    Convert an event log to a dictionary of traces.

    :param log: The event log to convert.
    :param trace_percentage: The percentage of traces to use from the log.
    :return: A dictionary with the case id as key and the sequence of activities as value.
    :rtype: Dict[Any, List[str]]
    """
    n_traces = int(len(log) * (trace_percentage))
    starting_index = len(log) - n_traces
    log = log[starting_index:]

    traces = {}
    for trace in log:
        # Get the case id and the sequence of activities
        case_id = trace.attributes["concept:name"]
        traces[case_id] = [event["concept:name"] for event in trace]

    return traces


def get_xes_gz_files(path: Path):
    return list(path.glob("*.xes.gz"))


def main(args: Args):
    xes_gz_files = get_xes_gz_files(args.path / "origin")
    if not xes_gz_files:
        raise ValueError(f"No XES.GZ files found in {args.path}")
    if len(xes_gz_files) > 1:
        raise ValueError(f"Multiple XES.GZ files found in {args.path}")
    log_file = xes_gz_files[0]

    df = pm4py.read_xes(str(log_file.resolve()))
    vocabulary = generate_vocab(df["concept:name"])
    event_log = df_to_sorted_event_log(df)

    for trace_percentage in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"Extracting traces with {trace_percentage * 100}% of the log...")

        traces = event_log_to_traces_dict(event_log, trace_percentage)

        extracted_path = args.path / f"{trace_percentage:.0%}" / "extracted"
        datasets_path = args.path / f"{trace_percentage:.0%}" / "datasets"

        extracted_path.mkdir(parents=True, exist_ok=True)
        datasets_path.mkdir(parents=True, exist_ok=True)

        # Write a CSV file with the all the traces
        print("Writing traces file...")
        with open(extracted_path / "traces.csv", "w") as f:
            f.write("trace_id;activity_ids\n")
            for trace_id, trace_activities in traces.items():
                trace_activities = vocabulary.to_idxs(trace_activities)
                f.write(trace_id + ";" + ",".join(map(str, trace_activities)) + "\n")

        # Split the dataset into training, validation, and test sets based on the trace ids
        print("Splitting dataset...")
        keys = list(traces.keys())
        # Shuffle the keys to split the dataset randomly
        random.seed(42)
        random.shuffle(keys)
        train_ids = keys[: int(len(keys) * TRAIN_RATIO)]
        val_ids = keys[
            int(len(keys) * TRAIN_RATIO) : int(len(keys) * (TRAIN_RATIO + VAL_RATIO))
        ]
        test_ids = keys[int(len(keys) * (TRAIN_RATIO + VAL_RATIO)) :]

        # Create and save the datasets
        print("Writing datasets...")
        for dataset, ids in zip(
            ["train", "val", "test"], [train_ids, val_ids, test_ids]
        ):
            dataset_traces = [
                vocabulary.to_idxs(["<sos>"] + traces[trace_id] + ["<eos>"])
                for trace_id in ids
            ]

            # Write the dataset file as a CSV file for easy reading
            csv_path = datasets_path / f"{dataset}.csv"
            with open(csv_path, "w") as f:
                for trace in dataset_traces:
                    f.write(",".join(map(str, trace)) + "\n")

            # Create a TraceDataset object and save it as a pickle file
            trace_dataset = TraceDataset(dataset_traces)
            trace_dataset.save(datasets_path / f"{dataset}.traces.pkl")

        print()

    print("Writing vocabulary files...")
    extracted_path = args.path / "extracted"
    extracted_path.mkdir(parents=True, exist_ok=True)
    vocabulary.save(extracted_path / "vocab.pkl")
    vocabulary.to_csv(extracted_path / "vocab.csv")
    print("Done!")


def parse_args() -> Args:
    argparser = argparse.ArgumentParser(
        prog="log2traces",
        description="Extract traces from a CSV log file and split them into training, validation, and test sets.",
    )
    argparser.add_argument(
        "--path",
        type=str,
        help="Path to the folder specific to the dataset. Expected to contain a folder 'origin' with the log file in xes.gz format.",
        required=True,
    )

    args = argparser.parse_args()
    cwd = Path.cwd()
    return Args(path=cwd / args.path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
