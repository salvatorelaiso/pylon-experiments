import argparse
import random
from pathlib import Path

import pandas as pd
import pm4py

from pylon_experiments.data.preprocessing.args import Args
from pylon_experiments.data.trace_dataset import TraceDataset
from pylon_experiments.data.vocab import generate_vocab

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def df_to_event_log(df: pd.DataFrame):
    """
    Converts a pandas dataframe to a PM4Py event log.

    :param df: The dataframe to convert to an event log.
    :return: An event log constructed from the dataframe.
    :rtype: pm4py.objects.log.obj.EventLog
    """
    # Columns to use as event attributes
    case_id = "case:concept:name"
    activity_key = "concept:name"
    timestamp_key = "time:timestamp"

    # Format the dataframe to be used by PM4Py
    df = pm4py.format_dataframe(
        df, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key
    )[["case:concept:name", "concept:name", "time:timestamp"]]

    # Convert the dataframe to an event log
    return pm4py.convert_to_event_log(df)


def event_log_to_traces_dict(
    log: pm4py.objects.log.obj.EventLog,
) -> dict[any, list[str]]:
    """
    Convert an event log to a dictionary of traces.

    :param log: The event log to convert.
    :return: A dictionary with the case id as key and the sequence of activities as value.
    :rtype: Dict[Any, List[str]]
    """
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

    extracted_path = args.path / "extracted"
    datasets_path = args.path / "datasets"

    extracted_path.mkdir(parents=True, exist_ok=True)
    datasets_path.mkdir(parents=True, exist_ok=True)

    df = pm4py.read_xes(str(log_file.resolve()))
    event_log = df_to_event_log(df)
    traces = event_log_to_traces_dict(event_log)
    vocabulary = generate_vocab(df["concept:name"])

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
    for dataset, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
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

    print("Writing vocabulary files...")
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
    argparser.add_argument(
        "--log_sep",
        type=str,
        help="Separator used in the log file. (default: ';')",
        default=";",
    )

    args = argparser.parse_args()
    cwd = Path.cwd()
    return Args(path=cwd / args.path, log_sep=args.log_sep)


if __name__ == "__main__":
    args = parse_args()
    main(args)
