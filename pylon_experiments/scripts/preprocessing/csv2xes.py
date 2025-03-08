import argparse
import gzip
from pathlib import Path

import pandas as pd
import pm4py
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer

from pylon_experiments.scripts.args import Args


def main(args: Args):
    df = pd.read_csv(args.path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    event_log = pm4py.format_dataframe(
        df, case_id="case", activity_key="activity", timestamp_key="timestamp"
    )
    print(pm4py.get_start_activities(event_log))
    xes_exporter.apply(event_log, "output.xes")

    with open("output.xes", "rb") as f_in:
        with gzip.open("output.xes.gz", "wb") as f_out:
            f_out.writelines(f_in)


def parse_args() -> Args:
    argparser = argparse.ArgumentParser(
        prog="csv2xes",
        description="Convert log from CSV to XES format and compresses it with GZIP.",
    )
    argparser.add_argument(
        "--path",
        type=str,
        help="Path to the csv file.",
        required=True,
    )

    args = argparser.parse_args()
    cwd = Path.cwd()
    return Args(path=cwd / args.path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
