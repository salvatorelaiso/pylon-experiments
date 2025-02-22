import csv
import pathlib

from tensorboardX import SummaryWriter


def convert_csv_to_tensorboard(csv_file, log_dir):
    writer = SummaryWriter(log_dir)

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, float(value), epoch)

    writer.close()


def main():
    history_files = [file for file in pathlib.Path("runs").rglob("history.csv")]

    for history_file in history_files:
        print(f"Converting {history_file} to tensorboard format.")
        log_dir = history_file.parent / "tensorboard"
        log_dir.mkdir(exist_ok=True)
        convert_csv_to_tensorboard(history_file, log_dir)


if __name__ == "__main__":
    main()
