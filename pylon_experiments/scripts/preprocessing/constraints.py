import argparse
import re
from pathlib import Path

from .args import Args


def get_files_path(path):
    return path / "origin"


def get_files(path):
    return (
        get_files_path(path) / "constraints.declare",
        get_files_path(path) / "constraints.txt",
    )


def is_declare_constraint(line):
    return re.match(r"(.*)\[(.*)\](.*)", line) is not None


def get_declare_constraint(line):
    return line.split("]")[0] + "]"


def get_support_percentage(line):
    return float(re.match(r"\d+\) In (.*)% of traces in the log, (.*)", line).group(1))


def get_constraints_with_support(declare_file, txt_file):
    with declare_file.open("r") as declare, txt_file.open("r") as textual:
        declare_constraints = []
        for line in declare:
            if is_declare_constraint(line):
                declare_constraints.append(get_declare_constraint(line))

        support_percentages = []
        skip = True
        for line in textual:
            if line == "Constraints:\n":
                skip = False
                continue
            if skip:
                continue
            support_percentages.append(get_support_percentage(line))

        assert len(declare_constraints) == len(support_percentages)
    for i in range(len(declare_constraints)):
        yield declare_constraints[i], support_percentages[i]


def main(args: Args):
    declare_file, txt_file = get_files(args.path)
    constraints = list(get_constraints_with_support(declare_file, txt_file))
    for constraint, support in constraints:
        print(f"{constraint} - Support: {support:.2f}%")
    print(f"Total constraints: {len(constraints)}")

    with open(args.path / "extracted" / "constraints.txt", "w") as f:
        for constraint, support in constraints:
            f.write(f"{constraint} - Support: {support:.2f}%\n")

    with open(args.path / "extracted" / "constraints_100.txt", "w") as f:
        for constraint, support in constraints:
            if support == 100:
                f.write(f"{constraint} - Support: {support:.2f}%\n")

    with open(args.path / "extracted" / "constraints_90_100.txt", "w") as f:
        for constraint, support in constraints:
            if 90 <= support < 100:
                f.write(f"{constraint} - Support: {support:.2f}%\n")

    with open(args.path / "extracted" / "constraints_80_90.txt", "w") as f:
        for constraint, support in constraints:
            if 80 <= support < 90:
                f.write(f"{constraint} - Support: {support:.2f}%\n")

    with open(args.path / "extracted" / "constraints_70_80.txt", "w") as f:
        for constraint, support in constraints:
            if 70 <= support < 80:
                f.write(f"{constraint} - Support: {support:.2f}%\n")


def parse_args() -> Args:
    argparser = argparse.ArgumentParser(
        prog="log2traces",
        description="Combines the constraints data from a '.declare' file and a '.txt' file.",
    )
    argparser.add_argument(
        "--path",
        type=str,
        help="Path to the folder specific to the dataset. Expected to contain a folder 'origin' with the files 'constraints.declare' and 'constraints.txt'.",
        required=True,
    )

    args = argparser.parse_args()
    cwd = Path.cwd()
    return Args(path=cwd / args.path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
