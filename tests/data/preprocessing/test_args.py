import pathlib

import pytest

from pylon_experiments.data.preprocessing.args import Args


@pytest.mark.parametrize(
    "path, expected_path",
    [
        (
            "data/mydata",
            pathlib.Path("data/mydata"),
        ),
        (
            pathlib.Path("data/mydata"),
            pathlib.Path("data/mydata"),
        ),
    ],
)
def test_args_path_field(path, expected_path):
    args = Args(path=path)
    assert args.path == expected_path
