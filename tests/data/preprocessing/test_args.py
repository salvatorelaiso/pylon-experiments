import pathlib
from contextlib import nullcontext as does_not_raise

import pytest

from pylon_experiments.data.preprocessing.args import Args


@pytest.mark.parametrize(
    "path, expected_path, expectation",
    [
        (
            "data/mydata",
            pathlib.Path("data/mydata"),
            does_not_raise(),
        ),
        (
            pathlib.Path("data/mydata"),
            pathlib.Path("data/mydata"),
            does_not_raise(),
        ),
        (
            None,
            None,
            pytest.raises(Exception),
        ),
    ],
)
def test_args_path_field(path, expected_path, expectation):
    with expectation:
        args = Args(path=path)
        assert args.path == expected_path
