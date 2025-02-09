import numpy as np
import pandas as pd
import pytest

from pylon_experiments.data.vocab import Vocab, generate_vocab


@pytest.mark.parametrize(
    "activities, expected_vocabulary",
    [
        (
            ["A", "B", "C"],
            ["<pad>", "<unk>", "<sos>", "<eos>", "A", "B", "C"],
        ),
        (
            ["A", "B", "C", "A"],
            ["<pad>", "<unk>", "<sos>", "<eos>", "A", "B", "C"],
        ),
        (
            ["A", "B", "C", "D"],
            ["<pad>", "<unk>", "<sos>", "<eos>", "A", "B", "C", "D"],
        ),
        (
            np.array(["A", "B", "C", "D"]),
            ["<pad>", "<unk>", "<sos>", "<eos>", "A", "B", "C", "D"],
        ),
        (
            pd.Series(["A", "B", "C", "D"]),
            ["<pad>", "<unk>", "<sos>", "<eos>", "A", "B", "C", "D"],
        ),
    ],
)
def test_vocab_init(activities, expected_vocabulary):
    vocabulary = generate_vocab(activities)
    assert vocabulary.activity2idx == {
        activity: idx for idx, activity in enumerate(expected_vocabulary)
    }
