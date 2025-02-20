import numpy as np
import pandas as pd
import pytest
import torch

from pylon_experiments.data.trace_dataset import TraceDataset


@pytest.mark.parametrize(
    "traces, expected_batch, expected_generated_prefixes",
    [
        (
            [[1, 2, 3, 4], [5, 6, 7], [8, 9], [10]],
            (
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0], [10, 0, 0, 0]]),
                torch.tensor([4, 3, 2, 1]),
            ),
            [
                (
                    torch.tensor([[1], [5], [8]]),
                    torch.tensor([2, 6, 9]),
                    torch.tensor([1, 1, 1]),
                ),
                (
                    torch.tensor([[1, 2], [5, 6]]),
                    torch.tensor([3, 7]),
                    torch.tensor([2, 2]),
                ),
                (torch.tensor([[1, 2, 3]]), torch.tensor([4]), torch.tensor([3])),
            ],
        ),
    ],
)
def test_(traces, expected_batch, expected_generated_prefixes):
    ds = TraceDataset(traces)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=4, collate_fn=TraceDataset.collate_fn
    )
    for batch in dataloader:
        assert torch.equal(batch[0], expected_batch[0])
        assert torch.equal(batch[1], expected_batch[1])

        for prefixes, expected in zip(
            TraceDataset.incremental_length_prefixes(
                traces=batch[0], max_length=batch[1][0]
            ),
            expected_generated_prefixes,
        ):
            assert torch.equal(prefixes[0], expected[0])
            assert torch.equal(prefixes[1], expected[1])
            assert torch.equal(prefixes[2], expected[2])
