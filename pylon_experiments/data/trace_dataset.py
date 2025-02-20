import os
import pathlib
import pickle

import numpy as np
import numpy.typing as npt
import torch

from pylon_experiments.data.dataset import Dataset

type Activity = int
type TraceList = list[Activity]
type TraceArray = npt.NDArray[np.int32]


class TraceDataset(Dataset):

    def __init__(self, traces: list[TraceList]):
        self.traces = np.array([np.array(trace) for trace in traces], dtype=object)
        self.traces_lengths = np.array(
            [len(trace) for trace in self.traces], dtype=np.uint8
        )

    def __len__(self):
        return len(self.traces_lengths)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.traces[idx]),
            torch.tensor(self.traces_lengths[idx]),
        )

    @staticmethod
    def collate_fn(batch):
        batch = sorted(batch, key=lambda item: item[-1], reverse=True)
        trace, lengths = zip(*batch)
        trace = torch.nn.utils.rnn.pad_sequence(trace, batch_first=True)
        return trace, torch.stack(lengths)

    @staticmethod
    def incremental_length_prefixes(traces: torch.Tensor, max_length: int):
        for length in range(1, max_length):
            yield TraceDataset.fixed_length_prefixes(traces, length)

    @staticmethod
    def fixed_length_prefixes(traces: torch.Tensor, length: int):
        prefixes = traces[:, :length].clone().detach()
        targets = traces[:, length].clone().detach()
        lengths = torch.tensor([length] * len(traces))
        mask = targets != 0
        return prefixes[mask], targets[mask], lengths[mask]

    def save(self, path: str | os.PathLike | pathlib.Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
