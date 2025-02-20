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


class LogDataset(Dataset):

    def __init__(self, traces: list[TraceList]):
        self.prefixes, self.targets, self.lengths = LogDataset._generate_prefixes(
            traces
        )

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.prefixes[idx]),
            torch.tensor(self.targets[idx]),
            torch.tensor(self.lengths[idx]),
        )

    @staticmethod
    def collate_fn(batch):
        batch = sorted(batch, key=lambda item: item[-1], reverse=True)
        prefixes, targets, lengths = zip(*batch)
        prefixes = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
        return prefixes, torch.stack(targets), torch.stack(lengths)

    def save(self, path: str | os.PathLike | pathlib.Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def _generate_prefixes(traces: npt.NDArray):
        prefixes: list[TraceArray, int] = []
        targets: list[Activity] = []
        prefixes_lengths: list[int] = []
        for trace in traces:
            for prefix, target, length in LogDataset._generate_prefixes_for_trace(
                trace
            ):
                prefixes.append(np.array(prefix))
                targets.append(target)
                prefixes_lengths.append(length)
        return (
            np.array(prefixes, dtype=object),
            np.array(targets),
            np.array(prefixes_lengths),
        )

    @staticmethod
    def _generate_prefixes_for_trace(trace: TraceArray):
        for i in range(1, len(trace)):
            prefix = trace[:i]
            target = trace[i]
            length = len(prefix)
            yield prefix, target, length
