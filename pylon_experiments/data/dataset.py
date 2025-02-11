import os
import pathlib
import pickle
from abc import ABC, abstractmethod
from typing import Self

from torch.utils.data import Dataset


class Dataset(Dataset, ABC):
    @staticmethod
    def load(path: str | os.PathLike | pathlib.Path) -> Self:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        raise NotImplementedError
