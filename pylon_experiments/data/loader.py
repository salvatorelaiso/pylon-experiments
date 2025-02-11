import pathlib

from pydantic import Field
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader

from pylon_experiments.data.dataset import Dataset


@dataclass(frozen=True, kw_only=True)
class Args:
    dataset_path: pathlib.Path = Field(
        default_factory=lambda value: pathlib.Path(value)
    )
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True


class Loader:
    def __init__(
        self,
        args: Args,
    ):
        self.dataset = pathlib.Path(args.dataset_path)
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.datasets_path = self.dataset / "datasets"

    def _load_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        train_path = self.datasets_path / "train.pkl"
        val_path = self.datasets_path / "val.pkl"
        test_path = self.datasets_path / "test.pkl"
        if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
            raise FileNotFoundError(
                f"Could not find all datasets at {self.datasets_path}."
            )
        return (
            Dataset.load(train_path),
            Dataset.load(val_path),
            Dataset.load(test_path),
        )

    def _dataset_to_dataloader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=dataset.collate_fn,
        )

    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, val_dataset, test_dataset = self._load_datasets()
        return (
            self._dataset_to_dataloader(train_dataset, shuffle=True),
            self._dataset_to_dataloader(val_dataset),
            self._dataset_to_dataloader(test_dataset),
        )
