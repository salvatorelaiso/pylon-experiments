import json
import os
import pathlib

from pydantic.dataclasses import Field, dataclass

from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.model.model import Args as ModelArgs


@dataclass(frozen=True, kw_only=True)
class Args:
    seed: int = 42
    epochs: int = 100
    learning_rate: float = 0.001
    constraints_multiplier: float = 1.0
    loader_args: LoaderArgs
    model_args: ModelArgs
    constraints: list[str] = Field(default_factory=list)

    def dump_args(self, path: str | os.PathLike | pathlib.Path) -> None:
        with open(path, "w") as f:
            data = {
                "seed": self.seed,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "dataset": self.loader_args.dataset_path.parent.name
                + "/"
                + self.loader_args.dataset_path.name,
                "model": {
                    "embedding": self.model_args.embedding_dim,
                    "hidden_size": self.model_args.hidden_size,
                    "num_layers": self.model_args.num_layers,
                    "dropout": self.model_args.dropout,
                },
                "constraints": self.constraints,
            }
            if self.constraints:
                data["constraints_multiplier"] = self.constraints_multiplier
            json.dump(data, f, indent=4)

    def print_args(self) -> None:
        print(f"Seed: {self.seed}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Constraints multiplier: {self.constraints_multiplier}")
        print(f"Dataset: {self.loader_args.dataset_path}")
        print(f"Model:")
        print(f"\tEmbedding: {self.model_args.embedding_dim}")
        print(f"\tHidden size: {self.model_args.hidden_size}")
        print(f"\tNumber of layers: {self.model_args.num_layers}")
        print(f"\tDropout: {self.model_args.dropout}")
        print(f"Constraints: {self.constraints}")
