import json
import os
import pathlib

from pydantic.dataclasses import dataclass

from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.model.model import Args as ModelArgs


@dataclass(frozen=True, kw_only=True)
class Args:
    seed: int = 42
    epochs: int = 100
    learning_rate: float = 0.001
    loader_args: LoaderArgs
    model_args: ModelArgs

    def dump_args(self, path: str | os.PathLike | pathlib.Path) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "seed": self.seed,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "dataset": self.loader_args.dataset_path.name,
                    "model": {
                        "embedding": self.model_args.embedding_dim,
                        "hidden_size": self.model_args.hidden_size,
                        "num_layers": self.model_args.num_layers,
                    },
                },
                f,
                indent=4,
            )
