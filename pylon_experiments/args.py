from pydantic.dataclasses import dataclass

from pylon_experiments.data.loader import Args as LoaderArgs
from pylon_experiments.model.model import Args as ModelArgs


@dataclass(frozen=True, kw_only=True)
class Args:
    seed: int = 42
    epochs: int = 100
    loader_args: LoaderArgs
    model_args: ModelArgs
