import pathlib

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Args:
    path: pathlib.Path = Field(default_factory=lambda value: pathlib.Path(value))
