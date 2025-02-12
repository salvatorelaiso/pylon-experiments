import json
from abc import ABC, abstractmethod

import torch


class Constraint(ABC):
    @abstractmethod
    def __call__(self, logits: torch.Tensor, *, lengths: list[int]) -> torch.Tensor:
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def short_str(self):
        pass

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
