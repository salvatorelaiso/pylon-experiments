import pathlib

import torch
import torch.nn as nn
from pydantic import Field
from pydantic.dataclasses import dataclass
from torch.nn.utils.rnn import pack_padded_sequence

from pylon_experiments.data.vocab import Vocab


@dataclass(frozen=True, kw_only=True)
class Args:
    hidden_size: int
    num_layers: int
    embedding_dim: int
    vocab_path: pathlib.Path = Field(default_factory=lambda value: pathlib.Path(value))


class NextActivityPredictor(nn.Module):
    def __init__(self, args: Args):
        super().__init__()

        self.vocab = Vocab.load(args.vocab_path)

        num_classes = len(self.vocab)
        padding_index = self.vocab["<pad>"]

        self.embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=args.embedding_dim,
            padding_idx=padding_index,
        )
        self.lstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            batch_first=True,
            num_layers=args.num_layers,
        )

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        self.fc = nn.Linear(args.hidden_size, num_classes)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

    def predict(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.forward(x, lengths).argmax(dim=-1)

    def predict_probabilities(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return torch.softmax(self.predict(x, lengths))
