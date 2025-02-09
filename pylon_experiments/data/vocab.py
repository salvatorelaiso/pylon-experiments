import pickle

import numpy as np
import pandas as pd


class Vocab:
    """
    Vocabulary class used to map activities to indices and vice versa.
    """

    def __init__(self):
        self.activity2idx: dict[str, int] = {}
        self.idx2activity: dict[int, str] = {}
        self.idx = 0

    def __len__(self):
        return len(self.activity2idx)

    def __getitem__(self, activity):
        return self.activity2idx[activity]

    def __contains__(self, activity):
        return activity in self.activity2idx

    def add_activity(self, activity):
        if activity not in self.activity2idx:
            self.activity2idx[activity] = self.idx
            self.idx2activity[self.idx] = activity
            self.idx += 1

    def add_activities(self, activities):
        for activity in activities:
            self.add_activity(activity)

    def to_idx(self, activity):
        return self.activity2idx[activity]

    def to_idxs(self, activities):
        return [self.activity2idx[activity] for activity in activities]

    def to_activity(self, idx):
        return self.idx2activity[idx]

    def to_activities(self, idxs):
        return [self.idx2activity[idx] for idx in idxs]

    def save(self, path):
        with open(path, "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return f"Vocab({self.idx2activity})"

    def __str__(self):
        return f"Vocab({len(self.activity2idx)} activities)"

    def to_csv(self, path):
        with open(path, "w") as f:
            for i in self.idx2activity:
                f.write(f"{i},{self.idx2activity[i]}\n")

    @staticmethod
    def from_csv(path):
        vocab = Vocab()
        with open(path, "r") as f:
            for line in f:
                idx, activity = line.strip().split(",")
                vocab.activity2idx[activity] = int(idx)
                vocab.idx2activity[int(idx)] = activity
        vocab.idx = len(vocab.activity2idx)
        return vocab


def generate_vocab(activities: list | np.ndarray | pd.Series) -> Vocab:
    """
    Generate a vocabulary from the unique activities in the traces.

    :param unique_activities: The set of unique activities in the traces.
    :return: A vocabulary object with the unique activities.
    :rtype: Vocab
    """
    vocab = Vocab()
    vocab.add_activities(["<pad>", "<unk>", "<sos>", "<eos>"])

    unique_activities = pd.Series(activities).unique()
    vocab.add_activities(unique_activities)

    return vocab
