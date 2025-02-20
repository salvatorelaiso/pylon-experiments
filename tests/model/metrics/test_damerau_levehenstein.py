import unittest

import torch
from pyxdameraulevenshtein import damerau_levenshtein_distance

from pylon_experiments.model.metrics.damerau_levehenstein import (
    DamerauLevenshteinDistance,
)


class TestDamerauLevenshteinMetrics(unittest.TestCase):
    def test_damerau_levenshtein_function(self):
        """Test the behavior of the function provided by the pyxdameraulevenshtein package."""

        identical_strings_distance = damerau_levenshtein_distance("hello", "hello")
        self.assertEqual(identical_strings_distance, 0.0)

        different_strings_distance = damerau_levenshtein_distance("hello", "world")
        self.assertEqual(
            different_strings_distance, 4.0
        )  # 5 (replacement) operations to transform 'hello' into 'world'

        identical_string_lists_distance = damerau_levenshtein_distance(
            ["hello", "world"], ["hello", "world"]
        )
        self.assertEqual(identical_string_lists_distance, 0.0)

        inverted_string_lists_distance = damerau_levenshtein_distance(
            ["hello", "world"], ["world", "hello"]
        )
        self.assertEqual(inverted_string_lists_distance, 1.0)

        swapped_string_lists_distance = damerau_levenshtein_distance(
            ["hello", "planet", "earth", "!"],
            ["hello", "earth", "planet", "!"],
        )
        self.assertEqual(swapped_string_lists_distance, 1.0)

        identical_int_lists_distance = damerau_levenshtein_distance(
            [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]
        )
        self.assertEqual(identical_int_lists_distance, 0.0)

        inverted_int_lists_distance = damerau_levenshtein_distance(
            [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
        )
        self.assertEqual(
            inverted_int_lists_distance, 4.0
        )  # 4 (replacement) operations to transform [1, 2, 3, 4, 5] into [5, 4, 3, 2, 1]

        identical_double_digits_distance = damerau_levenshtein_distance(
            [10, 20, 30, 40, 50], [10, 20, 30, 40, 50]
        )
        self.assertEqual(identical_double_digits_distance, 0.0)

        inverted_double_digits_distance = damerau_levenshtein_distance(
            [10, 20, 30, 40, 50], [50, 40, 30, 20, 10]
        )
        self.assertEqual(
            inverted_double_digits_distance, 4.0
        )  # 4 (replacement) operations to transform [10, 20, 30, 40, 50] into [50, 40, 30, 20, 10]

        mixed_int_lists_distance = damerau_levenshtein_distance(
            [1, 2, 3, 4, 5], [10, 20, 30, 40, 50]
        )
        self.assertEqual(mixed_int_lists_distance, 5.0)

        mixed_int_lists_distance = damerau_levenshtein_distance(
            [1, 2, 3, 4], [1, 2, 99, 4]
        )
        self.assertEqual(mixed_int_lists_distance, 1.0)

        mixed_length_lists_distance = damerau_levenshtein_distance(
            [1, 2, 3, 4], [1, 2, 3, 4, 5]
        )
        self.assertEqual(mixed_length_lists_distance, 1.0)

    def test_damerau_levenshtein_distance(self):
        metric = DamerauLevenshteinDistance()

        y = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        y_hat = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        metric.update(y_hat, y)
        self.assertAlmostEqual(metric.compute().item(), 0.0)
        metric.reset()

        y_hat = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])
        metric.update(y_hat, y)
        self.assertAlmostEqual(metric.compute().item(), 1.0)
        metric.reset()

        y_hat = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
        metric.update(y_hat, y)
        self.assertAlmostEqual(metric.compute().item(), 1.0)
        metric.reset()

        y = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
        )
        y_hat = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        )
        metric.update(y_hat, y)
        self.assertAlmostEqual(metric.compute().item(), 0.5)

        # Note: metric is not reset here
        y_hat = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        )
        metric.update(y_hat, y)
        self.assertAlmostEqual(
            metric.compute().item(), 0.75
        )  # ( 0.0 + 1.0 + 1.0 + 1.0 ) / 4 = 0.75


class TestAccuracyMetric(unittest.TestCase):
    def test_accuracy(self):
        import torchmetrics

        metric = torchmetrics.Accuracy(task="multiclass", num_classes=11)

        y = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        y_hat = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        metric.update(y_hat, y)
        self.assertAlmostEqual(metric.compute().item(), 1.0)
        metric.reset()

        y = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        y_hat = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])
        metric.update(y_hat, y)
        self.assertAlmostEqual(metric.compute().item(), 0.9)
        metric.reset()

        y = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        )
        y_hat = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
        )
        metric.update(y_hat, y)
        self.assertAlmostEqual(metric.compute().item(), 0.9)
        metric.reset()

        y = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        )
        y_hat = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 99, 99], [1, 2, 3, 4, 5, 6, 7, 8, 99, 99]]
        )
        metric.update(y_hat, y)
        self.assertAlmostEqual(
            metric.compute().item(), 0.8
        )  # 1.0 - (2/10 + 2/10) / 2 = 0.8

        metric.update(y_hat, y)
        self.assertAlmostEqual(
            metric.compute().item(), 0.8
        )  # 1.0 - (2/10 + 2/10 + 2/10 + 2/10) / 4 = 0.8

        y_hat = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 99]]
        )
        metric.update(y_hat, y)
        self.assertAlmostEqual(
            metric.compute().item(), 0.85
        )  # 1.0 - (2/10 + 2/10 + 2/10 + 2/10 + 0/10 + 1/10) / 6 = 0.85


if __name__ == "__main__":
    unittest.main()
