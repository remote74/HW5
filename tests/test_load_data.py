"""Check the validation boundary that later supervised stages will reuse."""

import tempfile
import unittest

import pandas as pd

from load_data import load_imdb, validate_data


class DataValidationTests(unittest.TestCase):
    def test_valid_supervised_and_unlabeled_data(self):
        validate_data(pd.DataFrame({"text": ["Bad movie", "Great movie"], "label": [0, 1]}))
        validate_data(pd.DataFrame({"text": ["A review"], "label": [-1]}), supervised=False)

    def test_rejects_unusable_data(self):
        cases = [
            pd.DataFrame({"text": ["review"]}),
            pd.DataFrame({"text": [], "label": []}),
            pd.DataFrame({"text": [None], "label": [0]}),
            pd.DataFrame({"text": ["review"], "label": [None]}),
            pd.DataFrame({"text": [123], "label": [0]}),
            pd.DataFrame({"text": [" \n\t"], "label": [0]}),
            pd.DataFrame({"text": ["review"], "label": [2]}),
            pd.DataFrame({"text": ["review"], "label": ["0"]}),
            pd.DataFrame({"text": ["review"], "label": [0.5]}),
            pd.DataFrame({"text": ["review"], "label": [True]}),
        ]
        for frame in cases:
            with self.subTest(frame=frame.to_dict()), self.assertRaises(ValueError):
                validate_data(frame)

    def test_unlabeled_sentinel_cannot_enter_supervised_data(self):
        with self.assertRaises(ValueError):
            validate_data(pd.DataFrame({"text": ["review"], "label": [-1]}))

    def test_offline_cache_miss_is_explicit(self):
        with tempfile.TemporaryDirectory() as cache, self.assertRaises(FileNotFoundError):
            load_imdb(cache, offline=True)


if __name__ == "__main__":
    unittest.main()
