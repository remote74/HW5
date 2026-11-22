"""Verify the shared cleaning contract, rejection alignment, and saved outputs."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from prepare_data import prepare_splits
from preprocess import prepare_data, preprocess_text


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_preserves_sentiment_cues(self):
        raw = " ＮＯＴ<br />bad! &amp; I don’t hate it.\nCafé 10/10\x00 "
        self.assertEqual(preprocess_text(raw), "not bad! & i don't hate it. café 10/10")
        self.assertEqual(preprocess_text("<b>Great</b><i>film</i>"), "great film")
        self.assertEqual(preprocess_text("Nice<!-- hidden --><script>bad()</script><style>x</style> film"),
                         "nice film")

    def test_unusable_inference_inputs(self):
        for raw in [None, float("nan"), pd.NA, 123, [], {}, "", " \n\t", "!!!", "<br/> &nbsp;",
                    "<script>bad()</script>", "<![broken]>", "\x00\u200b"]:
            with self.subTest(raw=raw):
                self.assertEqual(preprocess_text(raw), "")

    def test_rows_labels_and_indexes_stay_aligned(self):
        frame = pd.DataFrame({
            "text": ["Good!", None, " <br> ", "BAD", "   ", "review", "review", "review"],
            "label": pd.Series([1, 0, 1, 0, 0, -1, True, "1"], dtype=object),
            "extra": list("abcdefgh"),
        })
        frame.index = [90, 10, 80, 20, 70, 30, 60, 40]
        original = frame.copy(deep=True)
        prepared, rejected = prepare_data(frame)
        self.assertEqual(prepared.index.tolist(), [90, 20])
        self.assertEqual(prepared["text"].tolist(), ["good!", "bad"])
        self.assertEqual(prepared["label"].tolist(), [1, 0])
        self.assertEqual(str(prepared["label"].dtype), "int64")
        self.assertEqual(prepared["extra"].tolist(), ["a", "d"])
        self.assertEqual(rejected.index.tolist(), [10, 80, 70, 30, 60, 40])
        self.assertEqual(rejected["rejection_reason"].tolist(), [
            "non_string_text", "empty_after_preprocessing", "blank_text",
            "invalid_label", "invalid_label", "invalid_label"])
        pd.testing.assert_frame_equal(frame, original)

    def test_schema_and_all_rejected(self):
        with self.assertRaises(ValueError):
            prepare_data(pd.DataFrame({"text": ["Good"]}))
        prepared, rejected = prepare_data(pd.DataFrame({"text": [None], "label": [0]}))
        self.assertTrue(prepared.empty)
        self.assertEqual(len(rejected), 1)

    def test_missing_labels_do_not_discard_other_valid_rows(self):
        frame = pd.DataFrame({"text": ["Good", "Bad", "Unknown", "Fractional"],
                              "label": [1, 0, None, 0.5]})
        prepared, rejected = prepare_data(frame)
        self.assertEqual(prepared["label"].tolist(), [1, 0])
        self.assertEqual(prepared.index.tolist(), [0, 1])
        self.assertEqual(rejected.index.tolist(), [2, 3])

    def test_saved_splits_and_label_mapping_round_trip(self):
        frames = {
            "train": pd.DataFrame({"text": ["Good<br/>film", "BAD", "<br>"], "label": [1, 0, 1]},
                                  index=[12, 4, 8]),
            "test": pd.DataFrame({"text": ["Fine"], "label": [1]}, index=[99]),
        }
        with tempfile.TemporaryDirectory() as directory:
            output, reports = Path(directory) / "data", Path(directory) / "reports"
            report = prepare_splits(frames, output, reports)
            for split, raw in frames.items():
                expected, _ = prepare_data(raw)
                pd.testing.assert_frame_equal(pd.read_parquet(output / f"{split}.parquet"), expected)
            self.assertEqual(json.loads((output / "label_mapping.json").read_text()),
                             {"0": "negative", "1": "positive"})
            rejected = pd.read_csv(reports / "preprocessing_rejections.csv")
            self.assertEqual(rejected["source_row"].tolist(), [8])
            self.assertEqual(rejected["label"].tolist(), [1])
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
