"""Check leakage boundaries, reproducible splits, and persisted raw-text inference."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from preprocess import preprocess_text
from train_baseline import build_pipeline, split_training_data, train_baseline


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "text": [f"<b>Terrible</b> boring movie uniqueneg{i}" for i in range(20)]
                    + [f"<b>Excellent</b> delightful movie uniquepos{i}" for i in range(20)],
            "label": [0] * 20 + [1] * 20,
        })

    def test_split_is_reproducible_stratified_and_aligned(self):
        train, validation, rejected = split_training_data(self.frame)
        again, _, _ = split_training_data(self.frame)
        pd.testing.assert_frame_equal(train, again)
        self.assertFalse(set(train.index) & set(validation.index))
        self.assertEqual(set(train.index) | set(validation.index), set(self.frame.index))
        self.assertEqual(validation["label"].value_counts().to_dict(), {0: 4, 1: 4})
        self.assertEqual(rejected, 0)
        pd.testing.assert_frame_equal(validation, self.frame.loc[validation.index])

    def test_cleaned_duplicates_and_conflicting_labels_stay_in_one_partition(self):
        copies = pd.DataFrame({
            "text": [self.frame.iloc[0]["text"],
                     " TERRIBLE   boring movie uniqueneg0 ",
                     "<b>terrible</b> boring movie uniqueneg0"],
            "label": [0, 0, 1],
        })
        frame = pd.concat([self.frame, copies], ignore_index=True)
        # Duplicate caller indexes must not affect positional assignments.
        frame.index = [0] * len(frame)
        train, validation, _ = split_training_data(frame)
        self.assertFalse(set(train["text"].map(preprocess_text)) &
                         set(validation["text"].map(preprocess_text)))
        self.assertEqual(set(train.index) | set(validation.index), set(range(len(frame))))
        duplicate_rows = {0, 40, 41, 42}
        self.assertTrue(duplicate_rows <= set(train.index) or
                        duplicate_rows <= set(validation.index))
        again, holdout, _ = split_training_data(frame)
        pd.testing.assert_frame_equal(train, again)
        pd.testing.assert_frame_equal(validation, holdout)

    def test_insufficient_cleaned_groups_fail(self):
        frame = pd.DataFrame({"text": ["bad"] * 10 + ["good"] * 10,
                              "label": [0] * 10 + [1] * 10})
        with self.assertRaisesRegex(ValueError, "distinct cleaned-text groups"):
            split_training_data(frame)

    def test_vocabulary_and_idf_only_learn_training_texts(self):
        pipeline = build_pipeline()
        pipeline.fit(["awful movie", "awful acting", "great movie", "great acting"], [0, 0, 1, 1])
        vectorizer = pipeline.named_steps["tfidf"]
        vocabulary, idf = vectorizer.vocabulary_.copy(), vectorizer.idf_.copy()
        pipeline.predict(["exclusiveholdoutword awful", "exclusiveholdoutword great"])
        self.assertNotIn("exclusiveholdoutword", vectorizer.vocabulary_)
        self.assertEqual(vectorizer.vocabulary_, vocabulary)
        np.testing.assert_array_equal(vectorizer.idf_, idf)
        np.testing.assert_array_equal(pipeline.predict(["<b>GREAT</b> movie"]),
                                      pipeline.predict(["great movie"]))
        for invalid in [[None], ["<br/>"], "great movie"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pipeline.predict(invalid)

    def test_rejections_preserve_pairing_and_small_classes_fail(self):
        frame = pd.concat([pd.DataFrame({"text": [None], "label": [0]}), self.frame])
        train, validation, rejected = split_training_data(frame)
        self.assertEqual(rejected, 1)
        self.assertEqual(len(train) + len(validation), 40)
        self.assertNotIn(0, set(train.index) | set(validation.index))
        with self.assertRaisesRegex(ValueError, "both sentiment classes"):
            split_training_data(self.frame.iloc[:20])

    def test_saved_reference_and_model_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output, models = Path(directory) / "reports", Path(directory) / "models"
            result = train_baseline(self.frame, output, models)
            stored = json.loads((output / "baseline_results.json").read_text())
            self.assertEqual(stored["metrics"], result["metrics"])
            self.assertTrue(stored["persistence"]["reload_predictions_identical"])
            self.assertFalse(stored["dataset"]["official_test_used"])
            self.assertEqual(len(pd.read_csv(output / "baseline_split.csv")), 40)
            self.assertTrue((output / "baseline.md").exists())
            model = joblib.load(models / "baseline.joblib")
            self.assertEqual(model.predict(["TERRIBLE<br>boring", "Excellent delightful"]).tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
