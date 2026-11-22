"""Verify fair comparisons, drift detection, and selected-model persistence."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from train_baseline import split_training_data, train_baseline
from train_improved import build_candidates, load_reference, select_best, train_improved


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "text": [f"<b>Terrible</b> boring movie negative{i}" for i in range(20)]
                    + [f"Excellent delightful movie positive{i}" for i in range(20)],
            "label": [0] * 20 + [1] * 20,
        })
        _, validation, _ = split_training_data(self.frame)
        self.frame.loc[validation.index, "text"] += " exclusiveholdoutword"

    def test_each_candidate_keeps_holdout_out_of_fitted_vocabulary(self):
        train, validation, _ = split_training_data(self.frame)
        for name, pipeline in build_candidates().items():
            with self.subTest(candidate=name):
                pipeline.fit(train["text"], train["label"])
                vectorizer = pipeline.named_steps["tfidf"]
                vocabulary = vectorizer.vocabulary_.copy()
                pipeline.predict(validation["text"])
                self.assertNotIn("exclusiveholdoutword", vectorizer.vocabulary_)
                self.assertEqual(vocabulary, vectorizer.vocabulary_)
                np.testing.assert_array_equal(pipeline.predict(["<b>EXCELLENT</b> delightful"]),
                                              pipeline.predict(["excellent delightful"]))

    def test_baseline_is_retained_on_ties_or_worse_candidates(self):
        records = [{"name": name, "metrics": {"f1_macro": score}}
                   for name, score in [("baseline", 0.9), ("tied", 0.9), ("worse", 0.8)]]
        self.assertEqual(select_best(records)["name"], "baseline")
        records.append({"name": "better", "metrics": {"f1_macro": 0.91}})
        self.assertEqual(select_best(records)["name"], "better")

    def test_comparison_and_selected_model_round_trip_preserve_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, output, models = root / "reference", root / "comparison", root / "models"
            baseline = train_baseline(self.frame, reference, models)
            preserved = {path: path.read_bytes() for path in reference.iterdir()}
            preserved[models / "baseline.joblib"] = (models / "baseline.joblib").read_bytes()
            result = train_improved(self.frame, reference, output, models)
            for path, contents in preserved.items():
                self.assertEqual(path.read_bytes(), contents)
            stored = json.loads((output / "improvement_results.json").read_text())
            self.assertEqual(stored, result)
            self.assertEqual(result["experiments"][0]["metrics"], baseline["metrics"])
            self.assertEqual(len(result["experiments"]), 4)
            comparison = pd.read_csv(output / "improvement_comparison.csv")
            self.assertEqual(comparison["selected"].sum(), 1)
            self.assertTrue((output / "improvements.md").exists())
            predictions = pd.read_csv(output / "selected_predictions.csv")
            model = joblib.load(models / "selected.joblib")
            np.testing.assert_array_equal(
                predictions["predicted_label"],
                model.predict(self.frame.iloc[predictions["source_row_position"]]["text"]))
            self.assertEqual(model.predict(["Awful terrible boring", "Excellent delightful"]).tolist(), [0, 1])
            self.assertFalse(result["dataset"]["official_test_used"])

    def test_reference_drift_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_baseline(self.frame, root, root / "models")
            changed = self.frame.copy()
            changed.loc[0, "text"] += " changed"
            with self.assertRaisesRegex(ValueError, "Dataset differs"):
                load_reference(changed, root)
            path = root / "baseline_results.json"
            original = path.read_text()
            for section, key, value, message in [
                ("preprocessing", "implementation_sha256", "wrong", "Preprocessing differs"),
                ("split", "random_state", 17, "Settings differ"),
                ("vectorizer", "max_features", 10, "Settings differ"),
            ]:
                with self.subTest(section=section):
                    record = json.loads(original)
                    record[section][key] = value
                    path.write_text(json.dumps(record))
                    with self.assertRaisesRegex(ValueError, message):
                        load_reference(self.frame, root)
            path.write_text(original)
            split_path = root / "baseline_split.csv"
            split = pd.read_csv(split_path)
            split.iloc[::-1].to_csv(split_path, index=False)
            with self.assertRaisesRegex(ValueError, "Split assignments differ"):
                load_reference(self.frame, root)


if __name__ == "__main__":
    unittest.main()
