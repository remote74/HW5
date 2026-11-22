"""Check provenance, row/text alignment, no refitting, and sparse-error cases."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.pipeline import Pipeline

from error_analysis import analyze_errors, representative_errors
from train_baseline import split_training_data, train_baseline
from train_improved import train_improved


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = self.root / "reference"
        self.experiments = self.root / "experiments"
        self.output = self.root / "analysis"
        self.models = self.root / "models"
        self.frame = pd.DataFrame({
            "text": [f"<b>Terrible</b> boring awful movie negative{i}" for i in range(20)]
                    + [f"Excellent delightful wonderful movie positive{i}" for i in range(20)],
            "label": [0] * 20 + [1] * 20,
        })
        _, validation, _ = split_training_data(self.frame)
        # Deliberately contradictory labeled holdout to exercise both directions.
        for index, label in validation["label"].items():
            self.frame.loc[index, "text"] = (
                ("Excellent delightful wonderful" if label == 0 else "Terrible boring awful")
                + f" unique{index}")
        train_baseline(self.frame, self.reference, self.models)
        train_improved(self.frame, self.reference, self.experiments, self.models)

    def analyze(self):
        return analyze_errors(self.frame, self.reference, self.experiments, self.output)

    def test_real_errors_preserve_text_labels_and_training_artifacts_without_fit(self):
        preserved = {path: path.read_bytes() for folder in
                     (self.reference, self.experiments, self.models) for path in folder.iterdir()}
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Must not fit")), \
                patch.object(Pipeline, "fit_transform", side_effect=AssertionError("Must not fit")):
            result = self.analyze()
        for path, contents in preserved.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(result["incorrect_predictions"], 8)
        self.assertEqual(result["confusion_matrix"], [[0, 4], [4, 0]])
        self.assertFalse(result["dataset"]["official_test_used"])
        errors = pd.read_csv(self.output / "misclassified_reviews.csv")
        self.assertTrue(errors["true_label"].ne(errors["predicted_label"]).all())
        for row in errors.itertuples(index=False):
            source = self.frame.iloc[row.source_row_position]
            self.assertEqual(row.text, source["text"])
            self.assertEqual(row.true_label, source["label"])
        examples = pd.read_csv(self.output / "representative_errors.csv")
        self.assertEqual(set(examples["true_label"]), {0, 1})
        self.assertTrue(set(examples["source_row_position"]) <= set(errors["source_row_position"]))
        saved = json.loads((self.output / "error_analysis_results.json").read_text())
        self.assertEqual(saved, result)
        self.assertTrue((self.output / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG"))

    def test_reordered_or_changed_predictions_fail_before_outputs(self):
        path = self.experiments / "selected_predictions.csv"
        original = pd.read_csv(path)
        changed = original.copy()
        changed.loc[0, "predicted_label"] = 1 - changed.loc[0, "predicted_label"]
        for invalid in (original.iloc[::-1], changed):
            invalid.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Selected predictions"):
                self.analyze()
            self.assertFalse(self.output.exists())

    def test_changed_dataset_and_experiment_split_are_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "text"] += " drift"
        with self.assertRaisesRegex(ValueError, "Dataset differs"):
            analyze_errors(changed, self.reference, self.experiments, self.output)
        path = self.experiments / "improvement_results.json"
        record = json.loads(path.read_text())
        record["split"]["random_state"] = 100
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Task 4 dataset, split"):
            self.analyze()
        self.assertFalse(self.output.exists())

    def test_no_errors_writes_empty_inspectable_tables_and_no_fake_examples(self):
        # Matching labels now allow the tiny fixture to be classified perfectly.
        self.frame["text"] = [
            ("Terrible boring awful" if label == 0 else "Excellent delightful wonderful")
            + f" unique{index}" for index, label in self.frame["label"].items()]
        train_baseline(self.frame, self.reference, self.models)
        train_improved(self.frame, self.reference, self.experiments, self.models)
        result = self.analyze()
        self.assertEqual(result["incorrect_predictions"], 0)
        for name in ("misclassified_reviews.csv", "representative_errors.csv"):
            table = pd.read_csv(self.output / name)
            self.assertTrue(table.empty)
            self.assertIn("text", table.columns)
        report = (self.output / "error_analysis.md").read_text()
        self.assertIn("No incorrect predictions", report)
        self.assertNotIn("### Source row", report)

    def test_sampling_handles_one_direction_and_small_groups_deterministically(self):
        errors = pd.DataFrame({"source_row_position": [9, 7, 8, 6, 5],
                               "true_label": [0] * 5, "predicted_label": [1] * 5,
                               "raw_word_count": [10, 10, 30, 20, 40]})
        sample = representative_errors(errors, per_pair=3)
        self.assertEqual(sample["source_row_position"].tolist(), [7, 6, 5])
        self.assertEqual(len(representative_errors(errors, per_pair=20)), 5)
        self.assertTrue(representative_errors(errors.iloc[:0]).empty)
        with self.assertRaisesRegex(ValueError, "positive"):
            representative_errors(errors, per_pair=0)


if __name__ == "__main__":
    unittest.main()
