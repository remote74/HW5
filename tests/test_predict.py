"""Verify raw-text inference, persistence, input alignment, and no refitting."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predict import load_model, predict_texts
from train_baseline import build_pipeline, train_baseline
from train_improved import train_improved
from verify_inference import EXAMPLE_TEXTS, verify_inference


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = build_pipeline().fit(
            ["Terrible boring movie", "Awful boring movie", "Excellent delightful movie",
             "Excellent wonderful movie"], [0, 0, 1, 1])
        self.path = self.root / "selected.joblib"
        joblib.dump(self.model, self.path)

    def test_saved_pipeline_matches_original_without_fitting_or_state_changes(self):
        texts = ["<b>EXCELLENT</b> wonderful movie", "Terrible boring movie novelword",
                 "<b>EXCELLENT</b> wonderful movie"]
        expected = self.model.predict(texts).tolist()
        before = self.path.read_bytes()
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No fitting")), \
                patch.object(Pipeline, "fit_transform", side_effect=AssertionError("No fitting")), \
                patch.object(TfidfVectorizer, "fit", side_effect=AssertionError("No fitting")), \
                patch.object(TfidfVectorizer, "fit_transform", side_effect=AssertionError("No fitting")), \
                patch.object(LogisticRegression, "fit", side_effect=AssertionError("No fitting")):
            model = load_model(self.path)
            state = joblib.hash(model)
            vocabulary = model.named_steps["tfidf"].vocabulary_.copy()
            idf = model.named_steps["tfidf"].idf_.copy()
            self.assertEqual(predict_texts(texts, model), expected)
            self.assertEqual(predict_texts(tuple(texts), model_path=self.path), expected)
            self.assertEqual(predict_texts(["excellent wonderful movie"], model), [1])
            self.assertEqual(joblib.hash(model), state)
            self.assertEqual(model.named_steps["tfidf"].vocabulary_, vocabulary)
            np.testing.assert_array_equal(model.named_steps["tfidf"].idf_, idf)
            self.assertNotIn("novelword", vocabulary)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(expected, [1, 0, 1])
        self.assertTrue(all(type(label) is int for label in expected))

    def test_invalid_batches_fail_without_silently_dropping_rows(self):
        for invalid in ("excellent", None, {"text": "excellent"}, {"excellent"},
                        [None], [123], ["excellent", ""], [" "], ["<br/>"],
                        ["<script>excellent</script>"], ["!!!"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                predict_texts(invalid, self.model)
        with patch("predict.load_model", side_effect=AssertionError("No loading needed")):
            self.assertEqual(predict_texts([]), [])
            self.assertEqual(predict_texts(()), [])

    def test_unfitted_or_incomplete_artifact_is_rejected(self):
        for model in (build_pipeline(), self.model.named_steps["classifier"]):
            joblib.dump(model, self.path)
            with self.assertRaises(ValueError):
                load_model(self.path)

    def test_cli_loads_from_fresh_process_and_preserves_json_order(self):
        script = Path(__file__).resolve().parents[1] / "predict.py"
        texts = ["excellent wonderful movie", "terrible boring movie"]
        source = self.root / "reviews.json"
        source.write_text(json.dumps(texts), encoding="utf-8")
        for arguments in (["--input-json", str(source)],
                          ["--text", texts[0], "--text", texts[1]]):
            run = subprocess.run([sys.executable, str(script), "--model-path", str(self.path),
                                  *arguments], cwd=self.root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            rows = json.loads(run.stdout)
            self.assertEqual([row["text"] for row in rows], texts)
            self.assertEqual([row["predicted_label"] for row in rows], [1, 0])
            self.assertEqual([row["predicted_sentiment"] for row in rows], ["positive", "negative"])
        source.write_text('["excellent", null]', encoding="utf-8")
        run = subprocess.run([sys.executable, str(script), "--input-json", str(source)],
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, "")

    def test_selected_workflow_records_real_examples_without_modifying_artifacts(self):
        frame = pd.DataFrame({
            "text": [f"Terrible boring awful movie negative{i}" for i in range(20)]
                    + [f"Excellent delightful wonderful movie positive{i}" for i in range(20)],
            "label": [0] * 20 + [1] * 20,
        })
        reference, models, output = self.root / "reports", self.root / "models", self.root / "output"
        train_baseline(frame, reference, models)
        train_improved(frame, reference, reference, models)
        preserved = {p: p.read_bytes() for folder in (reference, models) for p in folder.iterdir()}
        with patch.object(Pipeline, "fit", side_effect=AssertionError("No fitting")), \
                patch.object(TfidfVectorizer, "fit_transform", side_effect=AssertionError("No fitting")), \
                patch.object(LogisticRegression, "fit", side_effect=AssertionError("No fitting")):
            result = verify_inference(reference, output)
        self.assertEqual(result, json.loads((output / "inference_examples.json").read_text()))
        self.assertEqual([row["text"] for row in result["examples"]], EXAMPLE_TEXTS)
        self.assertEqual([row["predicted_label"] for row in result["examples"]],
                         self.model_labels(models))
        for key in ("task_4_original_validation_reload_check", "new_examples_reload_predictions_identical",
                    "fitted_state_unchanged", "saved_artifact_unchanged"):
            self.assertTrue(result[key])
        for path, contents in preserved.items():
            self.assertEqual(path.read_bytes(), contents)
        relocated = self.root / "relocated.joblib"
        relocated.write_bytes((models / "selected.joblib").read_bytes())
        copy_result = verify_inference(reference, self.root / "copy_output", relocated)
        self.assertTrue(copy_result["selected_artifact_digest_verified"])
        changed = joblib.load(relocated)
        changed.named_steps["classifier"].coef_ *= -1
        changed.named_steps["classifier"].intercept_ *= -1
        self.assertNotEqual(predict_texts(EXAMPLE_TEXTS, changed),
                            self.model_labels(models))
        joblib.dump(changed, relocated)
        with self.assertRaisesRegex(ValueError, "Artifact digest differs"):
            verify_inference(reference, self.root / "changed_output", relocated)
        self.assertFalse((self.root / "changed_output").exists())
        record_path = reference / "improvement_results.json"
        original_record = record_path.read_text()
        missing_digest = json.loads(original_record)
        del missing_digest["persistence"]["model_sha256"]
        record_path.write_text(json.dumps(missing_digest))
        with self.assertRaisesRegex(ValueError, "Missing selected artifact digest"):
            verify_inference(reference, self.root / "missing_digest_output")
        self.assertFalse((self.root / "missing_digest_output").exists())
        record_path.write_text(original_record)
        record = json.loads(record_path.read_text())
        selected = next(exp for exp in record["experiments"]
                        if exp["name"] == record["selection"]["selected"])
        selected["preprocessing"]["implementation_sha256"] = "changed"
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Preprocessing differs"):
            verify_inference(reference, self.root / "invalid_output")
        self.assertFalse((self.root / "invalid_output").exists())

    def model_labels(self, models):
        return predict_texts(EXAMPLE_TEXTS, model_path=models / "selected.joblib")


if __name__ == "__main__":
    unittest.main()
