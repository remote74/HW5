"""Verify persistence on new unlabeled examples and record reloaded predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import joblib

from load_data import LABELS
from predict import load_model, predict_texts
from train_improved import readable_params

EXAMPLE_TEXTS = [
    "Wonderful acting!<br />I loved every minute of this movie.",
    "A boring, predictable mess. The acting was terrible and I wanted it to end.",
    "I don’t usually like romantic movies, but this one was delightful.",
    "The cast was talented, but the dull story made this a disappointing film.",
    "An excellent, moving story with beautiful music. Highly recommended!",
    "Great visuals, weak dialogue. I have mixed feelings about this movie.",
]


def verify_inference(experiment_dir: str | Path = "reports",
                     output_dir: str | Path = "reports",
                     model_path: str | Path | None = None) -> dict:
    """Round-trip the existing fitted model without accessing data or training.

    Task 4 already checks original-fitted versus saved validation predictions.
    Require its recorded artifact digest before attributing that historical check.
    Here the selected fitted artifact is saved again to a temporary location,
    reloaded, and compared on new inputs. The selected artifact is preserved.
    """
    record = json.loads((Path(experiment_dir) / "improvement_results.json").read_text(
        encoding="utf-8"))
    selected = next(exp for exp in record["experiments"]
                    if exp["name"] == record["selection"]["selected"])
    path = Path(model_path if model_path is not None else record["persistence"]["model_path"])
    original_bytes = path.read_bytes()
    digest = hashlib.sha256(original_bytes).hexdigest()
    if not record["persistence"].get("model_sha256"):
        raise ValueError("Missing selected artifact digest; rerun improvement training")
    if digest != record["persistence"]["model_sha256"]:
        raise ValueError("Artifact digest differs from the selected fitted model")
    model = load_model(path)
    cleaning_hash = hashlib.sha256(Path(__file__).with_name("preprocess.py").read_bytes()).hexdigest()
    if selected["preprocessing"]["implementation_sha256"] != cleaning_hash:
        raise ValueError("Preprocessing differs from the selected training configuration")
    if record["label_mapping"] != {str(key): value for key, value in LABELS.items()}:
        raise ValueError("Label mapping differs from the selected training configuration")
    for step, section in [("tfidf", "vectorizer"), ("classifier", "classifier")]:
        estimator = model.named_steps[step]
        if (type(estimator).__name__ != selected[section]["type"]
                or readable_params(estimator) != selected[section]["params"]):
            raise ValueError("Model settings differ from the selected training configuration")
    original_predictions = predict_texts(EXAMPLE_TEXTS, model)
    state = joblib.hash(model)
    with tempfile.TemporaryDirectory() as directory:
        temporary_model = Path(directory) / "selected.joblib"
        joblib.dump(model, temporary_model)
        reloaded = load_model(temporary_model)
        predictions = predict_texts(EXAMPLE_TEXTS, reloaded)
        if predictions != original_predictions:
            raise RuntimeError("Reloaded predictions differ from the fitted pipeline")
        if joblib.hash(reloaded) != state:
            raise RuntimeError("Inference changed fitted model state")
    if path.read_bytes() != original_bytes:
        raise RuntimeError("Inference changed the saved artifact")
    result = {
        "experiment": "task_6_inference", "selected_model": selected["name"],
        "model_path": str(path), "model_sha256": digest, "selected_artifact_digest_verified": True,
        "label_mapping": record["label_mapping"], "training_versions": record["versions"],
        "preprocessing": selected["preprocessing"],
        "task_4_original_validation_reload_check": record["persistence"]["reload_predictions_identical"],
        "new_examples_reload_predictions_identical": True,
        "fitted_state_unchanged": True, "saved_artifact_unchanged": True,
        "interpretation": "New unlabeled usage examples; no verified targets or accuracy claim. "
                          "Mixed sentiment still receives one of the two classes. See Task 5 error review.",
        "examples": [{"text": text, "predicted_label": label,
                      "predicted_sentiment": LABELS[label]}
                     for text, label in zip(EXAMPLE_TEXTS, predictions)],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inference_examples.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="reports")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-path")
    args = parser.parse_args()
    result = verify_inference(args.experiment_dir, args.output_dir, args.model_path)
    print(json.dumps(result["examples"], indent=2, ensure_ascii=False))
    print(f"Verified reload; examples: {Path(args.output_dir) / 'inference_examples.json'}")


if __name__ == "__main__":
    main()
