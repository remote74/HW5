"""Classify unlabeled movie reviews with the fitted Task 4 pipeline."""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from load_data import LABELS
from train_baseline import TextPreprocessor

DEFAULT_MODEL_PATH = Path("models/selected.joblib")


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a trusted fitted pipeline; keep project modules importable.

    The artifact contains the stateless cleaner, fitted vocabulary/IDF, and
    classifier. Numeric IMDB labels retain their meaning in load_data.LABELS.
    """
    model = joblib.load(model_path)
    if (not isinstance(model, Pipeline)
            or list(model.named_steps) != ["preprocess", "tfidf", "classifier"]
            or not isinstance(model.named_steps["preprocess"], TextPreprocessor)):
        raise ValueError("Expected the complete raw-text classification pipeline")
    check_is_fitted(model.named_steps["tfidf"], "vocabulary_")
    check_is_fitted(model.named_steps["classifier"])
    if model.classes_.tolist() != list(LABELS):
        raise ValueError("Model classes differ from the IMDB label mapping")
    return model


def predict_texts(texts: list[str] | tuple[str, ...], model: Pipeline | None = None,
                  *, model_path: str | Path = DEFAULT_MODEL_PATH) -> list[int]:
    """Return one integer label per raw input, in order, without fitting.

    Accept a list or tuple of strings, including an empty batch (returns []).
    Invalid containers/elements or texts empty after cleaning raise ValueError;
    no rows are silently dropped. Pass a loaded model to reuse it across batches,
    otherwise model_path is loaded for each nonempty call.
    """
    if not isinstance(texts, (list, tuple)) or any(not isinstance(t, str) for t in texts):
        raise ValueError("Pass a list or tuple of raw text strings")
    if not texts:
        return []
    pipeline = load_model(model_path) if model is None else model
    return [int(label) for label in pipeline.predict(texts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Raw text; repeat for a batch")
    source.add_argument("--input-json", type=Path, help="UTF-8 JSON array of raw strings")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    try:
        texts = (json.loads(args.input_json.read_text(encoding="utf-8"))
                 if args.input_json is not None else args.text)
        labels = predict_texts(texts, model_path=args.model_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps([
        {"text": text, "predicted_label": label, "predicted_sentiment": LABELS[label]}
        for text, label in zip(texts, labels)
    ], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
