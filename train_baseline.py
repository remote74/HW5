"""Train the Task 3 reference on an internal IMDB train/validation split."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from load_data import DATASET_ID, LABELS, REVISION, load_imdb
from preprocess import prepare_data, preprocess_text

SEED = 42
VALIDATION_SIZE = 0.2


class TextPreprocessor(TransformerMixin, BaseEstimator):
    """Apply the Task 2 cleaner and reject unusable inference inputs."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):
            raise ValueError("Pass a sequence of texts, not a single string")
        cleaned = [preprocess_text(text) for text in X]
        if any(not text for text in cleaned):
            raise ValueError("No usable text remains for one or more inputs")
        return cleaned


def build_pipeline() -> Pipeline:
    """Return an unfitted pipeline accepting raw, unlabeled review strings."""
    return Pipeline([
        ("preprocess", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(lowercase=False, ngram_range=(1, 1),
                                 max_features=20000, min_df=2)),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                          random_state=SEED)),
    ])


def split_training_data(frame: pd.DataFrame):
    """Filter with Task 2 rules, retaining raw inputs for the pipeline's cleaner.

    The internal holdout is the development test set for this stage. The official
    test split must remain untouched until final evaluation. Return original row
    positions as well as frames so later experiments can reuse exact assignments.
    """
    # Positional indexes avoid ambiguity if a caller supplies duplicate indexes.
    raw = frame.reset_index(drop=True)
    prepared, rejected = prepare_data(raw)
    counts = prepared["label"].value_counts()
    if set(counts.index) != set(LABELS) or counts.min() < 2:
        raise ValueError("Need both sentiment classes with at least two usable rows each")
    # Group by the actual stateless cleaner, including differently formatted copies
    # and copies with conflicting labels. Never split a group by its label.
    groups = pd.factorize(prepared["text"], sort=False)[0]
    class_groups = prepared.groupby("label")["text"].nunique()
    if class_groups.min() < 2:
        raise ValueError("Need at least two distinct cleaned-text groups per sentiment class")
    folds = min(round(1 / VALIDATION_SIZE), int(counts.min()), len(set(groups)))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    train_positions, validation_positions = next(splitter.split(
        prepared["text"], prepared["label"], groups=groups))
    train_rows = prepared.index.to_numpy()[train_positions]
    validation_rows = prepared.index.to_numpy()[validation_positions]
    overlap = set(prepared.loc[train_rows, "text"]) & set(prepared.loc[validation_rows, "text"])
    if overlap:
        raise RuntimeError("Cleaned training and validation texts overlap")
    if any(set(prepared.loc[rows, "label"]) != set(LABELS)
           for rows in (train_rows, validation_rows)):
        raise ValueError("Cannot create grouped partitions containing both sentiment classes")
    usable_raw = raw.copy()
    usable_raw.loc[prepared.index, "label"] = prepared["label"]
    train = usable_raw.loc[train_rows, ["text", "label"]].copy()
    validation = usable_raw.loc[validation_rows, ["text", "label"]].copy()
    train["label"] = train["label"].astype("int64")
    validation["label"] = validation["label"].astype("int64")
    return train, validation, len(rejected)


def train_baseline(frame: pd.DataFrame, output_dir: str | Path = "reports",
                   model_dir: str | Path = "models") -> dict:
    """Fit only training rows, evaluate the holdout, and persist the reference."""
    train, validation, rejected_count = split_training_data(frame)
    pipeline = build_pipeline()
    pipeline.fit(train["text"], train["label"])
    predictions = pipeline.predict(validation["text"])
    report = classification_report(validation["label"], predictions, labels=list(LABELS),
                                   target_names=list(LABELS.values()), output_dict=True,
                                   zero_division=0)
    # The hash binds assignments to exact input content and order, including labels.
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(frame[["text", "label"]], index=True).values.tobytes()
    ).hexdigest()
    results = {
        "experiment": "task_3_baseline", "reference_for_later_experiments": True,
        "dataset": {"id": DATASET_ID, "revision": REVISION,
                    "source_split": "train", "input_sha256": fingerprint,
                    "input_rows": len(frame), "rejected_rows": rejected_count,
                    "official_test_used": False},
        "split": {"method": "StratifiedGroupKFold; first fold; groups are stateless cleaned texts",
                  "cleaned_text_overlap": 0,
                  "random_state": SEED, "validation_size": VALIDATION_SIZE,
                  "row_positions": "baseline_split.csv",
                  "train_rows": len(train), "validation_rows": len(validation),
                  "class_counts": {name: {str(label): int(part["label"].eq(label).sum())
                                          for label in LABELS}
                                   for name, part in [("train", train), ("validation", validation)]}},
        "preprocessing": {"function": "preprocess.preprocess_text",
                          "rules": "Task 2: HTML removal, entity decoding, NFKC, lowercase, "
                                   "apostrophe/control/whitespace normalization; keep negation; "
                                   "reject inputs without letters or digits",
                          "implementation_sha256": hashlib.sha256(
                              Path(__file__).with_name("preprocess.py").read_bytes()).hexdigest()},
        "vectorizer": pipeline.named_steps["tfidf"].get_params(),
        "classifier": pipeline.named_steps["classifier"].get_params(),
        "fitted_features": len(pipeline.named_steps["tfidf"].vocabulary_),
        "classifier_iterations": pipeline.named_steps["classifier"].n_iter_.tolist(),
        "metrics": {"accuracy": float(accuracy_score(validation["label"], predictions)),
                    "precision_macro": report["macro avg"]["precision"],
                    "recall_macro": report["macro avg"]["recall"],
                    "f1_macro": report["macro avg"]["f1-score"]},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(validation["label"], predictions,
                                               labels=list(LABELS)).tolist(),
        "label_mapping": {str(key): value for key, value in LABELS.items()},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "joblib": joblib.__version__},
    }
    output, models = Path(output_dir), Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    model_path = models / "baseline.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    if not np.array_equal(predictions, reloaded.predict(validation["text"])):
        raise RuntimeError("Reloaded model predictions differ")
    results["persistence"] = {"model_path": str(model_path),
                              "reload_predictions_identical": True}
    pd.concat([
        pd.DataFrame({"source_row_position": part.index, "partition": name,
                      "label": part["label"].to_numpy()})
        for name, part in [("train", train), ("validation", validation)]
    ], ignore_index=True).to_csv(output / "baseline_split.csv", index=False)
    # JSON represents the tuple ngram_range as a list; dtype is recorded by name.
    results["vectorizer"]["dtype"] = np.dtype(results["vectorizer"]["dtype"]).name
    (output / "baseline_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    metrics = "\n".join(f"| {name} | {value:.6f} |" for name, value in results["metrics"].items())
    notes = f"""# Task 3 — baseline reference

TF-IDF word unigrams (maximum 20,000 features, min_df=2, L2 normalization,
smoothed IDF, standard two-or-more-character word tokens) feed Logistic Regression
(C=1, lbfgs, max_iter=1000, no class weighting, seed 42). Task 2's stateless cleaner
runs inside the pipeline; TF-IDF lowercasing is disabled because cleaning does it.
Default word tokens discard punctuation and split contractions; this is a baseline limitation.
All vectorizer and classifier parameters and package versions are in `baseline_results.json`.

## Evaluation conditions

Source: `{DATASET_ID}` revision `{REVISION}`, official **training** split.
After Task 2 filtering, {len(train):,} rows train the model and {len(validation):,}
rows form an approximately 20% stratified grouped validation holdout (seed 42);
{rejected_count} rows were rejected. StratifiedGroupKFold uses five folds (fewer for
small inputs) and takes the first holdout. Identical cleaned texts stay together,
even with conflicting labels; cleaned training/validation overlap is checked to be zero.
Group sizes can prevent exact target sizes and class proportions.
This holdout is the development test set for Task 3. The official test split is
reserved for final evaluation, and is neither fitted nor scored here.
Only the training partition fits vocabulary, IDF weights, and classifier parameters.
`baseline_split.csv` saves ordered zero-based source row positions and labels;
JSON records the input fingerprint and class counts. Reuse these exact conditions
for later experiments. Repeated holdout comparisons can overfit validation results.

## Measured validation results

| Metric | Value |
| --- | --- |
{metrics}

Accuracy is the fraction of correct predictions. Precision measures how often a
predicted class is correct; recall measures how much of the actual class is found.
F1 is the harmonic mean of precision and recall. Macro metrics average the two
class scores equally; these classes are balanced. Undefined scores are set to zero.
Per-class scores and support are saved in JSON. Confusion matrix rows are true
labels and columns predicted labels, both ordered [negative (0), positive (1)]:

```text
{np.array(results['confusion_matrix'])}
```

This is the reference baseline for subsequent improvement attempts. No tuning or
improvement experiments were performed. Unigrams lose word order and negation
scope; sarcasm, mixed sentiment, and domain shift remain limitations. Task 1 found
exact duplicates and official train/test overlap; official boundaries are preserved.
Internal partitions group identical cleaned reviews. Near-duplicates are not detected,
and repeated reviews within a partition still receive repeated weight.

## Persistence

The complete raw-text pipeline is saved to `{model_path}`. Reloaded predictions
matched all validation predictions. Keep the project modules importable when loading
and load only trusted joblib files. Unusable new inputs raise ValueError.
"""
    (output / "baseline.md").write_text(notes, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/imdb")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    # Keep custom transformers importable when joblib is loaded from another script.
    from train_baseline import train_baseline as run
    results = run(load_imdb(args.cache_dir, offline=args.offline)["train"],
                  args.output_dir, args.model_dir)
    print(json.dumps(results["metrics"], indent=2))
    print(f"Reference report: {Path(args.output_dir) / 'baseline.md'}")


if __name__ == "__main__":
    main()
