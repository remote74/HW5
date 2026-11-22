"""Compare predefined Task 4 candidates on the exact Task 3 validation rows."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from load_data import DATASET_ID, LABELS, REVISION, load_imdb
from train_baseline import SEED, VALIDATION_SIZE, build_pipeline, split_training_data


MOTIVATIONS = {
    "baseline": "Task 3 reference: TF-IDF unigrams with Logistic Regression.",
    "word_bigrams": "Add adjacent-word context such as 'not good'; increase the feature "
                    "cap from 20,000 to 40,000 to accommodate unigrams and bigrams.",
    "count_naive_bayes": "Replace TF-IDF with unigram counts and Logistic Regression "
                         "with MultinomialNB (alpha=1): test a simple generative classifier.",
    "linear_svm": "Keep baseline TF-IDF and replace Logistic Regression with LinearSVC "
                  "(C=1, max_iter=5000, seed 42): test a margin-based classifier.",
}


def build_candidates() -> dict:
    """Return fresh pipelines; settings are fixed before inspecting holdout scores."""
    return {
        "baseline": build_pipeline(),
        "word_bigrams": build_pipeline().set_params(
            tfidf__ngram_range=(1, 2), tfidf__max_features=40000),
        "count_naive_bayes": build_pipeline().set_params(
            tfidf=CountVectorizer(lowercase=False, max_features=20000, min_df=2),
            classifier=MultinomialNB(alpha=1.0)),
        "linear_svm": build_pipeline().set_params(
            classifier=LinearSVC(C=1.0, max_iter=5000, random_state=SEED)),
    }


def readable_params(estimator) -> dict:
    """Serialize the full settings, including vectorizer dtype and n-gram tuples."""
    params = estimator.get_params()
    if "dtype" in params:
        params["dtype"] = np.dtype(params["dtype"]).name
    return json.loads(json.dumps(params))


def load_reference(frame: pd.DataFrame, reference_dir: str | Path):
    """Fail before fitting if the input, cleaning, settings, or assignments drift."""
    directory = Path(reference_dir)
    reference = json.loads((directory / "baseline_results.json").read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(frame[["text", "label"]], index=True).values.tobytes()
    ).hexdigest()
    dataset = reference["dataset"]
    if (dataset["input_sha256"] != fingerprint or dataset["id"] != DATASET_ID
            or dataset["revision"] != REVISION or dataset["source_split"] != "train"
            or dataset["official_test_used"]):
        raise ValueError("Dataset differs from the Task 3 baseline reference")
    cleaning_hash = hashlib.sha256(Path(__file__).with_name("preprocess.py").read_bytes()).hexdigest()
    if reference["preprocessing"]["implementation_sha256"] != cleaning_hash:
        raise ValueError("Preprocessing differs from the Task 3 baseline reference")
    pipeline = build_pipeline()
    if (reference["vectorizer"] != readable_params(pipeline.named_steps["tfidf"])
            or reference["classifier"] != readable_params(pipeline.named_steps["classifier"])
            or reference["split"]["random_state"] != SEED
            or reference["split"]["validation_size"] != VALIDATION_SIZE):
        raise ValueError("Settings differ from the Task 3 baseline reference")
    train, validation, rejected = split_training_data(frame)
    expected = pd.concat([
        pd.DataFrame({"source_row_position": part.index, "partition": name,
                      "label": part["label"].to_numpy()})
        for name, part in [("train", train), ("validation", validation)]
    ], ignore_index=True)
    saved = pd.read_csv(directory / reference["split"]["row_positions"])
    if not expected.equals(saved) or rejected != dataset["rejected_rows"]:
        raise ValueError("Split assignments differ from the Task 3 baseline reference")
    return reference, train, validation


def select_best(experiments: list[dict]) -> dict:
    """Use macro F1, keeping the baseline (first) on ties or no improvement."""
    return max(experiments, key=lambda experiment: experiment["metrics"]["f1_macro"])


def train_improved(frame: pd.DataFrame, reference_dir: str | Path = "reports",
                   output_dir: str | Path = "reports", model_dir: str | Path = "models") -> dict:
    """Fit training rows only; compare once, preserve the baseline, save the winner."""
    reference, train, validation = load_reference(frame, reference_dir)
    experiments = []
    best_pipeline = None
    best_predictions = None
    best_score = -1.0
    for name, pipeline in build_candidates().items():
        print(f"Training {name}...", flush=True)
        started = perf_counter()
        pipeline.fit(train["text"], train["label"])
        fit_seconds = perf_counter() - started
        started = perf_counter()
        predictions = pipeline.predict(validation["text"])
        predict_seconds = perf_counter() - started
        report = classification_report(
            validation["label"], predictions, labels=list(LABELS),
            target_names=list(LABELS.values()), output_dict=True, zero_division=0)
        metrics = {"accuracy": float(accuracy_score(validation["label"], predictions)),
                   "precision_macro": report["macro avg"]["precision"],
                   "recall_macro": report["macro avg"]["recall"],
                   "f1_macro": report["macro avg"]["f1-score"]}
        if name == "baseline" and any(
                not np.isclose(value, reference["metrics"][metric], rtol=0, atol=1e-12)
                for metric, value in metrics.items()):
            raise ValueError("Rerun baseline metrics differ from the saved Task 3 reference")
        vectorizer, classifier = pipeline.named_steps["tfidf"], pipeline.named_steps["classifier"]
        experiments.append({
            "name": name, "motivation_and_changes": MOTIVATIONS[name],
            "preprocessing": reference["preprocessing"],
            "vectorizer": {"type": type(vectorizer).__name__, "params": readable_params(vectorizer)},
            "classifier": {"type": type(classifier).__name__, "params": readable_params(classifier)},
            "fitted_features": len(vectorizer.vocabulary_), "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(validation["label"], predictions,
                                                   labels=list(LABELS)).tolist(),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        })
        if metrics["f1_macro"] > best_score:
            best_pipeline, best_predictions = pipeline, predictions
            best_score = metrics["f1_macro"]
        print(f"  macro F1={metrics['f1_macro']:.6f}", flush=True)

    selected = select_best(experiments)
    output, models = Path(output_dir), Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    model_path = models / "selected.joblib"
    joblib.dump(best_pipeline, model_path)
    reloaded = joblib.load(model_path)
    if not np.array_equal(best_predictions, reloaded.predict(validation["text"])):
        raise RuntimeError("Reloaded selected model predictions differ")
    # Record the current runtime alongside the historical baseline runtime.
    import platform
    import sklearn
    result = {
        "experiment": "task_4_improvements", "dataset": reference["dataset"],
        "split": reference["split"], "baseline_reference": str(Path(reference_dir) / "baseline_results.json"),
        "baseline_versions": reference["versions"],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "joblib": joblib.__version__},
        "label_mapping": reference["label_mapping"],
        "selection": {"metric": "f1_macro", "selected": selected["name"],
                      "tie_policy": "Baseline first, then candidate declaration order",
                      "method": "One comparison of three predefined candidates; no hyperparameter search"},
        "experiments": experiments,
        "persistence": {"model_path": str(model_path),
                        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        "reload_predictions_identical": True},
    }
    (output / "improvement_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows = [{"experiment": exp["name"], **exp["metrics"],
             "delta_f1_macro": exp["metrics"]["f1_macro"] - experiments[0]["metrics"]["f1_macro"],
             "fitted_features": exp["fitted_features"], "fit_seconds": exp["fit_seconds"],
             "predict_seconds": exp["predict_seconds"], "selected": exp["name"] == selected["name"]}
            for exp in experiments]
    pd.DataFrame(rows).to_csv(output / "improvement_comparison.csv", index=False)
    pd.DataFrame({"source_row_position": validation.index, "true_label": validation["label"].to_numpy(),
                  "predicted_label": best_predictions}).to_csv(output / "selected_predictions.csv", index=False)
    write_report(result, rows, output)
    return result


def write_report(result: dict, rows: list[dict], output: Path) -> None:
    table = "\n".join(
        f"| {row['experiment']} | {row['accuracy']:.6f} | {row['precision_macro']:.6f} | "
        f"{row['recall_macro']:.6f} | {row['f1_macro']:.6f} | {row['delta_f1_macro']:+.6f} | "
        f"{row['fitted_features']:,} | {row['fit_seconds']:.2f} | {row['predict_seconds']:.2f} |"
        for row in rows)
    observations = "\n".join(
        f"- **{row['experiment']}** {'helped' if row['delta_f1_macro'] > 0 else 'did not improve macro F1'} "
        f"relative to baseline (delta {row['delta_f1_macro']:+.6f})."
        for row in rows[1:])
    motivations = "\n".join(f"- **{name}**: {motivation}" for name, motivation in MOTIVATIONS.items())
    selected = result["selection"]["selected"]
    text = f"""# Task 4 — baseline improvement experiments

## Evaluation and selection

All four pipelines use the pinned IMDB official training split, the shared Task 2
cleaner, and Task 3's exact stratified grouped approximately 80/20 assignments (seed 42):
{result['split']['train_rows']:,} training rows and {result['split']['validation_rows']:,} validation rows.
Input fingerprint, preprocessing hash, baseline settings, and ordered split rows
are checked before training. The rerun baseline must reproduce the recorded metrics.
Only training rows fit vocabulary, IDF, and classifiers. The official test set is
neither fitted nor scored (the shared loader still validates it).

The selection metric is **macro F1**: the mean of the negative and positive class
F1 scores, giving each sentiment equal weight. F1 balances precision (correctness
of predicted class membership) and recall (coverage of actual class members).
Accuracy is the fraction correct. Precision, recall, and F1 use the baseline's
macro averaging and zero_division=0; both classes are balanced.
Select the largest macro F1; retain the baseline on a tie or no improvement.
Candidates and parameters are predefined, with no hyperparameter search or
follow-up tuning on this holdout. Future tuning must use validation or CV within
the training partition. This holdout comparison is model selection, not an
unbiased final generalization estimate; the official test remains reserved.

## Motivations and changes

{motivations}

All unchanged parameters inherit the baseline; full estimator parameters,
preprocessing details, versions, per-class metrics, and confusion matrices
(true rows / predicted columns, negative then positive) are in `improvement_results.json`.
The pipeline step named `tfidf` contains CountVectorizer in the Naive Bayes candidate.

## Measured validation comparison

| Experiment | Accuracy | Macro precision | Macro recall | Macro F1 | Delta F1 | Features | Fit s | Predict s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{table}

{observations}

**Selected: {selected}**, with macro F1 {max(row['f1_macro'] for row in rows):.6f}.
This follows the largest measured macro F1 under the stated tie policy.

## Tradeoffs and limitations

The table measures feature counts and end-to-end fit/prediction time, including
cleaning; timings depend on hardware and load and are descriptive single runs.
Bigrams allow local word order at a larger feature budget (up to twice baseline).
Count/Naive Bayes changes both representation and classifier, so its result cannot
isolate either contribution; its independence assumption ignores word interactions.
LinearSVC changes only the classifier and has no predict_proba interface.
Unchanged cleaning and word tokenization still discard punctuation features and
split contractions. None of these approaches explicitly models sarcasm or long
negation scope. Identical cleaned reviews stay in one partition; near-duplicates are not detected.
Small score differences have no statistical-significance guarantee.

## Reusable artifacts

- `improvement_results.json`: all settings, measured results, selection, and reference provenance.
- `improvement_comparison.csv`: comparison table with selected flag and raw timings.
- `selected_predictions.csv`: selected validation predictions linked to original source row positions.
- `{result['persistence']['model_path']}`: complete selected raw-text pipeline, fitted only on training rows.
  Reloaded predictions matched every validation prediction; import project modules
  and load only trusted joblib files. The Task 3 reports and model are preserved.
  Reuse `baseline_split.csv` from the baseline reference directory for later analysis.
"""
    (output / "improvements.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/imdb")
    parser.add_argument("--reference-dir", default="reports")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    results = train_improved(load_imdb(args.cache_dir, offline=args.offline)["train"],
                             args.reference_dir, args.output_dir, args.model_dir)
    print(f"Selected: {results['selection']['selected']}")
    print(f"Comparison report: {Path(args.output_dir) / 'improvements.md'}")


if __name__ == "__main__":
    main()
