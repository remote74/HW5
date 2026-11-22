"""Diagnose the fitted Task 4 winner on the established validation holdout."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from load_data import LABELS, load_imdb
from train_improved import load_reference, readable_params


def representative_errors(errors: pd.DataFrame, per_pair: int = 4) -> pd.DataFrame:
    """Cover each observed confusion direction and a range of raw text lengths.

    Evenly spaced ranks include shortest and longest errors, with source position
    breaking length ties. This is an inspection sample, not a prevalence estimate.
    """
    if per_pair < 1:
        raise ValueError("per_pair must be positive")
    samples = []
    for _, group in errors.groupby(["true_label", "predicted_label"], sort=True):
        ordered = group.sort_values(["raw_word_count", "source_row_position"])
        ranks = np.linspace(0, len(ordered) - 1, min(per_pair, len(ordered)), dtype=int)
        samples.append(ordered.iloc[ranks])
    return pd.concat(samples, ignore_index=True) if samples else errors.iloc[:0].copy()


def plot_confusion(matrix: np.ndarray, output: Path) -> None:
    """Plot raw counts with stable sentiment ordering; no normalization."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    names = [f"{name} ({label})" for label, name in LABELS.items()]
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names, xlabel="Predicted label",
           ylabel="True label", title="Validation confusion matrix\nSelected model — raw counts")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, label="Reviews (not normalized)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def analyze_errors(frame: pd.DataFrame, reference_dir: str | Path = "reports",
                   experiment_dir: str | Path = "reports",
                   output_dir: str | Path = "reports",
                   model_path: str | Path | None = None, per_pair: int = 4) -> dict:
    """Reuse fitted transformations and fail on dataset, split, or prediction drift.

    Relative paths recorded by Task 4 are relative to the project working directory,
    as in the training commands. model_path can override the path after relocation.
    No fit or fit_transform is called here; the official test split is not scored.
    """
    if per_pair < 1:
        raise ValueError("per_pair must be positive")
    reference, _, validation = load_reference(frame, reference_dir)
    experiments = Path(experiment_dir)
    record = json.loads((experiments / "improvement_results.json").read_text(encoding="utf-8"))
    if (record["dataset"] != reference["dataset"] or record["split"] != reference["split"]
            or record["label_mapping"] != reference["label_mapping"]):
        raise ValueError("Task 4 dataset, split, or labels differ from the baseline reference")
    selected = next(exp for exp in record["experiments"]
                    if exp["name"] == record["selection"]["selected"])
    if selected["preprocessing"] != reference["preprocessing"]:
        raise ValueError("Selected preprocessing differs from the baseline reference")
    path = Path(model_path if model_path is not None else record["persistence"]["model_path"])
    pipeline = joblib.load(path)  # Only load trusted project artifacts.
    for step, settings in [("tfidf", "vectorizer"), ("classifier", "classifier")]:
        estimator = pipeline.named_steps[step]
        if (type(estimator).__name__ != selected[settings]["type"]
                or readable_params(estimator) != selected[settings]["params"]):
            raise ValueError("Loaded model settings differ from the selected Task 4 model")
    predictions = pipeline.predict(validation["text"])
    current = pd.DataFrame({"source_row_position": validation.index,
                            "true_label": validation["label"].to_numpy(),
                            "predicted_label": predictions})
    saved = pd.read_csv(experiments / "selected_predictions.csv")
    if not current.equals(saved):
        raise ValueError("Selected predictions or ordered validation rows differ from Task 4")
    matrix = confusion_matrix(current["true_label"], predictions, labels=list(LABELS))
    report = classification_report(current["true_label"], predictions, labels=list(LABELS),
                                   target_names=list(LABELS.values()), output_dict=True,
                                   zero_division=0)
    if (matrix.tolist() != selected["confusion_matrix"]
            or not np.isclose(report["macro avg"]["f1-score"],
                              selected["metrics"]["f1_macro"], rtol=0, atol=1e-12)):
        raise ValueError("Selected evaluation results differ from Task 4")
    current["text"] = validation["text"].to_numpy()
    current["true_sentiment"] = current["true_label"].map(LABELS)
    current["predicted_sentiment"] = current["predicted_label"].map(LABELS)
    current["raw_word_count"] = current["text"].str.split().str.len()
    errors = current.loc[current["true_label"].ne(current["predicted_label"])].copy()
    examples = representative_errors(errors, per_pair)
    summary = {
        "experiment": "task_5_error_analysis", "selected_model": selected["name"],
        "model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset": reference["dataset"], "split": reference["split"],
        "preprocessing": selected["preprocessing"],
        "task_4_record": str(experiments / "improvement_results.json"),
        "predictions_match_task_4": True, "evaluated_rows": len(current),
        "incorrect_predictions": len(errors), "error_rate": len(errors) / len(current),
        "label_order": list(LABELS.values()), "confusion_matrix": matrix.tolist(),
        "confusion_normalization": "none; raw counts, true rows and predicted columns",
        "classification_report": report,
        "representative_source_rows": examples["source_row_position"].tolist(),
        "sampling": f"Up to {per_pair} evenly spaced length ranks per observed confusion direction; "
                    "raw whitespace word counts, source position breaks ties",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output / "misclassified_reviews.csv", index=False)
    examples.to_csv(output / "representative_errors.csv", index=False)
    pd.DataFrame(matrix, index=[f"true_{name}" for name in LABELS.values()],
                 columns=[f"predicted_{name}" for name in LABELS.values()]).to_csv(
                     output / "confusion_matrix.csv", index_label="true / predicted")
    plot_confusion(matrix, output / "confusion_matrix.png")
    (output / "error_analysis_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, examples, output)
    return summary


def write_report(result: dict, examples: pd.DataFrame, output: Path) -> None:
    matrix = np.array(result["confusion_matrix"])
    class_rows = []
    for index, (label, name) in enumerate(LABELS.items()):
        support = int(matrix[index].sum())
        mistakes = support - int(matrix[index, index])
        class_rows.append(f"| {name} ({label}) | {support} | {mistakes} | {mistakes / support:.2%} |")
    example_sections = []
    for row in examples.itertuples(index=False):
        # Escape HTML in raw reviews so markup and quoted content render literally.
        quoted = "\n".join("> " + escape(line) for line in row.text.splitlines())
        example_sections.append(
            f"### Source row {row.source_row_position}: {row.true_sentiment} → {row.predicted_sentiment}\n\n"
            f"Raw text, {row.raw_word_count} whitespace-separated words:\n\n{quoted}")
    no_errors = "No incorrect predictions were found; there are no real errors to sample." if examples.empty else ""
    rendered_examples = "\n\n".join(example_sections)
    text = f"""# Task 5 — selected-model error diagnostics

## Model and evaluation provenance

Analyze **{result['selected_model']}**, loaded from `{result['model_path']}` without
refitting. Dataset: `{result['dataset']['id']}`, revision `{result['dataset']['revision']}`.
Use Task 3's exact {result['evaluated_rows']:,}-row validation holdout from the official
training split (stratified grouped approximately 20%, seed 42). Identical cleaned
texts stay together. The input fingerprint, cleaning code,
ordered split, selected estimator settings, saved predictions, and confusion counts
are checked against Tasks 3–4 before writing outputs. The official test set is
neither fitted nor scored. See `error_analysis_results.json` for hashes and provenance.

## Observed class confusion

There are **{result['incorrect_predictions']:,} incorrect predictions** out of
{result['evaluated_rows']:,} ({result['error_rate']:.2%}). The selected model's
macro F1 is {result['classification_report']['macro avg']['f1-score']:.6f}.

![Validation confusion matrix](confusion_matrix.png)

The plot and `confusion_matrix.csv` use **raw counts, no normalization**.
Rows are true labels and columns are predicted labels, ordered negative (0),
positive (1). Off-diagonal cells are mistakes. The rates below divide each class's
mistakes by its true support, rather than by predicted-class totals.

| True sentiment | Reviews | Incorrect | Class error rate |
| --- | --- | --- | --- |
{chr(10).join(class_rows)}

Both directions are reported even if one has no errors. Full precision, recall,
F1, and support are saved in JSON. A confusion count describes a label mismatch;
it does not by itself establish sarcasm, label noise, or a causal explanation.

## Representative real errors

{result['sampling']}. This deterministic sample covers each available direction
and text-length variation; it is not random and cannot estimate failure-category
frequency. Source rows are zero-based positions in the original official training
split. Full original text and both labels are preserved in `misclassified_reviews.csv`
(all errors) and `representative_errors.csv` (these examples).

{no_errors}
{rendered_examples}

## Interpretation and next steps

Read these texts alongside their true and predicted labels before attributing
causes. The separately maintained `error_review.md` records the inspected snapshot's
recurring failure cases, evidence, hypotheses, and proposed experiments. Refresh
that review after changing data or model; this script regenerates measured
diagnostics and examples, not semantic annotations.

This is the same validation holdout already used for Task 4 model selection;
its errors are diagnostic, not an unbiased final evaluation. Do not tune features,
thresholds, or parameters on these errors. Test future proposals with validation
or cross-validation **inside the training partition**, then reserve the official
test split for a final evaluation. Keep cleaned duplicates together in internal
folds. Near-duplicates are not detected, repeated rows still receive repeated
weight, and sentiment labels force mixed reviews into a binary decision.
"""
    (output / "error_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/imdb")
    parser.add_argument("--reference-dir", default="reports")
    parser.add_argument("--experiment-dir", default="reports")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--model-path", help="Override the trusted artifact path recorded in Task 4")
    parser.add_argument("--per-pair", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    result = analyze_errors(load_imdb(args.cache_dir, offline=args.offline)["train"],
                            args.reference_dir, args.experiment_dir, args.output_dir,
                            args.model_path, args.per_pair)
    print(f"{result['selected_model']}: {result['incorrect_predictions']} errors / "
          f"{result['evaluated_rows']} validation reviews")
    print(f"Diagnostics: {Path(args.output_dir) / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
