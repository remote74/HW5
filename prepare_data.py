"""Prepare official supervised IMDB splits and report Task 2 cleaning results."""

import argparse
import json
from pathlib import Path

import pandas as pd

from load_data import LABELS, REVISION, load_imdb
from preprocess import prepare_data, preprocess_text


def prepare_splits(frames: dict[str, pd.DataFrame], output_dir: str | Path,
                   report_dir: str | Path) -> Path:
    """Save prepared Parquet splits, label meanings, rejection audit, and examples."""
    output, reports = Path(output_dir), Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    summary, examples, rejections = [], [], []
    for split, frame in frames.items():
        prepared, rejected = prepare_data(frame)
        if prepared.empty:
            raise ValueError(f"No usable supervised rows remain in {split}")
        prepared.to_parquet(output / f"{split}.parquet", index=True)
        summary.append({
            "split": split, "input_rows": len(frame), "prepared_rows": len(prepared),
            "rejected_rows": len(rejected),
            "changed_texts": int((frame["text"] != frame["text"].map(preprocess_text)).sum()),
            **{name: int(prepared["label"].eq(label).sum()) for label, name in LABELS.items()},
        })
        rejected = rejected.copy()
        rejected.insert(0, "split", split)
        rejections.append(rejected)
        if split == "train":
            for label in LABELS:
                sample = frame.loc[frame["label"].eq(label)]
                for index, row in sample.sample(min(2, len(sample)), random_state=42).iterrows():
                    examples.append({"split": split, "row": index, "label": int(label),
                                     "original": row["text"], "processed": preprocess_text(row["text"])})
    mapping = {str(label): name for label, name in LABELS.items()}
    (output / "label_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    pd.concat(rejections).to_csv(reports / "preprocessing_rejections.csv", index=True, index_label="source_row")
    pd.DataFrame(summary).to_csv(reports / "preprocessing_summary.csv", index=False)
    pd.DataFrame(examples).to_csv(reports / "preprocessing_examples.csv", index=False)
    sections = [
        "# IMDB preprocessing — Task 2",
        f"Source: the Task 1 loader, pinned revision `{REVISION}`. Only official supervised splits "
        "are prepared. Original row indexes, order, and split boundaries are retained in Parquet. "
        "Raw cached files are unchanged.",
        "## Rules and label representation\n\n"
        "HTML tags become spaces; entities are decoded; comments and script/style contents are removed. "
        "Unicode NFKC normalization, lowercase, straight apostrophes, control-character replacement "
        "with spaces, and whitespace collapse follow. Keep negation, punctuation, numbers, and accents. "
        "A usable result must contain a Unicode letter or digit. Non-string, blank, or cleaned-empty "
        "texts and invalid labels are rejected together with their paired values.\n\n"
        "Labels remain integers: **0 = negative, 1 = positive**, saved as int64. "
        "`label_mapping.json` retains these meanings. No label encoder or learned transformation is fitted. "
        "The same label-independent `preprocess_text` function is available for inference; callers must "
        "reject its empty-string sentinel. Future learned transformations must fit only on training data.",
        "## Measured results\n\n```text\n" + pd.DataFrame(summary).to_string(index=False) + "\n```\n\n"
        "Changed texts counts all input rows. Rejection reasons and original values are in "
        "`preprocessing_rejections.csv` (source row indexes included). Duplicates and label conflicts "
        "are not removed; see the Task 1 report before modeling.",
        "## Before and after\n\nTwo training examples per class use seed 42. "
        "The excerpts below show at most 650 characters per field; full pairs are in "
        "`preprocessing_examples.csv`. JSON code blocks preserve punctuation and escape newlines.",
    ]
    for example in examples:
        pair = {key: example[key][:650] for key in ("original", "processed")}
        sections.append(f"### Training row {example['row']} — {LABELS[example['label']]}\n\n"
                        + "```json\n" + json.dumps(pair, indent=2, ensure_ascii=False) + "\n```")
    sections.append("## Unusable and unlabeled input checks\n\n"
                    "These synthetic inputs have no labels and exercise the inference function directly.")
    for raw in [None, 123, " \n\t", "<br /> &nbsp;", "!!!", "<script>bad()</script>",
                " ＧＲＥＡＴ<br />movie! &amp; I don’t dislike it. "]:
        sections.append("```json\n" + json.dumps({"original": raw, "processed": preprocess_text(raw)},
                                               ensure_ascii=False) + "\n```")
    sections.append("## Limitations\n\nLowercasing loses emphasis; NFKC and whitespace normalization "
                    "lose formatting. Punctuation-only input is intentionally unusable. HTML parsing is "
                    "best effort: literal angle-bracket phrases can look like markup, and malformed "
                    "script/style tags can suppress subsequent text. Cleaning cannot resolve sarcasm, "
                    "annotation conflicts, or domain shift. No feature extraction or training is performed.")
    report = reports / "preprocessing.md"
    report.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/imdb")
    parser.add_argument("--output-dir", default="data/prepared")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = prepare_splits(load_imdb(args.cache_dir, offline=args.offline),
                            args.output_dir, args.report_dir)
    print(report.read_text(encoding="utf-8"))
    print(f"\nReport saved to {report.resolve()}")


if __name__ == "__main__":
    main()
