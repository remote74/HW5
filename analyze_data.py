"""Generate the reproducible Task 1 EDA report without modifying raw texts."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from load_data import DATASET_ID, LABELS, REVISION, load_imdb


def markdown_table(frame: pd.DataFrame) -> str:
    """Render small numeric tables without an optional pandas dependency."""
    def cell(value):
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def analyze(frames: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    quality, counts, lengths, examples = [], [], [], []
    word_lengths = {}
    for split, frame in frames.items():
        text = frame["text"]
        words = text.str.split().str.len()
        word_lengths[split] = words
        quality.append({
            "split": split, "rows": len(frame), "columns": len(frame.columns),
            "text_dtype": str(text.dtype), "label_dtype": str(frame["label"].dtype),
            "missing_text": int(text.isna().sum()),
            "missing_label": int(frame["label"].isna().sum()),
            "blank_text": int(text.str.strip().eq("").sum()),
            "invalid_labels": int((~frame["label"].isin([-1] if split == "unsupervised" else LABELS)).sum()),
            "duplicate_texts_beyond_first": int(text.duplicated().sum()),
            "conflicting_label_texts": int((frame.groupby("text")["label"].nunique() > 1).sum()),
            "texts_with_html_breaks": int(text.str.contains(r"<br\s*/?>", case=False, regex=True).sum()),
        })
        for label, count in frame["label"].value_counts().sort_index().items():
            counts.append({"split": split, "label": label,
                           "class": LABELS.get(label, "unlabeled"),
                           "count": int(count), "percent": 100 * count / len(frame)})
        for unit, values in (("characters", text.str.len()), ("whitespace_words", words)):
            stats = values.describe(percentiles=[.25, .5, .75, .9, .95, .99])
            lengths.append({"split": split, "unit": unit, **stats.to_dict()})
        if split == "train":
            for label in LABELS:
                for index, row in frame.loc[frame["label"].eq(label)].sample(2, random_state=42).iterrows():
                    examples.append({"split": split, "row": index, "label": label,
                                     "class": LABELS[label], "text": row["text"]})

    quality = pd.DataFrame(quality)
    counts = pd.DataFrame(counts)
    lengths = pd.DataFrame(lengths)
    for name, table in (("quality", quality), ("class_distribution", counts),
                        ("text_lengths", lengths), ("examples", pd.DataFrame(examples))):
        table.to_csv(output / f"{name}.csv", index=False)

    train_texts = set(frames["train"]["text"])
    test_texts = set(frames["test"]["text"])
    overlap = len(train_texts & test_texts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    supervised = counts[counts["split"].isin(["train", "test"])]
    supervised.pivot(index="split", columns="class", values="count").plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="Supervised class distribution", ylabel="Reviews", xlabel="Official split")
    for split, values in word_lengths.items():
        axes[1].hist(values, bins=60, histtype="step", label=split)
    axes[1].set(title="Raw text length distribution", xlabel="Whitespace-separated words per review",
                ylabel="Reviews (log scale)", yscale="log")
    axes[1].legend()
    fig.savefig(output / "distributions.png", dpi=150)
    plt.close(fig)

    sections = [
        "# IMDB: loading and initial analysis",
        "## Task and source\n\nPredict binary sentiment in English movie reviews from the `text` column: "
        "`label=0` means negative and `label=1` means positive. "
        "There is no neutral target class. The source also includes an unsupervised split "
        "whose `label=-1` means unlabeled; it is excluded from supervised classification.\n\n"
        f"Source: [Hugging Face {DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID}). "
        f"Pinned dataset revision: `{REVISION}`. Official split boundaries and raw texts are preserved.",
        "## Structure and validation\n\n" + markdown_table(quality) + "\n\n"
        f"Total loaded rows: {sum(map(len, frames.values())):,}; supervised rows: "
        f"{len(frames['train']) + len(frames['test']):,}. Required columns are `text` and `label`. "
        "The loader checks nonempty frames, required columns, missing values, string and nonblank texts, "
        "integer labels, and allowed label values before returning data. Duplicate counts are exact raw "
        "text duplicates beyond the first occurrence; conflicting-label counts refer to distinct raw texts.",
        "## Class distribution\n\n" + markdown_table(counts) + "\n\n"
        "Percentages use the row count of each split as denominator. The unlabeled rows are not a third class.",
        "## Text lengths\n\nLengths are measured on raw text before cleaning: characters use Python string "
        "length (including spaces and markup); words use whitespace splitting (`str.split()`), not linguistic "
        "tokenization. Attached HTML tags may therefore count as part of a word. Standard deviation is "
        "the sample standard deviation. Percentile columns describe the full split.\n\n" + markdown_table(lengths),
        "![Class counts and text lengths](distributions.png)",
        "## Labeled examples\n\nTwo training reviews per class sampled with `random_state=42`. "
        "Excerpts below are limited to 450 characters, with HTML escaped for display. "
        "Full original examples are in `examples.csv`. These illustrate the labels, rather than "
        "establishing how typical every review is.",
    ]
    import html
    for example in examples:
        raw = example["text"]
        excerpt = html.escape(raw[:450]).replace("\n", " ") + ("…" if len(raw) > 450 else "")
        sections.append(f"### Train row {example['row']} — {example['class']} ({example['label']})\n\n> {excerpt}")
    balanced = supervised.groupby("split")["count"].nunique().eq(1).all()
    sections.append(
        "## Initial observations and later-stage implications\n\n"
        f"- Both supervised splits {'are exactly balanced' if balanced else 'have unequal class counts'} "
        "across positive and negative reviews (see counts above). Class balance alone does not establish model quality.\n"
        f"- Validation passed for every split: no missing or blank texts and no invalid labels were found. "
        "This is a structural check, not a manual audit of sentiment annotation accuracy.\n"
        f"- HTML break markup occurs in {int(quality['texts_with_html_breaks'].sum()):,} reviews. "
        "Later preprocessing should handle markup consistently; this stage preserves it.\n"
        "- Lengths are right-skewed: the upper percentiles and maxima exceed the median substantially. "
        "Review length and unusually long inputs will matter when selecting text features.\n"
        f"- There are {int(quality['duplicate_texts_beyond_first'].sum()):,} within-split duplicate rows beyond "
        f"the first, {int(quality['conflicting_label_texts'].sum()):,} within-split distinct texts with conflicting "
        f"labels, and {overlap:,} distinct exact texts shared by train and test. "
        "These counts use unnormalized strings; near-duplicates are not detected. "
        "Review overlap and label conflicts before modeling; do not silently alter the official benchmark.\n"
        "- Movie-review sentiment is the scope of this dataset. Results on it would not establish performance "
        "on neutral reviews, other domains, or other languages.\n"
        "- Keep the official test split for final evaluation. In later stages, create a reproducible validation "
        "split from training data and fit learned transformations on training data only. "
        "No model, learned transformation, or text cleaning was fitted in this stage."
    )
    report = output / "dataset_analysis.md"
    report.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/imdb")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--offline", action="store_true", help="Use only already cached parquet files")
    args = parser.parse_args()
    frames = load_imdb(args.cache_dir, offline=args.offline, include_unsupervised=True)
    report = analyze(frames, args.output_dir)
    print(report.read_text(encoding="utf-8"))
    print(f"\nReport saved to {report.resolve()}")


if __name__ == "__main__":
    main()
