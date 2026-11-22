# IMDB movie-review sentiment classifier

A complete educational text classification workflow: validated data → reusable
cleaning → features → baseline → three improvement attempts → evaluation and
error analysis → a saved pipeline that predicts sentiment for new reviews.
Input is an English movie-review string; output is **0 = negative** or
**1 = positive**. There is no neutral or mixed class.

## Project structure

| Location | Purpose |
| --- | --- |
| `load_data.py`, `analyze_data.py` | Download/cache, validate, and explore the pinned IMDB dataset |
| `preprocess.py`, `prepare_data.py` | Shared cleaner, paired filtering, prepared data and inspection report |
| `train_baseline.py` | Reproducible split, raw-text pipeline, baseline evaluation and persistence |
| `train_improved.py` | Three fixed improvement attempts, comparable metrics, selection and persistence |
| `error_analysis.py` | Verify selected predictions and generate held-out error diagnostics |
| `predict.py`, `verify_inference.py` | Unlabeled inference API/CLI and persistence/demo verification |
| `examples/new_reviews.json` | Six new, unlabeled demo inputs |
| `tests/` | Dataset, preprocessing, training, evaluation and inference checks |
| `reports/` | Measured metrics, plots, experiment records, errors and demo predictions |
| `data/` | Generated raw cache and prepared Parquet files (ignored by Git) |
| `models/` | Generated complete fitted joblib pipelines |
| `requirements.txt` | Pinned direct dependencies |

## Setup and complete reproduction

Tested with **Python 3.14.4**. Run all commands from the project directory.
The first data download needs internet access; no account or token is required.
Create an environment and install the pinned packages:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the complete workflow in this order:

```bash
.venv/bin/python analyze_data.py
.venv/bin/python prepare_data.py --offline
.venv/bin/python train_baseline.py --offline
.venv/bin/python train_improved.py --offline
.venv/bin/python error_analysis.py --offline
.venv/bin/python verify_inference.py
.venv/bin/python predict.py --input-json examples/new_reviews.json
.venv/bin/python predict.py --text "Wonderful acting!" --text "Boring and disappointing."
.venv/bin/python -m unittest discover -s tests -v
```

The first command downloads any missing original splits and writes EDA outputs.
For a completely cached run, use `analyze_data.py --offline` instead. Subsequent
commands above require that cache. Omit `--offline` on a data-consuming command
to permit downloading missing splits. Downloads use temporary files and cache
successful files by revision. A damaged cached file must be deleted and downloaded
again; `--offline` fails clearly when a required file is absent.

Training produces `models/baseline.joblib` and `models/selected.joblib`. Both
contain preprocessing, the fitted vectorizer and the classifier. A fresh checkout must run the training commands
before inference. Reports are regenerated in place. The commands rerun experiments
with the existing split and settings; no manual notebook execution is needed.

Prepared Parquet files are an inspectable output of the same shared cleaner.
Training deliberately consumes retained **raw** texts through that cleaner inside
the pipeline, rather than reading already-cleaned files and cleaning twice.
`train_improved.py` requires the baseline report and saved split before it runs;
error analysis and verification require the selected-model experiment record.

Each script supports `--help`. Data-consuming commands accept `--cache-dir PATH`.
EDA and training reports default to `reports`; prepared data to `data/prepared`;
models to `models`. Training accepts `--output-dir` and `--model-dir`.
Preparation instead uses `--output-dir` for data and `--report-dir` for reports.
If relocating outputs, pass `--reference-dir` to later stages for baseline records,
`--experiment-dir` for improvement records, and `--model-path` to error analysis,
verification or prediction for a relocated artifact. Relative paths resolve from
the working directory; the default commands above need no path configuration.

## Dataset and preprocessing

The source is [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb),
pinned to revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` in
`load_data.py`. Files are cached under `data/imdb/<revision>/`. Columns are
`text` (review string) and `label` (integer sentiment).

| Official split | Rows | Negative | Positive | Use |
| --- | ---: | ---: | ---: | --- |
| Train | 25,000 | 12,500 | 12,500 | 20,000 training + 5,000 validation |
| Test | 25,000 | 12,500 | 12,500 | Loaded/validated, reserved; not scored |
| Unsupervised | 50,000 | — | — | EDA only; sentinel label -1 is not a class |

The loader validates required columns, nonempty data, missing values, text types,
blank texts, integer labels and allowed values. EDA records class balance, lengths,
duplicates and examples. No missing/blank texts or invalid labels were observed.
Training has 96 exact duplicate rows beyond first occurrences; there are 123 distinct
raw texts shared by official train and test. Official boundaries remain unchanged;
internal training/validation partitions keep identical cleaned texts together.
See [dataset analysis](reports/dataset_analysis.md) and
[distribution plots](reports/distributions.png).

`preprocess_text` removes HTML/comments/script/style content, decodes entities,
applies Unicode NFKC and lowercase, normalizes apostrophes/control characters,
and collapses whitespace. It preserves negation, punctuation, numbers and accents;
there is no stemming, lemmatization or stopword removal. No parameters are learned
by cleaning. `prepare_data` filters invalid labels and unusable texts together,
recording rejected source rows and reasons. All 25,000 rows in each supervised
split survived preparation. See [before/after examples](reports/preprocessing.md)
and [manual review](reports/preprocessing_review.md).

## Evaluation and selected model

A stratified grouped split of the official training data uses seed **42** and the
first of five StratifiedGroupKFold folds: 20,000 training reviews and 5,000 validation
reviews (2,500 per sentiment). Groups use the stateless cleaned text, so differently
formatted copies stay together even if their labels disagree. The cleaned training
and validation text sets are checked to be disjoint. Grouping targets an 80/20
split and class balance; exact proportions are not guaranteed for other inputs.
All approaches use the exact ordered assignments in `reports/baseline_split.csv`. Vectorizers
and classifiers fit only training rows; validation only calls fitted transformations.
Input fingerprints, cleaning hashes, settings and baseline reproduction are checked
before improvement training. Full settings and runtime versions are recorded in
[baseline_results.json](reports/baseline_results.json) and
[improvement_results.json](reports/improvement_results.json).

These are **validation results**, including the data used to select the winner;
they are not an independent final test estimate. TASK_7 integrates the existing
evaluation without introducing another model search. The official test set
remains unscored. Future tuning should use validation/CV within the 20,000-row
training partition before a separate final test evaluation.

| Approach | Accuracy | Macro precision | Macro recall | Macro F1 | Δ macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline: TF-IDF unigrams + Logistic Regression | 0.884600 | 0.884736 | 0.884600 | 0.884590 | — |
| **Selected: TF-IDF unigrams/bigrams + Logistic Regression** | **0.895200** | **0.895334** | **0.895200** | **0.895191** | **+0.010601** |
| Unigram counts + Multinomial Naive Bayes | 0.848600 | 0.850647 | 0.848600 | 0.848379 | -0.036211 |
| TF-IDF unigrams + LinearSVC | 0.890600 | 0.890639 | 0.890600 | 0.890597 | +0.006007 |

Accuracy is the fraction correct. Precision measures how often a predicted class
is correct; recall measures how many examples of an actual class are found. F1 is
the harmonic mean of precision and recall. Macro metrics average the two class
scores equally, making **macro F1** the selection criterion. Classes are balanced;
per-class support and scores are also saved. Deltas above are absolute F1 changes,
not relative percentages. The selected gain is about **1.06 percentage points**;
improvement is not guaranteed, as the Naive Bayes attempt shows.

The baseline uses up to 20,000 TF-IDF unigram features (`min_df=2`, L2 normalization,
smoothed IDF) and Logistic Regression (`C=1`, `lbfgs`, `max_iter=1000`, seed 42).
The selected variant adds adjacent word pairs and raises the feature cap to 40,000;
other Logistic Regression settings stay fixed. It has the highest measured macro
F1 of the four predefined approaches, at the cost of more features. Ties favor the
baseline, then candidate declaration order. No hyperparameter search is performed.
The saved winner remains fitted on the same 20,000 training rows; it is not refitted
on validation data. Timing and feature counts appear in the
[comparison CSV](reports/improvement_comparison.csv) and
[improvement report](reports/improvements.md); the
[baseline report](reports/baseline.md) preserves the reference.

## Errors and practical limitations

The selected pipeline makes **524 / 5,000 errors (10.48%)**. Confusion counts:

| True sentiment | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| Negative | 2,215 | 285 |
| Positive | 239 | 2,261 |

See the [confusion matrix plot](reports/confusion_matrix.png),
[generated diagnostics](reports/error_analysis.md),
[full error CSV](reports/misclassified_reviews.csv), and
[interpretation of eight inspected mistakes](reports/error_review.md).
Representative source rows refer to the original official training split:

- Rows 3116 and 5577 sound favorable despite negative source labels: possible label
  disagreement, requiring independent annotation before treating them as label errors.
- Row 1127 criticizes characterization and filmmaking across a long plot discussion:
  negative → positive.
- Row 17620 praises a comedy while objecting to its poor ratings: positive → negative.
- Row 17790 enjoys an unintentionally funny thriller while criticizing its acting
  and plausibility: positive → negative.
- Row 20077 praises a performance, then moves into a long discussion of capital
  punishment: positive → negative. Topic vocabulary does not directly establish sentiment.

The eight length-spaced errors are an inspection sample, not category-frequency
estimates or proof of which features caused predictions. Unordered unigram/bigram
features have limited discourse, quotation, and negation-scope representation.
Default word tokenization discards punctuation and splits contractions despite
the cleaner preserving them. Mixed opinions must receive one binary label; there
is no neutral class, uncertainty output or abstention mechanism. Sarcasm is a
plausible limitation, but was not established as a recurring inspected pattern.

Performance on other languages, financial news or other domains is unmeasured.
Even simple movie reviews may fail. Identical cleaned texts cannot cross the
internal evaluation boundary. Near-duplicates are not detected, repeated rows within a partition still carry repeated weight, and
model selection on this holdout makes its score optimistic as an estimate of unseen
performance. Out-of-vocabulary words are ignored; an input
containing only unknown tokens can receive an uninformative prediction from a
zero feature vector. Aggregate validation metrics do not guarantee reliability
for an individual review.

## Inference and reproducible demos

Once training has created `models/selected.joblib`, inference needs no dataset,
network or training. Run from the project directory with the pinned environment
and project modules available: joblib imports the shared custom transformer from
`train_baseline.py`, which uses `preprocess.py` and project imports. Load only
trusted joblib artifacts. Missing artifacts cause an error; inference does not
silently train a replacement.

The two CLI commands in the workflow demonstrate file and inline input. The
inline command prints:

```json
[
  {"text": "Wonderful acting!", "predicted_label": 1, "predicted_sentiment": "positive"},
  {"text": "Boring and disappointing.", "predicted_label": 0, "predicted_sentiment": "negative"}
]
```

`examples/new_reviews.json` is a UTF-8 JSON array of the following six raw inputs.
The saved selected pipeline produced these **unlabeled demonstration predictions**;
there are no verified targets and no accuracy claim for this table.

| Input text | Predicted label | Predicted sentiment |
| --- | ---: | --- |
| `Wonderful acting!<br />I loved every minute of this movie.` | 1 | positive |
| `A boring, predictable mess. The acting was terrible and I wanted it to end.` | 0 | negative |
| `I don’t usually like romantic movies, but this one was delightful.` | 0 | negative |
| `The cast was talented, but the dull story made this a disappointing film.` | 0 | negative |
| `An excellent, moving story with beautiful music. Highly recommended!` | 1 | positive |
| `Great visuals, weak dialogue. I have mixed feelings about this movie.` | 1 | positive |

The romantic-movie example shows a counterintuitive negative prediction; the mixed
example illustrates forced binary output. `verify_inference.py` records the same
six examples in [inference_examples.json](reports/inference_examples.json), together
with artifact hash, training versions, preprocessing provenance and reload checks.
It first requires the artifact SHA-256 to match the digest recorded when Task 4
saved the selected fitted pipeline. Identical copies at a different `--model-path`
pass; different fitted state or missing digest fails before a verification report
is written. Older records without a digest require rerunning improvement training.
It saves/reloads a temporary copy, requires identical predictions, and checks fitted
state and artifact bytes are unchanged. Training also checks original-fitted versus
reloaded predictions on all 5,000 validation reviews.

For reuse in Python:

```python
from predict import load_model, predict_texts
from load_data import LABELS

model = load_model("models/selected.joblib")
labels = predict_texts(["Wonderful acting!", "Boring and disappointing."], model)
print(labels)                      # [1, 0]
print([LABELS[x] for x in labels])  # ['positive', 'negative']
```

The API accepts a list or tuple of raw strings and returns ordered Python integer
labels. Duplicates and order are preserved; an empty batch returns `[]`. A bare
string, non-string elements, blanks or text without letters/digits after cleaning
raise `ValueError`; invalid batches are not silently filtered. The CLI accepts
exactly one of repeated `--text` or `--input-json PATH`, with optional
`--model-path PATH`; it prints JSON to stdout and errors to stderr with nonzero
exit status. Only fitted transformations and prediction run at inference time.

## Verification record

See [TASK_7 verification](reports/final_verification.md) for the executed setup,
full-data workflow, measured-result checks, generated artifacts and test outcome.
Semantic notes in `reports/preprocessing_review.md` and `reports/error_review.md`
are maintained separately from generated diagnostics; review them if data, cleaning
or model settings change.
