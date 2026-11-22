# TASK_7 — integration verification after P2 fixes

Verified on **2026-11-21** with the existing Python **3.14.4** environment and
pinned dependencies. This record supersedes the earlier row-wise-split model
snapshot. `REVIEW.md` remains the historical review that identified both P2 issues.

## Executed workflow

All commands ran from the project directory and exited successfully. Plotting
configuration used `MPLCONFIGDIR=/tmp/hw5-fix-mpl`.

```bash
.venv/bin/python train_baseline.py --offline
.venv/bin/python train_improved.py --offline
.venv/bin/python error_analysis.py --offline
.venv/bin/python verify_inference.py
.venv/bin/python predict.py --input-json examples/new_reviews.json
.venv/bin/python predict.py --text "Wonderful acting!" --text "Boring and disappointing."
.venv/bin/python -m unittest discover -s tests -v
```

Setup, dependency installation, remote downloading, EDA and prepared-data generation
were not rerun for this fix. Existing validated cached data and the unchanged
stateless cleaner were reused. Official test boundaries were preserved; the test
split was loaded/validated by the shared loader but neither fitted nor scored.

## Duplicate grouping and regenerated evaluation

The internal split now takes the first of five StratifiedGroupKFold folds with
seed 42, grouping by stateless cleaned text regardless of label. All usable
25,000 official training rows remain assigned exactly once. There are 20,000
training rows (10,000 per class) and 5,000 validation rows (2,500 per class).
Independent checks using saved assignments confirm zero cleaned-text overlap.
Group sizes may prevent exact balance or an exact 20% holdout on other inputs.

Baseline assignments/model/results, all four experiment results, the selected
model and validation predictions, error diagnostics/plot/CSVs, semantic error
interpretation and inference records were refreshed together. The baseline macro
F1 is **0.8845898023549361**. **word_bigrams** remains selected, macro F1
**0.8951911289771566**, an absolute gain of **0.010601326622220464**.

The selected model makes **524 / 5,000 errors (10.48%)**, confusion counts
`[[2215, 285], [239, 2261]]` (true rows / predicted columns, negative then positive).
Error analysis reproduces every saved validation prediction without refitting.
Both training stages verify all validation predictions after artifact reload.
These remain model-selection validation scores, not independent final test scores.
Changing assignments also changes which reviews are evaluated; score differences
from the earlier snapshot do not isolate the effect of removing duplicate leakage.

## Selected artifact identity and inference

Selected artifact SHA-256:
`720366899987f8d4dd689ff80aa3f0bd61e331cc41b39817f5307361855b26d4`.

Task 4 records this digest immediately after saving the selected fitted pipeline.
Inference verification requires the digest to match before loading and attributing
training provenance or historical reload checks to the supplied artifact. A
byte-identical relocated copy passes; a changed fitted model or missing digest
raises ValueError before writing a successful report. Old records without a digest
require rerunning improvement training.

The selected training record, error record and inference record all match the
current artifact digest. Temporary save/reload preserves demo predictions and
fitted state, and inference preserves the original artifact bytes. Both CLI input
forms and the README Python API example succeed. The six demo predictions remain
`[1, 0, 0, 0, 1, 1]`, matching the README and inference JSON. Demos are unlabeled
usage examples, with no accuracy claim. All local README links resolve.

## Automated regression checks

**30 tests passed**. New coverage includes raw duplicates, formatting variants
that clean identically, conflicting labels within a group, duplicate caller
indexes, reproducibility, full row coverage, insufficient groups, relocated
artifact copies, missing digests, and the review's coefficient/intercept-negation
reproduction. That modified model has unchanged constructor settings and different
predictions, and verification rejects it without creating an output directory.
Existing checks cover training-only vocabulary/IDF fitting, provenance drift,
selection, error alignment, no-refit inference and fresh-process CLI loading.
Error-analysis fixtures now use distinct review texts so intentional sentiment
contradictions exercise error reporting without violating the grouping boundary.

## Remaining limitations

Official train/test overlap remains part of the original dataset; no independent
official-test estimate is claimed. Near-duplicates are not detected, repeated rows
within a partition keep repeated weight, and selection on the validation holdout
can overfit it. Future tuning must use grouped training-only folds. The revised
`error_review.md` distinguishes observed text patterns, possible label disagreement
and untested explanations for the new representative mistakes.
