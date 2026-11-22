# Task 4 — baseline improvement experiments

## Evaluation and selection

All four pipelines use the pinned IMDB official training split, the shared Task 2
cleaner, and Task 3's exact stratified grouped approximately 80/20 assignments (seed 42):
20,000 training rows and 5,000 validation rows.
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

- **baseline**: Task 3 reference: TF-IDF unigrams with Logistic Regression.
- **word_bigrams**: Add adjacent-word context such as 'not good'; increase the feature cap from 20,000 to 40,000 to accommodate unigrams and bigrams.
- **count_naive_bayes**: Replace TF-IDF with unigram counts and Logistic Regression with MultinomialNB (alpha=1): test a simple generative classifier.
- **linear_svm**: Keep baseline TF-IDF and replace Logistic Regression with LinearSVC (C=1, max_iter=5000, seed 42): test a margin-based classifier.

All unchanged parameters inherit the baseline; full estimator parameters,
preprocessing details, versions, per-class metrics, and confusion matrices
(true rows / predicted columns, negative then positive) are in `improvement_results.json`.
The pipeline step named `tfidf` contains CountVectorizer in the Naive Bayes candidate.

## Measured validation comparison

| Experiment | Accuracy | Macro precision | Macro recall | Macro F1 | Delta F1 | Features | Fit s | Predict s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.884600 | 0.884736 | 0.884600 | 0.884590 | +0.000000 | 20,000 | 8.52 | 1.33 |
| word_bigrams | 0.895200 | 0.895334 | 0.895200 | 0.895191 | +0.010601 | 40,000 | 13.76 | 1.96 |
| count_naive_bayes | 0.848600 | 0.850647 | 0.848600 | 0.848379 | -0.036211 | 20,000 | 5.35 | 1.29 |
| linear_svm | 0.890600 | 0.890639 | 0.890600 | 0.890597 | +0.006007 | 20,000 | 5.89 | 1.37 |

- **word_bigrams** helped relative to baseline (delta +0.010601).
- **count_naive_bayes** did not improve macro F1 relative to baseline (delta -0.036211).
- **linear_svm** helped relative to baseline (delta +0.006007).

**Selected: word_bigrams**, with macro F1 0.895191.
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
- `models/selected.joblib`: complete selected raw-text pipeline, fitted only on training rows.
  Reloaded predictions matched every validation prediction; import project modules
  and load only trusted joblib files. The Task 3 reports and model are preserved.
  Reuse `baseline_split.csv` from the baseline reference directory for later analysis.
