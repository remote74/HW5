# Task 3 — baseline reference

TF-IDF word unigrams (maximum 20,000 features, min_df=2, L2 normalization,
smoothed IDF, standard two-or-more-character word tokens) feed Logistic Regression
(C=1, lbfgs, max_iter=1000, no class weighting, seed 42). Task 2's stateless cleaner
runs inside the pipeline; TF-IDF lowercasing is disabled because cleaning does it.
Default word tokens discard punctuation and split contractions; this is a baseline limitation.
All vectorizer and classifier parameters and package versions are in `baseline_results.json`.

## Evaluation conditions

Source: `stanfordnlp/imdb` revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`, official **training** split.
After Task 2 filtering, 20,000 rows train the model and 5,000
rows form an approximately 20% stratified grouped validation holdout (seed 42);
0 rows were rejected. StratifiedGroupKFold uses five folds (fewer for
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
| accuracy | 0.884600 |
| precision_macro | 0.884736 |
| recall_macro | 0.884600 |
| f1_macro | 0.884590 |

Accuracy is the fraction of correct predictions. Precision measures how often a
predicted class is correct; recall measures how much of the actual class is found.
F1 is the harmonic mean of precision and recall. Macro metrics average the two
class scores equally; these classes are balanced. Undefined scores are set to zero.
Per-class scores and support are saved in JSON. Confusion matrix rows are true
labels and columns predicted labels, both ordered [negative (0), positive (1)]:

```text
[[2188  312]
 [ 265 2235]]
```

This is the reference baseline for subsequent improvement attempts. No tuning or
improvement experiments were performed. Unigrams lose word order and negation
scope; sarcasm, mixed sentiment, and domain shift remain limitations. Task 1 found
exact duplicates and official train/test overlap; official boundaries are preserved.
Internal partitions group identical cleaned reviews. Near-duplicates are not detected,
and repeated reviews within a partition still receive repeated weight.

## Persistence

The complete raw-text pipeline is saved to `models/baseline.joblib`. Reloaded predictions
matched all validation predictions. Keep the project modules importable when loading
and load only trusted joblib files. Unusable new inputs raise ValueError.
