# Task 5 — interpretation of inspected errors

## Scope and evidence

Reviewed all eight full texts in `representative_errors.csv` and
[generated diagnostics](error_analysis.md) on 2026-11-21 after the P2 split fix.
The selected **word_bigrams** pipeline uses TF-IDF unigrams/bigrams with a
40,000-feature cap and Logistic Regression. It was fitted on 20,000 official
training rows and evaluated on 5,000 held-out rows, using seed 42 and the first
StratifiedGroupKFold fold. Identical cleaned texts stay within one partition.
No fitting occurred during error analysis. The official test split remains unscored.

This interpretation replaces the earlier review of the row-wise split. It binds to
model SHA-256 `720366899987f8d4dd689ff80aa3f0bd61e331cc41b39817f5307361855b26d4` and input SHA-256
`e4eca11b4f218baf836a69dfa9d66e545ab8966a9834ef8a988c541804f3836a`,
dataset revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`.
Source rows are zero-based positions in the original official training split.

## Measured errors

| True sentiment | Predicted negative | Predicted positive | Within-class error rate |
| --- | ---: | ---: | ---: |
| Negative | 2,215 | 285 | 11.40% |
| Positive | 239 | 2,261 | 9.56% |

There are **524 errors / 5,000 reviews (10.48%)**, macro F1 **0.895191**.
Both classes have 2,500 examples. The 46 additional negative-to-positive errors
show a modest directional difference; they do not establish statistical significance.
Negative recall is 88.60%, positive recall 90.44%. Counts are unnormalized.

The sample takes four evenly spaced length ranks per confusion direction,
spanning 23–1,263 raw whitespace words. It supports inspection, not category
frequency estimates or a conclusion that longer reviews fail more often.

## Representative mistakes

| Source row | True → predicted | Observation | Interpretation or uncertainty |
| --- | --- | --- | --- |
| 3116 | Negative → positive | Calls the movie great, particularly its music, and describes finding a special person. | The text appears favorable despite its negative source label. Possible annotation disagreement; no verified relabeling. |
| 5577 | Negative → positive | Calls Darkman 3 awesome, praises the trilogy, then discusses DVD availability and playback. | Another apparently favorable text with a negative label. Prediction may be reasonable; independent annotation is needed. |
| 915 | Negative → positive | Summarizes corrupt police and an investigation, praises some performances but says the director barely keeps the film afloat and one actor seems lost. | Mixed assessment and substantial plot summary obscure overall polarity. Individual feature effects were not measured. |
| 1127 | Negative → positive | Long review criticizes shallow characterization, camerawork and cultural portrayal, ending with dislike despite wanting to love the film. | Positive concessions and plot exposition coexist with an explicit negative verdict. The model may underweight the conclusion; this is a hypothesis. |
| 17620 | Positive → negative | Praises a comedy and urges viewers to laugh, while expressing anger at its poor ratings. | Criticism targets other ratings rather than the film. The classifier misses a clearly favorable assessment; attribution is a possible explanation. |
| 24224 | Positive → negative | Says the movie is awesome and watched repeatedly; criticizes opening dialogue, then discusses remakes, actors and religious reception. | Mixed sentiment and tangential discussion accompany an overall favorable opening. No exact feature cause was established. |
| 17790 | Positive → negative | Enjoys a thriller as an ingenious comedy while criticizing acting, storyline and implausible scenes. | Enjoyment of perceived poor filmmaking reverses the usual relationship between criticism and sentiment. Local word pairs may miss this perspective. |
| 20077 | Positive → negative | Opens with strong praise for a performance and script, then discusses capital punishment, religion and forgiveness at length. | Much of the vocabulary concerns a serious topic rather than film quality. Topic language may dominate aggregated features; causal attribution remains untested. |

All eight original texts and predictions are retained in the diagnostics and CSV.
Possible label disagreement is kept separate from confirmed classifier error against
the supplied targets; labels were not changed. Rows 3116 and 5577 do not establish
an overall label-noise rate. Row 17620 shows that clear praise can still be missed.

## Practical limitations and next steps

Mixed judgments recur in this sample (915, 1127, 24224, 17790), as do lengthy plot
or topic discussions (915, 1127, 20077). Unigram/bigram aggregation has no explicit
representation of author stance, attribution or final verdict. These are observed
text patterns and architectural limitations, not proven causes of the predictions.
The cleaner preserves negation, but default word tokenization splits contractions
and discards punctuation. Sarcasm and negation scope remain plausible limitations;
this sample does not establish their frequency or a measured failure rate.

1. Compare opening/closing-sentence features with full-review features using
   training-only grouped CV, motivated by 1127, 24224 and 20077. Inspect learned
   contributions before attributing failures to topic or conclusion weighting.
2. Test contraction-aware tokens and longer local context under the same training
   folds. Review cases like 17620 and 17790 where apparently negative language
   accompanies favorable judgment. Larger feature budgets have a cost and need
   not improve generalization.
3. Independently annotate a training-only sample for overall sentiment and
   ambiguity, motivated by 3116 and 5577. Preserve original evaluation labels;
   possible disagreement should be reported rather than silently corrected.
4. Measure category error rates on a training/CV sample containing correct as well
   as incorrect predictions. Annotate mixed judgment, attribution and topic-heavy
   discussion before choosing a change; eight selected mistakes lack denominators.

Use grouped folds within the existing 20,000-row training partition for future
selection. These 5,000 diagnostic rows already informed Task 4 winner selection;
repeated optimization against them would overfit the holdout. Official boundaries
are preserved. Identical cleaned reviews cannot cross internal partitions, but
near-duplicates are undetected and repeated rows retain repeated weight. The model
has no neutral or mixed class, and other languages or domains remain unvalidated.
