# Manual preprocessing inspection

Inspected the four seed-42 training before/after pairs in `preprocessing.md`,
including both sentiment classes, and the synthetic unusable-input examples.

- **Negative, row 1766:** lowercasing preserves `not in my opinion`, `worst`,
  question marks, and exclamation marks. The complaint remains readable and
  keeps its negative sentiment cues.
- **Negative, row 11919:** repeated `<br />` tags become single spaces between
  sentences. `Horrible!!` becomes `horrible!!`; punctuation and word boundaries
  survive, and the label remains 0.
- **Positive, row 14266:** `CLASSICS!!!` becomes `classics!!!`; quoted film names,
  hyphens, and positive words remain. Capitalization emphasis is lost, as intended.
- **Positive, row 24419:** `not enough` and `why not` survive. Numbers such as
  `.45` and the contraction `Cobern's` remain. This review mixes criticism with
  approval; cleaning preserves that ambiguity rather than resolving sentiment.
- **Synthetic unlabeled input:** fullwidth `ＧＲＥＡＴ` becomes `great`, the break
  becomes a space, `&amp;` becomes `&`, and `don’t` becomes `don't`. The result is
  `great movie! & i don't dislike it.`; negation is preserved.
- **Unusable inputs:** `None`, an integer, whitespace, markup plus a nonbreaking
  space, punctuation alone, and a script-only string all produce the empty-string
  sentinel. Dataset preparation removes such rows together with their labels;
  inference callers must reject that sentinel before prediction.

The full run retains 25,000 rows in each official split, with 12,500 rows per class
in each. No downloaded review becomes unusable. Text changes occur in 24,935
training rows and 24,921 test rows. Filtering and invalid-label behavior are
therefore checked with synthetic frames in the tests, including missing labels
that cause pandas to promote the column to floats. Retained labels are int64,
and saved Parquet files preserve their association with texts and source indexes.

These checks support the chosen normalization rules, not a claim that every review
has been manually audited. HTML interpretation is best effort, case information
is lost, and the original duplicate/conflicting-label issues remain. No test-set
examples were used to tune a learned transformation or model.
