# LOSO Subject Failure Analysis

This analysis summarizes per-subject leave-one-subject-out variation across the current GAMEEMO baselines.

## Hardest Held-Out Subjects

Using bandpower + logistic regression, the lowest macro-F1 subjects are:

- S21 (bandpower, LogisticRegression): accuracy=0.240, macro-F1=0.219
- S08 (bandpower, LogisticRegression): accuracy=0.287, macro-F1=0.249
- S22 (bandpower, LogisticRegression): accuracy=0.270, macro-F1=0.257
- S25 (bandpower, LogisticRegression): accuracy=0.337, macro-F1=0.271
- S09 (bandpower, LogisticRegression): accuracy=0.329, macro-F1=0.275

## Strongest Held-Out Subjects

Using bandpower + logistic regression, the highest macro-F1 subjects are:

- S03 (bandpower, LogisticRegression): accuracy=0.579, macro-F1=0.568
- S01 (bandpower, LogisticRegression): accuracy=0.545, macro-F1=0.539
- S02 (bandpower, LogisticRegression): accuracy=0.515, macro-F1=0.511
- S17 (bandpower, LogisticRegression): accuracy=0.505, macro-F1=0.496
- S15 (bandpower, LogisticRegression): accuracy=0.500, macro-F1=0.487

## Protocol Gap

| Feature Set | Model | Subject-Dependent Macro-F1 | LOSO Mean Macro-F1 | Drop | LOSO Macro-F1 Std |
| --- | --- | ---: | ---: | ---: | ---: |
| statistical | LogisticRegression | 0.374935 | 0.299302 | 0.075633 | 0.077698 |
| statistical | LinearSVM | 0.372721 | 0.298395 | 0.074326 | 0.076706 |
| bandpower | LogisticRegression | 0.490146 | 0.367046 | 0.123101 | 0.093395 |
| bandpower | LinearSVM | 0.488620 | 0.361954 | 0.126666 | 0.096874 |

## Takeaways

- LOSO performance varies substantially by held-out subject, so aggregate scores hide subject-level failure modes.
- Bandpower improves the baseline, but several held-out subjects remain difficult.
- The next analysis target is to inspect whether difficult subjects share label confusions, signal-quality issues, or preprocessing differences.
