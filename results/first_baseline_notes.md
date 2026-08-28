# First Baseline Notes

These are rough first-pass results for GAMEEMO using the local preprocessed CSV files. The pipeline uses 2-second non-overlapping windows at 128 Hz and simple per-channel statistical features: mean, standard deviation, min, max, and mean squared value.

GAMEEMO has four target classes, so the uniform random chance baseline is 25% accuracy. These first-pass results should be interpreted relative to chance and relative to the project's own controlled subject-dependent versus LOSO protocols, not directly against published papers yet.

## Results

| Protocol | Model | Windows | Train/Test Windows | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Random chance baseline | Uniform 4-class guess | 16,688 | N/A | 0.250000 | 0.250000 |
| Subject-dependent random split | Logistic regression | 16,688 | 13,352 / 3,336 | 0.376199 | 0.374935 |
| Subject-dependent random split | Linear SVM | 16,688 | 13,352 / 3,336 | 0.374700 | 0.372721 |
| LOSO mean across 28 folds | Logistic regression | 16,688 | 16,092 / 596 per fold | 0.321309 | 0.299302 |
| LOSO mean across 28 folds | Linear SVM | 16,688 | 16,092 / 596 per fold | 0.320709 | 0.298395 |

Output tables:

- `results/subject_dependent_baseline.csv`
- `results/subject_dependent_svm_baseline.csv`
- `results/loso_baseline.csv`
- `results/loso_summary.csv`
- `results/loso_svm_baseline.csv`
- `results/loso_svm_summary.csv`

## Caveats

- This is a quick baseline, not a final EEG feature pipeline.
- GAMEEMO is a 4-class problem, so 25% random chance is the relevant first reference point.
- Both logistic regression and linear SVM use the same simple statistical window features.
- These statistical features are not final EEG bandpower features.
- Published GAMEEMO results may use different preprocessing, windowing, feature extraction, and split protocols, so direct comparisons are premature.
- The split is subject-dependent, so windows from the same subjects can appear in both train and test.
- LOSO results are more relevant for unseen-subject generalization, but preprocessing details still need verification.
- Stronger EEG-specific features are the next step before drawing conclusions about model quality.
