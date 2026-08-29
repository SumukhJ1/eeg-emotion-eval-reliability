# First Baseline Notes

These are rough first-pass results for GAMEEMO using the local preprocessed CSV files. The pipeline uses 2-second non-overlapping windows at 128 Hz and simple per-channel statistical features: mean, standard deviation, min, max, and mean squared value.

GAMEEMO has four target classes, so the uniform random chance baseline is 25% accuracy. These first-pass results should be interpreted relative to chance and relative to the project's own controlled subject-dependent versus LOSO protocols, not directly against published papers yet.

For literature context, see `docs/gameemo_prior_methods.md`. Prior GAMEEMO papers report a wide range of accuracies, but many use binary labels, different feature pipelines, or split protocols that are not confirmed to match these experiments.

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

## EEGNet Protocol Gap

| Model | Input | Protocol | Accuracy | Macro-F1 | Chance | Accuracy drop vs subject-dependent | Macro-F1 drop vs subject-dependent |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | Raw EEG windows | Subject-dependent | 0.552458 | 0.543291 | 0.250000 | 0.234625 | 0.278458 |
| EEGNet | Raw EEG windows | LOSO | 0.317833 | 0.264833 | 0.250000 | N/A | N/A |

EEGNet improves subject-dependent performance, but LOSO determines whether the gain transfers to unseen subjects. The current full LOSO run used only 3 CPU epochs per fold, so it should be treated as a first complete cross-subject baseline rather than a tuned EEGNet result.
