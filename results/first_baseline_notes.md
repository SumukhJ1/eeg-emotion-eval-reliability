# First Subject-Dependent Baseline Notes

These are rough first-pass results for GAMEEMO using the local preprocessed CSV files. The pipeline uses 2-second non-overlapping windows at 128 Hz, simple per-channel statistical features, and a logistic regression classifier.

## Result

| Protocol | Windows | Train Windows | Test Windows | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Subject-dependent random split | 16,688 | 13,352 | 3,336 | 0.376199 | 0.374935 |

The output table is saved at `results/subject_dependent_baseline.csv`.

## Caveats

- This is a quick baseline, not a final EEG feature pipeline.
- Features are simple statistics only: mean, standard deviation, min, max, and mean squared value per channel.
- The split is subject-dependent, so windows from the same subjects can appear in both train and test.
- These results should be compared against LOSO subject-independent results before drawing generalization conclusions.
