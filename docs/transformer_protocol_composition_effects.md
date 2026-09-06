# Transformer Protocol-Composition Effects

This table isolates the main composition issue behind the earlier surprising result where LOSO appeared stronger than subject-dependent evaluation. All rows use the same temporal-patch Transformer configuration on 4-second GAMEEMO windows with train-only channel standardization.

| **Protocol** | **Train windows** | **Validation windows** | **Test design** | **Accuracy** | **Macro-F1** |
| ------------ | ----------------: | ---------------------: | --------------- | -----------: | -----------: |
| Original subject-dependent | 5305 | 1327 | Random windows across all subjects; 1656 test windows | 0.766908 | 0.767266 |
| Matched-budget subject-dependent | 6393 | 1599 | Random windows across all subjects; 296 test windows | 0.807432 | 0.807998 |
| LOSO | 6393 | 1599 | One held-out subject per fold; 296 test windows per fold | 0.801520 | 0.801141 |

## Interpretation

The original subject-dependent run used fewer training windows and a larger mixed-subject test set than the LOSO folds. After matching the subject-dependent train/validation/test budget to the mean LOSO budget, subject-dependent performance rose from `0.766908` accuracy / `0.767266` macro-F1 to `0.807432` accuracy / `0.807998` macro-F1.

This means the apparent LOSO advantage should not be interpreted as stronger unseen-subject generalization by itself. A more careful interpretation is:

> The apparent LOSO advantage is not interpreted as stronger generalization until training-budget and split-composition effects are controlled.

With the matched-budget control, subject-dependent and LOSO performance are very close. This suggests that the earlier LOSO > subject-dependent pattern was likely driven largely by train/test budget and split-composition differences, not by the model being inherently better on unseen subjects.

## Paper-Relevant Takeaway

This is a useful reliability finding: protocol comparisons can change conclusions even when the model, dataset, labels, window length, and preprocessing are held fixed. Reporting only the original subject-dependent and LOSO scores would have made the Transformer result look suspicious. Adding the matched-budget control turns that anomaly into evidence that the evaluation protocol itself must be audited.

## Source Files

- `results/transformer_subject_dependent_repeated_seed_summary.csv`
- `results/subject_dependent_transformer_matched_budget.csv`
- `results/transformer_loso_repeated_seed_summary.csv`
- `results/subject_dependent_matched_budget_split_summary.csv`
- `results/transformer_protocol_composition_effects.csv`
