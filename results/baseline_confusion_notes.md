# Baseline Confusion Matrix Notes

This file summarizes aggregate confusion matrices for the current GAMEEMO baseline settings.

Rows are saved in `results/baseline_confusion_matrices.csv`; each block is normalized by true label.

## Largest Aggregate Confusions

- loso / statistical / LinearSVM: calm -> boring count=1287 normalized=0.308485
- loso / statistical / LogisticRegression: calm -> boring count=1209 normalized=0.289789
- loso / statistical / LogisticRegression: horror -> funny count=1158 normalized=0.277565
- loso / statistical / LinearSVM: funny -> boring count=1149 normalized=0.275407
- loso / log_relative_bandpower / LinearSVM: calm -> boring count=1146 normalized=0.274688
- loso / statistical / LinearSVM: horror -> boring count=1131 normalized=0.271093
- loso / statistical / LinearSVM: horror -> funny count=1110 normalized=0.266059
- loso / statistical / LogisticRegression: funny -> boring count=1103 normalized=0.264382

## Use In Meeting

- Use this to show which emotion conditions are most often confused, not just the headline accuracy.
- The LOSO blocks are especially useful because they aggregate predictions from subject-held-out folds.
