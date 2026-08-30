# Preprocessing Audit Results

This summarizes the first conservative quality-filtering audit for GAMEEMO bandpower baselines. The filter policy was defined before using filtered results: remove missing values, nonfinite values, near-zero variance windows, and flat-channel windows; flag extreme-amplitude and extreme-variance windows without automatically removing them.

The hard filtering rules removed 0 of 16,688 windows. Therefore, the before/after bandpower results are unchanged.

| **Setup** | **SD acc** | **LOSO acc** | **SD F1** | **LOSO F1** | **Interpretation** |
| --------- | ---------- | ------------ | --------- | ----------- | ------------------ |
| Bandpower LR, before filtering | 0.491607 | 0.391898 | 0.490146 | 0.367046 | Baseline bandpower logistic regression result using all preprocessed CSV windows. |
| Bandpower LR, after conservative hard filtering | 0.491607 | 0.391898 | 0.490146 | 0.367046 | No change because the hard filter removed 0 windows. |
| Bandpower SVM, before filtering | 0.490408 | 0.388603 | 0.488620 | 0.361954 | Baseline bandpower linear SVM result using all preprocessed CSV windows. |
| Bandpower SVM, after conservative hard filtering | 0.490408 | 0.388603 | 0.488620 | 0.361954 | No change because the hard filter removed 0 windows. |

Filtering did not materially improve accuracy, suggesting the bottleneck is not simple missing/flat/extreme-window artifacts.

Important caveat: extreme-amplitude and extreme-variance windows were flagged during the audit but not removed in this filtered rerun. The current conservative filter only removes invalid or degenerate windows. A separate sensitivity analysis would be needed before excluding flagged extreme windows.

Output files:

- `results/gameemo_signal_quality_audit_summary.csv`
- `results/gameemo_signal_quality_audit_by_subject.csv`
- `results/gameemo_signal_quality_flagged_windows.csv`
- `results/bandpower_quality_filter_summary.csv`
- `results/subject_dependent_bandpower_quality_filtered_baselines.csv`
- `results/loso_bandpower_quality_filtered_baselines.csv`
- `results/loso_bandpower_quality_filtered_summary.csv`
