# Consolidated GAMEEMO Result Table

This table is the current paper-ready summary for the GAMEEMO experiments. All results use the four GAMEEMO game-condition labels, so the uniform random chance baseline is `0.250000`. Published GAMEEMO numbers should still be treated as literature context, not direct comparisons, unless preprocessing, labels, and split protocol match.

| **Input/model** | **Window** | **Protocol** | **Accuracy** | **Macro-F1** | **Chance** | **Notes** |
| --------------- | ---------- | ------------ | ------------ | ------------ | ---------- | --------- |
| Time-stat LR | 2s | Subject-dependent | 0.376199 | 0.374935 | 0.250000 | Mean/std/min/max/energy per channel; seed 0 random split. |
| Time-stat LR | 2s | LOSO | 0.321309 | 0.299302 | 0.250000 | Mean across 28 held-out-subject folds. |
| Time-stat SVM | 2s | Subject-dependent | 0.374700 | 0.372721 | 0.250000 | StandardScaler + LinearSVC on statistical features. |
| Time-stat SVM | 2s | LOSO | 0.320709 | 0.298395 | 0.250000 | Mean across 28 held-out-subject folds. |
| Log-relative bandpower LR | 2s | Subject-dependent | 0.468225 | 0.467152 | 0.250000 | First EEG-specific spectral feature baseline. |
| Log-relative bandpower LR | 2s | LOSO | 0.387824 | 0.370181 | 0.250000 | Mean across 28 held-out-subject folds. |
| Log-relative bandpower SVM | 2s | Subject-dependent | 0.469724 | 0.466975 | 0.250000 | Best classical subject-dependent baseline so far. |
| Log-relative bandpower SVM | 2s | LOSO | 0.388063 | 0.367017 | 0.250000 | Mean across 28 held-out-subject folds. |
| Tuned EEGNet | 2s | Subject-dependent | 0.658873 | 0.651943 | 0.250000 | LR 0.001, dropout 0.25, temporal kernel 32, 8 filters. |
| Tuned EEGNet | 2s | LOSO | 0.522771 | 0.500855 | 0.250000 | Full 28-fold LOSO; clear protocol drop remains. |
| Tuned EEGNet | 4s | Subject-dependent | 0.652174 | 0.648623 | 0.250000 | LR 0.001, dropout 0.25, temporal kernel 32, 8 filters. |
| Tuned EEGNet | 4s | LOSO | 0.564310 | 0.543326 | 0.250000 | Full 28-fold LOSO; fairer 4s neural comparison. |
| Temporal-patch Transformer | 4s | Subject-dependent | 0.766908 | 0.767266 | 0.250000 | Mean across seeds 0, 1, 2; train-only channel standardization. |
| Temporal-patch Transformer | 4s | LOSO | 0.801520 | 0.801141 | 0.250000 | Mean across seeds 0, 1, 2; leakage audits completed. |
| CNN-Transformer | 4s | Subject-dependent | 0.764493 | 0.764492 | 0.250000 | Secondary first checkpoint; LOSO not run yet. |

## Source Files

| **Rows** | **Source file** |
| -------- | --------------- |
| Time-stat LR subject-dependent | `results/subject_dependent_baseline.csv` |
| Time-stat LR LOSO | `results/loso_summary.csv` |
| Time-stat SVM subject-dependent | `results/subject_dependent_svm_baseline.csv` |
| Time-stat SVM LOSO | `results/loso_svm_summary.csv` |
| Log-relative bandpower LR/SVM subject-dependent | `results/subject_dependent_log_relative_bandpower_baselines.csv` |
| Log-relative bandpower LR/SVM LOSO | `results/loso_log_relative_bandpower_summary.csv` |
| Tuned EEGNet subject-dependent | `results/eegnet_kernel_filter_grid.csv` |
| Tuned EEGNet LOSO | `results/loso_eegnet_tuned_summary.csv` |
| Tuned EEGNet 4s subject-dependent | `results/subject_dependent_eegnet_4s_baseline.csv` |
| Tuned EEGNet 4s LOSO | `results/loso_eegnet_4s_summary.csv` |
| Temporal-patch Transformer subject-dependent | `results/transformer_subject_dependent_repeated_seed_summary.csv` |
| Temporal-patch Transformer LOSO | `results/transformer_loso_repeated_seed_summary.csv` |
| CNN-Transformer subject-dependent | `results/cnn_transformer_subject_dependent.csv` |

## Interpretation

The result ladder shows three main points. First, simple statistical features are above chance but weak, so they are useful only as a sanity-check baseline. Second, EEG-specific bandpower features improve classical LR/SVM performance, but still leave a large gap to neural raw-window models. Third, the temporal-patch Transformer is currently the strongest GAMEEMO model in both subject-dependent and LOSO evaluation, but the unexpectedly strong LOSO result should continue to be treated carefully and defended with the existing leakage, normalization, confusion-matrix, and per-subject analyses.
