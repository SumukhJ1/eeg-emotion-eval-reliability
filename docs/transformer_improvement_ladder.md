# GAMEEMO Transformer improvement ladder

These results use GAMEEMO preprocessed EEG CSV files, 4-class game-condition labels, train-only channel standardization for neural models, and fixed random seed `0` unless noted. Chance is `25%` for the 4-class task.

| Stage | Model / Input | Protocol | Window | Accuracy | Macro-F1 | Result file | Interpretation |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| First raw Transformer checkpoint | Channel-token Transformer | Subject-dependent | 2s / 256 samples | 0.592326 | 0.591329 | `results/subject_dependent_transformer_baseline.csv` | Plain channel tokens were better than classical feature baselines but below tuned EEGNet. |
| Explicit train-normalized Transformer | Channel-token Transformer | Subject-dependent | 2s / 256 samples | 0.596523 | 0.596109 | `results/transformer_normalized_baseline.csv` | Train-only normalization is now explicit and recorded; this also uses the better `lr=0.001` setting. |
| Longer-window Transformer | Channel-token Transformer | Subject-dependent | 4s / 512 samples | 0.650362 | 0.650782 | `results/transformer_window_length_comparison.csv` | Longer windows improved the channel-token Transformer substantially. |
| Temporal-patch Transformer | Temporal-patch Transformer, 32-sample patches | Subject-dependent | 4s / 512 samples | 0.766304 | 0.766356 | `results/transformer_input_mode_comparison.csv` | Temporal patching produced the largest Transformer gain, suggesting input representation mattered more than simply using attention. |
| Class-weighted temporal-patch Transformer | Temporal-patch Transformer, balanced loss | Subject-dependent | 4s / 512 samples | 0.764493 | 0.764544 | `results/transformer_class_weighted_baseline.csv` | Class weighting did not materially help because the training split was already nearly balanced. |
| CNN-Transformer hybrid | EEGNet-style CNN front end + Transformer | Subject-dependent | 4s / 512 samples | 0.764493 | 0.764492 | `results/cnn_transformer_subject_dependent.csv` | The hybrid was strong but did not beat temporal patching alone. |
| Tuned EEGNet reference | EEGNet | Subject-dependent | 2s / 256 samples | 0.658873 | 0.651943 | `results/eegnet_kernel_filter_grid.csv` | EEGNet remains a compact EEG-specific reference model, but the temporal-patch Transformer is now higher under the subject-dependent split. |
| Tuned EEGNet reference | EEGNet | LOSO | 2s / 256 samples | 0.522771 | 0.500855 | `results/loso_eegnet_tuned_summary.csv` | EEGNet improved over simple baselines but had a large subject-generalization drop. |
| Best Transformer LOSO checkpoint | Temporal-patch Transformer, 32-sample patches | LOSO | 4s / 512 samples | 0.798866 | 0.798241 | `results/loso_transformer_summary.csv` | The temporal-patch Transformer gain largely survived unseen-subject evaluation in this checkpoint. |

## Current Takeaway

The highest-impact changes were not generic Transformer tuning. The meaningful gains came from EEG-window representation choices:

1. Moving from 2-second to 4-second windows improved the channel-token Transformer from `0.596109` to `0.650782` macro-F1.
2. Replacing channel tokens with temporal patches improved subject-dependent macro-F1 to `0.766356`.
3. The best temporal-patch configuration reached LOSO mean macro-F1 `0.798241` across 28 held-out subjects, with worst held-out subject S26 at `0.725831` macro-F1 and best held-out subject S03 at `0.884837`.

The next checks should focus on whether this strong LOSO result is robust to repeated seeds and whether the same temporal-patch design transfers to a second dataset such as DREAMER.
