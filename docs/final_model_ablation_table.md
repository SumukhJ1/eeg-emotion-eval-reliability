# Final GAMEEMO Model Ablation Table

This table summarizes the current controlled GAMEEMO model ladder. All rows use the same four GAMEEMO game-condition labels, so random chance is `0.250000`. The `Macro-F1` column reports `subject-dependent / LOSO` when both protocols are available.

| Input / representation | Model | Window length | Subject-dependent accuracy | LOSO accuracy | Macro-F1 | Main interpretation |
| --- | --- | --- | ---: | ---: | --- | --- |
| Time-statistical features | Logistic regression | 2s | 0.376199 | 0.321309 | 0.374935 / 0.299302 | Sanity-check baseline is above 25% chance but weak; simple time statistics are not enough for strong EEG emotion recognition. |
| Log-relative bandpower | Linear SVM | 2s | 0.469724 | 0.388063 | 0.466975 / 0.367017 | EEG-specific spectral representation improves the classical baseline but still has a large cross-subject gap. |
| Raw EEG | EEGNet | 2s | 0.658873 | 0.522771 | 0.651943 / 0.500855 | Compact EEG-specific neural model improves substantially over classical features, but unseen-subject transfer remains limited. |
| Raw EEG | EEGNet | 4s | 0.652174 | 0.564310 | 0.648623 / 0.543326 | Longer windows modestly improve EEGNet LOSO versus 2s while leaving subject-dependent accuracy similar. |
| Raw EEG channel tokens | Transformer | 4s | 0.650362 | not run | 0.650782 / not run | Channel-token Transformer is comparable to EEGNet subject-dependent, but this representation has no matched LOSO run yet. |
| Raw EEG temporal patches | Transformer | 4s | 0.766908 | 0.801520 | 0.767266 / 0.801141 | Temporal patching gives the strongest current GAMEEMO result and remains stable under repeated-seed LOSO with leakage checks. |

## Paper Story

The ablation supports a clean progression: simple time-domain features are only a sanity-check baseline, EEG-specific bandpower features help classical models, EEGNet improves raw-window learning, and the largest gain comes from changing the Transformer input representation from channel tokens to temporal patches. The current strongest result is the 4-second temporal-patch Transformer, but it should still be framed as a GAMEEMO finding until the same pattern is tested on a second dataset.

## Source Files

| Result | Source files |
| --- | --- |
| Time-statistical LR | `results/subject_dependent_baseline.csv`; `results/loso_summary.csv` |
| Log-relative bandpower SVM | `results/subject_dependent_log_relative_bandpower_baselines.csv`; `results/loso_log_relative_bandpower_summary.csv` |
| EEGNet 2s | `results/eegnet_kernel_filter_grid.csv`; `results/loso_eegnet_tuned_summary.csv` |
| EEGNet 4s | `results/subject_dependent_eegnet_4s_baseline.csv`; `results/loso_eegnet_4s_summary.csv` |
| Channel-token Transformer 4s | `results/transformer_window_length_comparison.csv`; `results/transformer_input_mode_comparison.csv` |
| Temporal-patch Transformer 4s | `results/transformer_subject_dependent_repeated_seed_summary.csv`; `results/transformer_loso_repeated_seed_summary.csv` |
