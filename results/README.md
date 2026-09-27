# Results

This folder contains frozen result artifacts for the GAMEEMO experiments. Result CSV files are not edited during documentation cleanup; README files only describe what is already present.

## Paper-facing files

| File | Description |
| --- | --- |
| `final_gameemo_result_table.csv` | Compact paper-ready table for the main model ladder. It lists model/input, window length, protocol, accuracy, macro-F1, chance, notes, and source file. |
| `transformer_subject_dependent_repeated_seed_summary.csv` | Three-seed summary for the original 4-second temporal-patch Transformer subject-dependent split. Main columns are mean accuracy, accuracy standard deviation, mean macro-F1, macro-F1 standard deviation, and fixed model config. |
| `transformer_subject_dependent_repeated_seed_results.csv` | Per-seed rows behind the original subject-dependent summary. Each row gives seed, protocol, accuracy, macro-F1, and config. |
| `transformer_matched_budget_repeated_seed_summary.csv` | Three-seed summary for the matched-budget subject-dependent Transformer control. Main control for train/validation/test budget effects. |
| `transformer_matched_budget_repeated_seed_results.csv` | Per-seed rows behind the matched-budget subject-dependent summary. Each row gives seed, protocol, accuracy, macro-F1, split protocol, and config. |
| `transformer_loso_repeated_seed_summary.csv` | Three-seed summary for temporal-patch Transformer LOSO. Main columns are mean accuracy, seed standard deviation, mean macro-F1, and fixed model config. |
| `transformer_loso_repeated_seed_results.csv` | Per-seed LOSO summary rows. Each row gives one seed's mean LOSO accuracy, macro-F1, fold standard deviations, and source summary file. |
| `transformer_loso_subject_analysis.csv` | Seed-0 held-out-subject LOSO fold table. Seed 1 and seed 2 full fold tables are under `transformer_loso_repeated_seed_runs/`. |
| `transformer_permutation_label_sanity_check.csv` | Matched-budget Transformer run with globally permuted training labels. Accuracy should be near the 25% four-class chance level. |
| `transformer_subject_permutation_leakage_check.csv` | Matched-budget Transformer run with labels permuted within subject groups. Accuracy should also be near chance. |
| `loso_leakage_audit.csv` | LOSO split audit. Expected values are `held_out_in_train=False`, `held_out_in_val=False`, and `held_out_in_test_only=True` for each fold. |
| `normalization_leakage_audit.csv` | Normalization audit. Expected values are `normalization_source=train only` and `test_data_used_in_fit=False`. |

## Transformer supporting files

| File or folder | Description |
| --- | --- |
| `transformer_subject_dependent_repeated_seed_verification.md` | Short verification note for the original subject-dependent repeated-seed run. It states the fixed config, three seeds, per-seed scores, and source files. |
| `transformer_matched_budget_repeated_seed_verification.md` | Short verification note for the matched-budget repeated-seed run. It states the fixed config, three seeds, per-seed scores, summary scores, and source files. |
| `transformer_loso_repeated_seed_verification.md` | Short verification note for the LOSO repeated-seed run. It states the fixed config, three seeds, per-seed scores, summary scores, and source files. |
| `transformer_loso_repeated_seed_runs/` | Fold-level and seed-level LOSO files for repeated-seed Transformer checks. Full seed-0 fold rows are stored at top level in `transformer_loso_subject_analysis.csv`; seed 1 and seed 2 fold rows are stored in this folder. |
| `transformer_subject_dependent_repeated_seed_runs/` | Per-seed output files for the original subject-dependent repeated-seed Transformer runs. |
| `transformer_matched_budget_repeated_seed_runs/` | Per-seed output files for matched-budget subject-dependent Transformer runs. |
| `transformer_split_composition/` | Split-composition audit files used to explain train/test budget differences between subject-dependent and LOSO evaluation. |
| `transformer_protocol_composition_effects.csv` | Compact table comparing original subject-dependent, matched-budget subject-dependent, and LOSO Transformer protocols. |
| `transformer_tokenization_ablation.csv` | Transformer tokenization comparison. It includes channel-token and temporal-patch variants. |
| `transformer_input_mode_comparison.csv` | Subject-dependent Transformer input-mode comparison used before repeated-seed verification. |
| `loso_channel_token_transformer_4s.csv` and `loso_channel_token_transformer_4s_summary.csv` | Single-seed 4-second channel-token Transformer LOSO comparison. |
| `transformer_loso_confusion_matrix.csv`, `transformer_loso_class_f1.csv`, and `transformer_loso_predictions.csv` | LOSO Transformer prediction analysis outputs. These support confusion-matrix, class-level F1, and prediction-level follow-up checks. |
| `transformer_lr_dropout_grid.csv`, `transformer_window_length_comparison.csv`, `transformer_normalized_baseline.csv`, and `transformer_class_weighted_baseline.csv` | Earlier Transformer tuning and ablation outputs. |
| `transformer_lr_dropout_grid_runs/`, `transformer_window_length_runs/`, and `transformer_input_mode_runs/` | Per-run files behind earlier Transformer tuning and input-mode comparisons. |
| `transformer_forward_shape_check.txt` | Shape check output for the Transformer forward pass. |

## Baseline and EEGNet files

| File or group | Description |
| --- | --- |
| `subject_dependent_baseline.csv`, `loso_baseline.csv`, and `loso_summary.csv` | First logistic-regression statistical-feature baselines. |
| `subject_dependent_svm_baseline.csv`, `loso_svm_baseline.csv`, and `loso_svm_summary.csv` | Linear SVM statistical-feature baselines. |
| `subject_dependent_log_relative_bandpower_baselines.csv`, `loso_log_relative_bandpower_baselines.csv`, and `loso_log_relative_bandpower_summary.csv` | Log-relative bandpower logistic-regression and SVM baselines. |
| `subject_dependent_bandpower_baselines.csv`, `loso_bandpower_baselines.csv`, and `loso_bandpower_summary.csv` | Earlier absolute-bandpower baselines. |
| `subject_dependent_class_balanced_baselines.csv`, `loso_class_balanced_baselines.csv`, and `loso_class_balanced_summary.csv` | Class-balanced classical baseline variants. |
| `subject_dependent_eegnet_baseline.csv`, `subject_dependent_eegnet_4s_baseline.csv`, `loso_eegnet_tuned_baseline.csv`, `loso_eegnet_tuned_summary.csv`, `loso_eegnet_4s_baseline.csv`, and `loso_eegnet_4s_summary.csv` | EEGNet subject-dependent and LOSO runs for 2-second and 4-second windows. |
| `eegnet_lr_dropout_grid.csv`, `eegnet_kernel_filter_grid.csv`, and `eegnet_window_length_comparison.csv` | EEGNet tuning and window-length comparison outputs. |
| `eegnet_lr_dropout_grid_runs/` and `eegnet_kernel_filter_grid_runs/` | Per-run files behind EEGNet tuning grids. |
| `cnn_transformer_subject_dependent.csv` | First CNN-Transformer subject-dependent checkpoint. |

## Audit and notes files

| File or group | Description |
| --- | --- |
| `gameemo_signal_quality_audit_summary.csv`, `gameemo_signal_quality_audit_by_subject.csv`, and `gameemo_signal_quality_flagged_windows.csv` | Signal-quality audit outputs for missing values, flat channels, near-zero variance, extreme amplitude, and extreme variance flags. |
| `bandpower_quality_filter_summary.csv`, `subject_dependent_bandpower_quality_filtered_baselines.csv`, `loso_bandpower_quality_filtered_baselines.csv`, and `loso_bandpower_quality_filtered_summary.csv` | Bandpower baseline reruns after conservative quality-filter rules. |
| `baseline_confusion_matrices.csv` and `baseline_confusion_notes.md` | Aggregate confusion-matrix notes for current baseline settings. |
| `loso_subject_analysis.csv` and `loso_failure_analysis.md` | Per-subject LOSO summaries and interpretation notes for earlier baselines. |
| `protocol_gap_summary.csv` | Summary of subject-dependent versus LOSO gaps for earlier model stages. |
| `first_baseline_notes.md`, `preprocessing_audit_results.md`, `loso_eegnet_smoke_notes.md`, `normalization_leakage_audit.md`, and `eegnet_window_length_comparison.md` | Short notes describing selected result groups. |

## Reading the columns

Accuracy and macro-F1 are the main metrics. Macro-F1 is included because the task has four labels and per-class behavior matters. Chance is `0.25` for uniform random guessing on the four GAMEEMO game-condition labels.
