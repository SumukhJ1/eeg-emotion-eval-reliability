# Paper claim provenance

Paper-facing claims mapped to the repository files that contain the supporting values. Verification date: 2026-09-27.

| Claim | Value | File | Verified |
| --- | --- | --- | --- |
| Original temporal-patch Transformer subject-dependent accuracy over seeds `0,1,2`. | Mean accuracy `0.766908`; seed s.d. `0.005195`; mean macro-F1 `0.767266`; macro-F1 seed s.d. `0.004914`. This rounds to `76.7 +/- 0.5` accuracy points. | `results/transformer_subject_dependent_repeated_seed_summary.csv`; `results/transformer_subject_dependent_repeated_seed_results.csv` | Yes, 2026-09-27. |
| Matched-budget temporal-patch Transformer subject-dependent accuracy over seeds `0,1,2`. | Mean accuracy `0.804054`; seed s.d. `0.008938`; mean macro-F1 `0.804870`; macro-F1 seed s.d. `0.009685`. This rounds to `80.4 +/- 0.9` accuracy points. | `results/transformer_matched_budget_repeated_seed_summary.csv`; `results/transformer_matched_budget_repeated_seed_results.csv`; `results/transformer_matched_budget_repeated_seed_verification.md` | Yes, 2026-09-27. |
| Temporal-patch Transformer LOSO accuracy over seeds `0,1,2`. | Mean accuracy `0.801520`; seed s.d. `0.003501`; mean macro-F1 `0.801141`; macro-F1 seed s.d. `0.003476`. This rounds to `80.2` accuracy with seed s.d. `0.4` points. | `results/transformer_loso_repeated_seed_summary.csv`; `results/transformer_loso_repeated_seed_results.csv`; `results/transformer_loso_repeated_seed_verification.md` | Yes, 2026-09-27. |
| LOSO per-seed accuracies. | Seed `0`: `0.798866`; seed `1`: `0.806467`; seed `2`: `0.799228`. | `results/transformer_loso_repeated_seed_results.csv` | Yes, 2026-09-27. |
| LOSO held-out-fold spread across all repeated-seed folds. | `84` folds across seeds `0,1,2`; min accuracy `0.702703`; max accuracy `0.885135`; population s.d. `0.043352`. This rounds to range `70.3-88.5` and fold s.d. `4.3` points. | `results/transformer_loso_subject_analysis.csv`; `results/transformer_loso_repeated_seed_runs/loso_seed_1.csv`; `results/transformer_loso_repeated_seed_runs/loso_seed_2.csv` | Yes, 2026-09-27. |
| Matched-budget control explains the earlier LOSO-over-subject-dependent pattern. | Original subject-dependent accuracy `0.766908`; matched-budget subject-dependent accuracy `0.804054`; LOSO accuracy `0.801520`; matched-budget SD minus LOSO is `0.002534`, or `0.2534` percentage points. | `results/transformer_subject_dependent_repeated_seed_summary.csv`; `results/transformer_matched_budget_repeated_seed_summary.csv`; `results/transformer_loso_repeated_seed_summary.csv`; `docs/transformer_protocol_composition_effects.md` | Yes, 2026-09-27. |
| Global training-label permutation collapses near chance. | Test accuracy `0.243243`; macro-F1 `0.238273`. | `results/transformer_permutation_label_sanity_check.csv` | Yes, 2026-09-27. |
| Within-subject training-label permutation collapses near chance. | Test accuracy `0.260135`; macro-F1 `0.259393`. | `results/transformer_subject_permutation_leakage_check.csv` | Yes, 2026-09-27. |
| LOSO split audit found no held-out subject leakage. | Each checked fold has `held_out_in_train=False`, `held_out_in_val=False`, and `held_out_in_test_only=True`. | `results/loso_leakage_audit.csv` | Yes, 2026-09-27. |
| Neural normalization audit found no test-data normalization leakage. | `normalization_source=train only`; `test_data_used_in_fit=False`; `held_out_subject_used_in_fit=False` for LOSO rows. | `results/normalization_leakage_audit.csv`; `results/normalization_leakage_audit.md` | Yes, 2026-09-27. |
| Final model ladder uses frozen result files. | Time-stat LR/SVM, log-relative bandpower LR/SVM, EEGNet, channel-token Transformer, temporal-patch Transformer, and CNN-Transformer rows are listed with source files. | `results/final_gameemo_result_table.csv`; `docs/final_model_ablation_table.md` | Yes, 2026-09-27. |

## Boundary

These files support a GAMEEMO protocol-sensitivity and representation-sensitivity claim. They do not support a state-of-the-art claim or a broad EEG emotion-recognition generalization claim across datasets.
