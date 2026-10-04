# Upgrade results summary

New results added after extending the Transformer checks.

| Result | Value | Source |
| --- | --- | --- |
| Temporal-patch Transformer subject-dependent, seeds `0-9` | Accuracy `0.771256 +/- 0.007836`; macro-F1 `0.771439 +/- 0.007619` | `results/transformer_subject_dependent_repeated_seed_summary.csv` |
| Temporal-patch Transformer matched-budget subject-dependent, seeds `0-9` | Accuracy `0.804392 +/- 0.014808`; macro-F1 `0.804275 +/- 0.015458` | `results/transformer_matched_budget_repeated_seed_summary.csv` |
| Channel-token Transformer matched-budget subject-dependent, seeds `0-2` | Accuracy `0.715090 +/- 0.035486`; macro-F1 `0.715064 +/- 0.035139` | `results/channel_token_matched_budget_repeated_seed_summary.csv` |
| Matched-budget split leakage audit | `10` matched-budget seeds checked; train/test window overlap `0` for every seed | `results/subject_dependent_matched_budget_split_summary.csv` |
| Normalization leakage audit | `300` rows checked; leakage failures `0`; channel standardization fit on train windows only | `results/normalization_leakage_audit.csv` |
| Channel-token global label permutation | Accuracy `0.209459`; macro-F1 `0.207156` | `results/channel_token_transformer_permutation_label_sanity_check.csv` |
| Channel-token within-subject label permutation | Accuracy `0.212838`; macro-F1 `0.208600` | `results/channel_token_transformer_subject_permutation_leakage_check.csv` |
| LOSO minus original subject-dependent gap, subject bootstrap | Point estimate `0.02966306`; 95% CI `[0.01416636, 0.04491830]`; Wilcoxon p `0.00106452` | `results/subject_level_bootstrap_summary.csv` |
| Matched-budget subject-dependent minus LOSO gap, subject bootstrap | Point estimate `0.00365965`; 95% CI `[-0.01024302, 0.01788312]`; Wilcoxon p `0.81392329` | `results/subject_level_bootstrap_summary.csv` |

The main updated reading is that the original subject-dependent score rises from the old 3-seed estimate once seeds `3-9` are included, while matched-budget subject-dependent and LOSO remain close. The bootstrap supports the same story: the LOSO-over-original-SD gap is real under the old split budget, but the matched-budget-vs-LOSO gap is small and uncertain.
