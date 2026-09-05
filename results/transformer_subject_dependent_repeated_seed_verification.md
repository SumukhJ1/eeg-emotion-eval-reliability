# Transformer Subject-Dependent Repeated-Seed Verification

This check reruns the same GAMEEMO temporal-patch Transformer subject-dependent setup across the same seeds used for LOSO verification. The goal is to test whether the observation that LOSO outperformed subject-dependent evaluation is stable or just a one-run artifact.

Configuration held fixed: 4-second windows, temporal-patch input mode, 32-sample patches, `d_model=64`, `n_heads=4`, `n_layers=2`, `dim_feedforward=128`, learning rate `0.001`, dropout `0.1`, weight decay `0.0001`, batch size `64`, max `30` epochs, patience `8`, train-only channel standardization, no class weighting.

| Seed | Subject-dependent accuracy | Subject-dependent macro-F1 |
| --- | --- | --- |
| 0 | 0.766304 | 0.766356 |
| 1 | 0.773551 | 0.773687 |
| 2 | 0.760870 | 0.761754 |

| Model | Protocol | Mean acc | Std acc | Mean F1 | Std F1 |
| --- | --- | --- | --- | --- | --- |
| TemporalPatchTransformer | Subject-dependent | 0.766908 | 0.005195 | 0.767266 | 0.004914 |
| TemporalPatchTransformer | LOSO | 0.801520 | 0.003501 | 0.801141 | 0.003476 |

Interpretation: the subject-dependent Transformer score is stable across seeds, and the LOSO advantage also remains stable across the same seed set. This suggests the LOSO > subject-dependent result is not just a one-run random initialization artifact. The likely explanation should still be investigated rather than overclaimed, because subject-dependent and LOSO use different train/test compositions.

Files:

- `results/transformer_subject_dependent_repeated_seed_results.csv`
- `results/transformer_subject_dependent_repeated_seed_summary.csv`
- `results/transformer_subject_dependent_repeated_seed_runs/`
- `results/transformer_loso_repeated_seed_results.csv`
- `results/transformer_loso_repeated_seed_summary.csv`
