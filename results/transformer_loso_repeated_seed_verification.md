# Transformer LOSO Repeated-Seed Verification

This check reruns the same GAMEEMO temporal-patch Transformer LOSO setup across three seeds to test whether the prior approximately 79.9% LOSO result was stable or a lucky initialization.

Configuration held fixed: 4-second windows, temporal-patch input mode, 32-sample patches, `d_model=64`, `n_heads=4`, `n_layers=2`, `dim_feedforward=128`, learning rate `0.001`, dropout `0.1`, weight decay `0.0001`, batch size `64`, max `30` epochs, patience `8`, train-only channel standardization, no class weighting.

| Seed | LOSO accuracy | LOSO macro-F1 |
| --- | --- | --- |
| 0 | 0.798866 | 0.798241 |
| 1 | 0.806467 | 0.806028 |
| 2 | 0.799228 | 0.799153 |

| Model | Mean acc | Std acc | Mean F1 | Std F1 |
| --- | --- | --- | --- | --- |
| TemporalPatchTransformer | 0.801520 | 0.003501 | 0.801141 | 0.003476 |

Interpretation: the LOSO Transformer result is stable across these three seeds, with very small between-seed standard deviation. This makes the approximately 80% LOSO result more credible than a single-run checkpoint, while still requiring careful comparison against published work because protocol and preprocessing details may differ.

Files:

- `results/transformer_loso_repeated_seed_results.csv`
- `results/transformer_loso_repeated_seed_summary.csv`
- `results/transformer_loso_repeated_seed_runs/`
