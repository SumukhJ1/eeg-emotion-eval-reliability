# Matched-Budget Transformer Repeated-Seed Verification

This run repeats the 4-second temporal-patch Transformer under the matched-budget subject-dependent split for seeds `0`, `1`, and `2`. The split matches the mean LOSO train/validation/test window budget while keeping subject-dependent random window mixing.

| Seed | Accuracy | Macro-F1 |
| ---: | ---: | ---: |
| 0 | 0.807432 | 0.807998 |
| 1 | 0.810811 | 0.812604 |
| 2 | 0.793919 | 0.794008 |

| Model | Protocol | Split | Mean accuracy | Std accuracy | Mean macro-F1 | Std macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TemporalPatchTransformer | Subject-dependent | matched budget | 0.804054 | 0.008938 | 0.804870 | 0.009685 |

Interpretation: the matched-budget subject-dependent result is stable across the three checked seeds and remains very close to the repeated-seed LOSO Transformer result (`0.801520` accuracy, `0.801141` macro-F1). This supports the protocol-composition interpretation: the earlier LOSO-over-subject-dependent pattern was mostly a train/test budget artifact, not evidence that LOSO is inherently easier.

Source files:

- `results/transformer_matched_budget_repeated_seed_results.csv`
- `results/transformer_matched_budget_repeated_seed_summary.csv`
- `results/transformer_matched_budget_repeated_seed_runs/`
