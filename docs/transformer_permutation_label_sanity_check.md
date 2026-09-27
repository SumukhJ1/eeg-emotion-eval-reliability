# Transformer permutation-label sanity check

Matched-budget subject-dependent Transformer run with randomly permuted training labels. Validation and test labels remain the real GAMEEMO labels.

| **Setup** | **Window** | **Split** | **Train labels** | **Accuracy** | **Macro-F1** | **Chance** |
| --------- | ---------- | --------- | ---------------- | -----------: | -----------: | ---------: |
| Temporal-patch Transformer | 4s | Subject-dependent matched budget | Real labels | 0.807432 | 0.807998 | 0.250000 |
| Temporal-patch Transformer | 4s | Subject-dependent matched budget | Permuted labels | 0.243243 | 0.238273 | 0.250000 |

The permuted-label run falls to chance, which supports the pipeline: the high Transformer result is not trivially explained by an accidental label-independent artifact in the features, split code, or evaluation code.

## Configuration

- Model: TemporalPatchTransformer
- Window length: 512 samples, or 4 seconds at 128 Hz
- Split: subject_dependent_matched_budget
- Train/validation/test windows: 6393 / 1599 / 296
- Label control: permute training labels only
- Label permutation seed: 1729
- Normalization: train-only channel standardization
- Epochs: 30 requested, early stopped at epoch 16
- Best validation epoch: 8

## Source files

- `results/subject_dependent_transformer_matched_budget.csv`
- `results/transformer_permutation_label_sanity_check.csv`
