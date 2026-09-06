# Transformer Subject-Permutation Leakage Check

This check tests whether the temporal-patch Transformer can still perform well when label structure is disrupted while preserving subject/file organization. The model uses the same 4-second matched-budget subject-dependent setup as the main Transformer control, but training labels are shuffled within each subject. Validation and test labels remain real.

| **Setup** | **Training labels** | **Accuracy** | **Macro-F1** | **Chance** |
| --------- | ------------------- | -----------: | -----------: | ---------: |
| Matched-budget Transformer | Real labels | 0.807432 | 0.807998 | 0.250000 |
| Global permutation control | Permuted across all training windows | 0.243243 | 0.238273 | 0.250000 |
| Subject-preserving permutation control | Permuted within each training subject | 0.260135 | 0.259393 | 0.250000 |

The within-subject permutation keeps subject membership, file/session organization, and approximate per-subject label balance intact while breaking the mapping between EEG windows and their labels. Performance collapses to chance, which supports the interpretation that the high Transformer score is not explained by subject/file ordering alone.

## Configuration

- Model: TemporalPatchTransformer
- Window length: 512 samples, or 4 seconds at 128 Hz
- Split: subject_dependent_matched_budget
- Train/validation/test windows: 6393 / 1599 / 296
- Label control: permute training labels within each subject
- Label permutation seed: 1729
- Normalization: train-only channel standardization
- Epochs: 30
- Best validation epoch: 26

## Source Files

- `results/subject_dependent_transformer_matched_budget.csv`
- `results/transformer_permutation_label_sanity_check.csv`
- `results/transformer_subject_permutation_leakage_check.csv`
