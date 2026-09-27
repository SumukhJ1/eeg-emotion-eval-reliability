# Transformer tokenization ablation

Current tokenization table for the GAMEEMO Transformer experiments. All rows use 4-second raw EEG windows, train-only channel standardization, `d_model=64`, `n_heads=4`, `n_layers=2`, learning rate `0.001`, dropout `0.1`, and 30 requested epochs.

| Tokenization | Protocol | Accuracy | Macro-F1 | Evidence basis | Main interpretation |
| --- | --- | ---: | ---: | --- | --- |
| Channel-token | Subject-dependent | 0.650362 | 0.650782 | Single seed | Baseline Transformer tokenization: each EEG channel is one token with 512 time samples. |
| Channel-token | LOSO | 0.740830 | 0.739784 | Single seed, full 28-fold LOSO | Channel-token LOSO is strong but remains below temporal-patch LOSO. |
| Temporal-patch | Subject-dependent | 0.766908 | 0.767266 | Mean across seeds `0`, `1`, `2` | Temporal patching improves subject-dependent performance over channel tokens. |
| Temporal-patch | LOSO | 0.801520 | 0.801141 | Mean across seeds `0`, `1`, `2` | Temporal patching is the strongest tested Transformer tokenization and remains stable in LOSO. |

## Interpretation

The key ablation is representation. Channel-token input treats each EEG channel as a token and asks attention to compare channels directly. Temporal-patch input instead divides the 4-second signal into time patches where each token sees all channels, which appears to better preserve short temporal EEG structure.

The strongest defensible claim is:

> Temporal-patch tokenization substantially improves GAMEEMO Transformer performance over channel-token input under the tested 4-second setup.

The channel-token LOSO result is currently one seed, while temporal-patch LOSO is repeated across three seeds. That should be stated clearly in slides or draft text.

## Source files

- `results/transformer_input_mode_comparison.csv`
- `results/loso_channel_token_transformer_4s_summary.csv`
- `results/transformer_subject_dependent_repeated_seed_summary.csv`
- `results/transformer_loso_repeated_seed_summary.csv`
- `results/transformer_tokenization_ablation.csv`
