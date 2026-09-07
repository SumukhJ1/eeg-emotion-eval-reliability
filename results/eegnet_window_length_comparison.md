# EEGNet 2s vs 4s Window Comparison

This comparison checks whether the Transformer advantage could be explained only by using 4-second windows while EEGNet used 2-second windows.

Configuration for the 4-second EEGNet run: `window_samples=512`, learning rate `0.001`, dropout `0.25`, temporal filters `8`, temporal kernel `32`, depth multiplier `2`, separable filters `16`, separable kernel `16`, batch size `64`, max `30` epochs, patience `8`, train-only channel standardization.

| Model | Window | Protocol | Accuracy | Macro-F1 | Status |
| --- | --- | --- | ---: | ---: | --- |
| EEGNet | 2s | SD | 0.658873 | 0.651943 | full |
| EEGNet | 2s | LOSO | 0.522771 | 0.500855 | full |
| EEGNet | 4s | SD | 0.652174 | 0.648623 | full |
| EEGNet | 4s | LOSO | 0.564310 | 0.543326 | full |
| ChannelTokenTransformer | 4s | SD | 0.650362 | 0.650782 | single seed |
| ChannelTokenTransformer | 4s | LOSO | 0.740830 | 0.739784 | single seed |
| TemporalPatchTransformer | 4s | SD | 0.766908 | 0.767266 | repeated-seed mean |
| TemporalPatchTransformer | 4s | LOSO | 0.801520 | 0.801141 | repeated-seed mean |

Interpretation: using 4-second windows does not explain the Transformer advantage by itself. The tuned 4-second EEGNet subject-dependent result remains much lower than the 4-second temporal-patch Transformer subject-dependent repeated-seed mean, and the full 28-fold 4-second EEGNet LOSO result remains below both tested 4-second Transformer LOSO settings. Temporal patches still outperform channel tokens in the available 4-second LOSO comparison.

The full 4-second EEGNet LOSO run shows high subject-level variability: the best held-out subject was `S17` with accuracy `0.905405` and macro-F1 `0.906322`, while the worst was `S14` with accuracy `0.307432` and macro-F1 `0.224249`. This replaces the earlier 3-fold smoke-test result.

Source files:

- `results/subject_dependent_eegnet_4s_baseline.csv`
- `results/loso_eegnet_4s_baseline.csv`
- `results/loso_eegnet_4s_summary.csv`
- `results/loso_channel_token_transformer_4s.csv`
- `results/loso_channel_token_transformer_4s_summary.csv`
- `results/eegnet_window_length_comparison.csv`
