# EEGNet 2s vs 4s Window Comparison

This comparison checks whether the Transformer advantage could be explained only by using 4-second windows while EEGNet used 2-second windows.

Configuration for the 4-second EEGNet run: `window_samples=512`, learning rate `0.001`, dropout `0.25`, temporal filters `8`, temporal kernel `32`, depth multiplier `2`, separable filters `16`, separable kernel `16`, batch size `64`, max `30` epochs, patience `8`, train-only channel standardization.

| Model | Window | Protocol | Accuracy | Macro-F1 | Status |
| --- | --- | --- | ---: | ---: | --- |
| EEGNet | 2s | SD | 0.658873 | 0.651943 | full |
| EEGNet | 2s | LOSO | 0.522771 | 0.500855 | full |
| EEGNet | 4s | SD | 0.652174 | 0.648623 | full |
| EEGNet | 4s | LOSO | 0.762387 | 0.761061 | smoke, first 3 folds only |
| TemporalPatchTransformer | 4s | SD | 0.766908 | 0.767266 | repeated-seed mean |
| TemporalPatchTransformer | 4s | LOSO | 0.801520 | 0.801141 | repeated-seed mean |

Interpretation: using 4-second windows does not explain the subject-dependent Transformer advantage by itself. The tuned 4-second EEGNet subject-dependent result remains much lower than the 4-second Transformer subject-dependent repeated-seed mean.

The 4-second EEGNet LOSO smoke run finished 3 held-out subjects and reached higher scores than the 2-second full LOSO result, but full 28-fold LOSO was not completed in this commit because the CPU runtime is high. This should be treated as feasibility evidence, not a final LOSO result.

Source files:

- `results/subject_dependent_eegnet_4s_baseline.csv`
- `results/loso_eegnet_4s_smoke.csv`
- `results/loso_eegnet_4s_smoke_summary.csv`
- `results/eegnet_window_length_comparison.csv`
