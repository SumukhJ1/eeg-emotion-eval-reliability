# Transformer verification report

Current verification evidence for the GAMEEMO temporal-patch Transformer result. This is lab-meeting and draft material, with more checks still needed before final publication claims.

## Result summary

| **Check** | **Result** | **Source** |
| --------- | ---------- | ---------- |
| Original Transformer LOSO run | Accuracy `0.798866`, macro-F1 `0.798241` across 28 held-out subjects | `results/loso_transformer_summary.csv` |
| Repeated-seed Transformer LOSO | Accuracy `0.801520 +/- 0.003501`, macro-F1 `0.801141 +/- 0.003476` across seeds `0, 1, 2` | `results/transformer_loso_repeated_seed_summary.csv` |
| Repeated-seed Transformer subject-dependent | Accuracy `0.766908 +/- 0.005195`, macro-F1 `0.767266 +/- 0.004914` across seeds `0, 1, 2` | `results/transformer_subject_dependent_repeated_seed_summary.csv` |
| Full 4-second EEGNet LOSO comparison | Accuracy `0.564310`, macro-F1 `0.543326` across 28 held-out subjects | `results/loso_eegnet_4s_summary.csv` |
| LOSO split leakage audit | 28 folds checked; held-out subject in train = `False`, held-out subject in validation = `False`, held-out subject in test only = `True` for all folds | `results/loso_leakage_audit.csv` |
| Normalization leakage audit | 87 rows checked; leakage failures = `0`; normalization statistics fit on train windows only | `results/normalization_leakage_audit.csv` |
| Per-subject analysis | Best held-out subject: `S03` macro-F1 `0.884837`; worst held-out subject: `S26` macro-F1 `0.725831` | `results/transformer_loso_subject_analysis.csv` |
| Class-level F1 | boring `0.839435`, calm `0.764677`, horror `0.790225`, funny `0.800000` | `results/transformer_loso_class_f1.csv` |

## Model configuration

The verified Transformer uses raw GAMEEMO EEG windows with train-only channel standardization. The strongest checked setting is:

- Input mode: temporal patches
- Window length: 4 seconds, or 512 samples at 128 Hz
- Patch size: 32 samples
- `d_model`: 64
- Heads: 4
- Layers: 2
- Feedforward dimension: 128
- Learning rate: 0.001
- Dropout: 0.1
- Epochs: 30
- Patience: 8
- Batch size: 64
- Weight decay: 0.0001

## Verification interpretation

The repeated-seed result is stable enough to present as a preliminary GAMEEMO finding. The LOSO mean changed only slightly from the original single run (`0.798866` accuracy, `0.798241` macro-F1) to the repeated-seed mean (`0.801520` accuracy, `0.801141` macro-F1), and the standard deviation across seeds is small. That makes the result less likely to be a lucky single initialization.

The leakage checks are also strong enough for a lab update. The LOSO audit shows that held-out subjects do not appear in training or validation folds. The normalization audit shows that validation and test windows use statistics fit only from training windows, with zero reported leakage failures.

## 2s vs 4s caveat

The Transformer result uses 4-second windows, while the statistical and bandpower baselines use 2-second windows. A 4-second window gives the model more temporal context and fewer total windows, so window length remains part of the experimental protocol and should be reported clearly.

The full 4-second EEGNet LOSO comparison is now complete across all 28 held-out subjects. Its mean accuracy is `0.564310` and mean macro-F1 is `0.543326`, below the temporal-patch Transformer repeated-seed LOSO mean of `0.801520` accuracy and `0.801141` macro-F1. The strongest defensible statement is:

> The temporal-patch Transformer is currently the strongest verified GAMEEMO model under the tested 4-second setup, and its LOSO result is stable across seeds with no detected split or normalization leakage.

Broad superiority beyond GAMEEMO would require the same pattern on a second benchmark such as DREAMER.

## Readiness judgment

This result is strong enough for a student-abstract progress story if it is framed as a controlled evaluation finding rather than as a final model claim. The credible argument is:

1. The project built a reproducible GAMEEMO pipeline with subject-dependent and LOSO protocols.
2. Classical statistical and bandpower baselines are above chance but limited.
3. EEGNet improves over classical baselines.
4. A temporal-patch Transformer gives the strongest current GAMEEMO result.
5. The surprising LOSO strength was checked with repeated seeds, split leakage audit, normalization leakage audit, confusion matrix, per-subject analysis, and class-level F1.

For the final paper direction, the remaining risk is external validity. The next most important step is not more random Transformer tuning; it is checking whether the same reliability story transfers to a second benchmark such as DREAMER.
