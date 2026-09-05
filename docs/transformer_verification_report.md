# Transformer Verification Report

This report summarizes the current verification evidence for the GAMEEMO temporal-patch Transformer result. It is intended as lab-meeting and draft material, not as a final publication claim.

## Result Summary

| **Check** | **Result** | **Source** |
| --------- | ---------- | ---------- |
| Original Transformer LOSO run | Accuracy `0.798866`, macro-F1 `0.798241` across 28 held-out subjects | `results/loso_transformer_summary.csv` |
| Repeated-seed Transformer LOSO | Accuracy `0.801520 +/- 0.003501`, macro-F1 `0.801141 +/- 0.003476` across seeds `0, 1, 2` | `results/transformer_loso_repeated_seed_summary.csv` |
| Repeated-seed Transformer subject-dependent | Accuracy `0.766908 +/- 0.005195`, macro-F1 `0.767266 +/- 0.004914` across seeds `0, 1, 2` | `results/transformer_subject_dependent_repeated_seed_summary.csv` |
| LOSO split leakage audit | 28 folds checked; held-out subject in train = `False`, held-out subject in validation = `False`, held-out subject in test only = `True` for all folds | `results/loso_leakage_audit.csv` |
| Normalization leakage audit | 87 rows checked; leakage failures = `0`; normalization statistics fit on train windows only | `results/normalization_leakage_audit.csv` |
| Per-subject analysis | Best held-out subject: `S03` macro-F1 `0.884837`; worst held-out subject: `S26` macro-F1 `0.725831` | `results/transformer_loso_subject_analysis.csv` |
| Class-level F1 | boring `0.839435`, calm `0.764677`, horror `0.790225`, funny `0.800000` | `results/transformer_loso_class_f1.csv` |

## Model Configuration

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

## Verification Interpretation

The repeated-seed result is stable enough to present as a preliminary GAMEEMO finding. The LOSO mean changed only slightly from the original single run (`0.798866` accuracy, `0.798241` macro-F1) to the repeated-seed mean (`0.801520` accuracy, `0.801141` macro-F1), and the standard deviation across seeds is small. That makes the result less likely to be a lucky single initialization.

The leakage checks are also strong enough for a lab update. The explicit LOSO audit confirms that held-out subjects do not appear in training or validation folds. The normalization audit confirms that validation and test windows are transformed using statistics fit only from training windows, with zero reported leakage failures.

## 2s vs 4s Caveat

The Transformer result uses 4-second windows, while earlier statistical, bandpower, and tuned EEGNet full LOSO runs mainly used 2-second windows. This is a real protocol caveat. A 4-second window gives the model more temporal context and fewer total windows, so Transformer gains should not be framed as architecture-only gains yet.

The current EEGNet 4-second comparison is only a 3-fold LOSO smoke test, not a full 28-fold result. Until a full 4-second EEGNet LOSO run is completed, the strongest defensible statement is:

> The temporal-patch Transformer is currently the strongest verified GAMEEMO model under the tested 4-second setup, and its LOSO result is stable across seeds with no detected split or normalization leakage.

It is not yet safe to claim that the Transformer universally beats EEGNet independent of window length.

## AAAI Readiness Judgment

This result is strong enough for an AAAI student abstract progress story if it is framed as a controlled evaluation finding rather than as a final model claim. The credible argument is:

1. The project built a reproducible GAMEEMO pipeline with subject-dependent and LOSO protocols.
2. Classical statistical and bandpower baselines are above chance but limited.
3. EEGNet improves over classical baselines.
4. A temporal-patch Transformer gives the strongest current GAMEEMO result.
5. The surprising LOSO strength was checked with repeated seeds, split leakage audit, normalization leakage audit, confusion matrix, per-subject analysis, and class-level F1.

For the final paper direction, the remaining risk is external validity. The next most important step is not more random Transformer tuning; it is checking whether the same reliability story transfers to a second benchmark such as DREAMER, and completing fairer 4-second EEGNet comparison if time allows.
