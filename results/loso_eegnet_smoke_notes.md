# EEGNet LOSO Smoke Test Notes

This is a partial runtime and leakage-check smoke test, not a final LOSO EEGNet result.

## Run Settings

- Script: `scripts/run_loso_eegnet_baseline.py`
- Output: `results/loso_eegnet_smoke.csv`
- Held-out subjects: S01, S02, S03, S04
- Epochs per fold: 3
- Batch size: 64
- Learning rate: 0.001
- Weight decay: 0.0001
- Device: CPU
- Window shape: 14 channels x 256 samples
- Total windows loaded: 16,688

## Runtime / Memory

- Wall-clock runtime: 623.60 seconds for 4 folds
- Approximate runtime per fold: 155.90 seconds
- No memory failure or process crash was observed during the 4-fold CPU run.

## Leakage Check

The LOSO runner validates each fold before training. For this smoke run, each fold used:

- Train windows: 12,873
- Validation windows: 3,219
- Test windows: 596

Validation windows are split only from the 27 training subjects. The held-out subject is checked not to appear in either training or validation.

## Smoke Metrics

| Held-out subject | Test accuracy | Test macro-F1 | Best epoch |
| --- | ---: | ---: | ---: |
| S01 | 0.406040 | 0.397375 | 3 |
| S02 | 0.328859 | 0.240336 | 3 |
| S03 | 0.672819 | 0.666488 | 3 |
| S04 | 0.521812 | 0.453540 | 3 |

Partial 4-fold mean:

- Mean test accuracy: 0.482382
- Mean test macro-F1: 0.439435
- Accuracy range: 0.328859 to 0.672819

These values confirm that the LOSO EEGNet runner executes and saves per-subject fold results. They should not be presented as the final 28-fold LOSO EEGNet score.
