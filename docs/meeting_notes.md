# Lab Meeting Prep Notes

## Current Framing

GAMEEMO is the immediate working dataset. DREAMER remains a more recognized follow-up benchmark if access is approved.

The current lab-meeting story is that the full first-pass GAMEEMO baseline pipeline is now running end to end. The central comparison is still subject-dependent performance versus leave-one-subject-out performance.

## Current Baseline Status

- Dataset inspected: 28 subjects, 14 EEG channels, 128 Hz sampling rate.
- Preprocessed CSV files chosen for the first pass.
- GAMEEMO loader implemented for subject IDs, game IDs, labels, and metadata.
- 2-second non-overlapping windows implemented, with 256 samples per window.
- Statistical window features implemented: mean, standard deviation, min, max, and mean squared value per channel.
- Subject-dependent logistic regression baseline completed.
- LOSO logistic regression baseline completed.

## First Results

| Protocol | Windows | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Subject-dependent random split | 16,688 | 0.376199 | 0.374935 |
| LOSO mean across 28 folds | 16,688 | 0.321309 | 0.299302 |

Saved outputs:

- `results/subject_dependent_baseline.csv`
- `results/loso_baseline.csv`
- `results/loso_summary.csv`

## Feedback Incorporated

- Go deeper on the accessible dataset instead of waiting for DREAMER.
- Add publication years to the prior-results table.
- Check preprocessing choices and possible outlier effects.
- Treat Transformer / ViT-style models as a modern-model extension after the baseline pipeline is stable.

## Next Meeting Target

Show code-level progress and first-pass baseline evidence:

- GAMEEMO structure inspected
- Loader, windowing, split generation, and feature extraction implemented
- Subject-dependent and LOSO logistic regression baselines completed
- Early protocol drop visible: LOSO mean macro-F1 is lower than the subject-dependent macro-F1

## Known Limitations

- These are first-pass results, not final conclusions.
- Statistical features are intentionally simple and are not final EEG bandpower features.
- GAMEEMO preprocessing details still need verification.
- EEGNet and Transformer / ViT-style models are next model tiers after the baseline is stable.

## Next Steps

1. Stabilize the baseline pipeline and result reporting.
2. Add EEGNet or a lightweight CNN as the next model tier.
3. Evaluate Transformer / ViT feasibility after the neural baseline is working.
