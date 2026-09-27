# GAMEEMO conservative filtering rules

These rules define the first conservative signal-quality filtering policy before any filtered baseline results are reported. The goal is to avoid tuning filtering decisions after seeing model performance.

## Current audit inputs

The signal-quality audit uses preprocessed GAMEEMO CSV files converted into 2-second non-overlapping windows at 128 Hz. The current audit output is saved in:

- `results/gameemo_signal_quality_audit_summary.csv`
- `results/gameemo_signal_quality_audit_by_subject.csv`
- `results/gameemo_signal_quality_flagged_windows.csv`

Current audit summary:

| Check | Count |
| --- | ---: |
| Total subjects | 28 |
| Total records | 112 |
| Total windows | 16,688 |
| Missing-value windows | 0 |
| Nonfinite-value windows | 0 |
| Near-zero variance windows | 0 |
| Flat-channel windows | 0 |
| Extreme-amplitude windows | 17 |
| Extreme-variance windows | 879 |

## Removal rules

Only windows that meet one of the following hard quality-failure rules should be removed in the first filtered rerun:

| Rule | Threshold | Action | Reason |
| --- | --- | --- | --- |
| Missing values | Any `NaN` value in the window | Remove window | Numeric models cannot reliably interpret missing EEG samples without imputation. |
| Nonfinite values | Any `inf` or `-inf` value in the window | Remove window | Nonfinite values break scaling, feature extraction, and neural training. |
| Near-zero window variance | Whole-window variance `<= 1e-12` | Remove window | A near-constant EEG window carries no useful time-varying signal. |
| Flat channel | Any channel variance `<= 1e-12` within a window | Remove window | A flat channel suggests dropout or unusable channel data for that window. |

These rules remove obvious invalid or degenerate windows only. On the current audit, they would remove 0 of 16,688 windows.

## Flag-only rules

The following windows should be flagged for analysis but not automatically removed in the first filtered rerun:

| Flag | Current threshold | Current count | First-pass action |
| --- | ---: | ---: | --- |
| Extreme amplitude | max absolute amplitude `> 370.145550` | 17 | Flag only |
| Extreme variance | whole-window variance `> 1255.610078` | 879 | Flag only |

The thresholds above were computed by the audit script using robust upper thresholds over window-level values. These are potential artifact indicators, but they are not automatically removed yet because high amplitude or high variance can also reflect valid subject/game differences. Removing them requires a separately documented rule before running filtered results.

## Reporting policy

Filtered results should clearly state:

- which rules were applied,
- how many windows were removed,
- whether removals were applied before subject-dependent and LOSO splits,
- whether any subjects or classes lost a disproportionate number of windows,
- and whether extreme-amplitude or extreme-variance windows were only flagged or actually removed.

The first filtered baseline should apply only the hard removal rules above. If it does not materially change accuracy or macro-F1, report that directly; that would suggest the current bottleneck is not simple missing, nonfinite, flat, or near-zero-variance artifacts.
