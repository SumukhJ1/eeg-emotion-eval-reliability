# Do Modern EEG Emotion Models Generalize?

A protocol-sensitivity and representation-sensitivity audit for EEG emotion recognition on GAMEEMO.

## Project Question

Do EEG emotion-recognition conclusions change when the same dataset, labels, and evaluation code are tested across subject-dependent splits, leave-one-subject-out (LOSO) splits, and different EEG input representations?

## Frozen GAMEEMO Setup

GAMEEMO is the main dataset for the current evidence package.

- 28 subjects
- 14-channel Emotiv EPOC+ EEG
- 128 Hz sampling rate
- Four game-condition labels: boring, calm, horror, funny
- Uniform random chance baseline: 25%
- Primary input: preprocessed GAMEEMO CSV files
- Main window settings: 2-second windows for classical/EEGNet baselines; 4-second windows for the strongest Transformer and matched 4-second EEGNet checks

DREAMER is treated as a future/secondary benchmark, not as current main evidence.

## Current Dataset Inspection

The local GAMEEMO inspection script confirms 28 subject folders, 112 raw EEG CSV files, 112 raw EEG MAT files, 112 preprocessed EEG CSV files, and 112 preprocessed EEG MAT files. A sample preprocessed CSV has 38,252 samples x 14 channels at 128 Hz, or about 298.84 seconds.

See `docs/gameemo_metadata.md` for the file structure and metadata summary.

## Frozen Result Summary

All results are for four-class GAMEEMO game-condition classification. Published GAMEEMO numbers should be treated as literature context, not direct comparisons, unless preprocessing, labels, split protocol, and evaluation unit match.

| Input / representation | Model | Window | Subject-dependent acc | LOSO acc | Macro-F1 | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| Time-statistical features | Logistic regression | 2s | 0.376199 | 0.321309 | 0.374935 / 0.299302 | Weak but above chance. |
| Log-relative bandpower | Linear SVM | 2s | 0.469724 | 0.388063 | 0.466975 / 0.367017 | EEG-specific features improve classical baselines. |
| Raw EEG | EEGNet | 2s | 0.658873 | 0.522771 | 0.651943 / 0.500855 | Compact EEG neural baseline. |
| Raw EEG | EEGNet | 4s | 0.652174 | 0.564310 | 0.648623 / 0.543326 | Fairer 4-second neural comparison. |
| Raw EEG channel tokens | Transformer | 4s | 0.650362 | 0.740830 | 0.650782 / 0.739784 | LOSO is one seed. |
| Raw EEG temporal patches | Transformer | 4s | 0.766908 | 0.801520 | 0.767266 / 0.801141 | Strongest current result; repeated seeds where available. |

See:

- `docs/final_gameemo_results.md`
- `docs/final_model_ablation_table.md`
- `docs/transformer_tokenization_ablation.md`
- `docs/transformer_protocol_composition_effects.md`
- `docs/transformer_verification_report.md`

## Current Pipeline

- GAMEEMO loader discovers preprocessed CSV files and parses subject/game metadata.
- Recordings are converted into fixed-length EEG windows.
- Classical baselines use statistical and log-relative bandpower features.
- Neural baselines use raw-window EEGNet and Transformer models.
- Subject-dependent and LOSO split helpers include leakage checks.
- Neural normalization uses train-only channel statistics.
- Transformer verification includes repeated seeds, split leakage audit, normalization leakage audit, matched-budget control, permutation-label sanity check, subject-permutation check, confusion matrix, and per-subject analysis.

## Claim Boundaries

This repo does not currently claim state of the art and does not claim broad EEG generalization. The defensible current claim is narrower:

> GAMEEMO results are highly sensitive to evaluation protocol and EEG representation. Under the tested setup, temporal-patch Transformer tokenization outperforms channel-token Transformer input, EEGNet, and classical bandpower/statistical baselines, while matched-budget and permutation controls help explain and validate the result.

## Known Limitations

- GAMEEMO is the only completed dataset so far.
- DREAMER remains a future/secondary benchmark for external validation.
- Published GAMEEMO comparisons may use different labels, preprocessing, splits, or evaluation units.
- Channel-token Transformer LOSO has one full seed; temporal-patch Transformer LOSO has repeated seeds.
- Statistical and bandpower features are first-pass baselines, not a final EEG feature study.

## Next Steps

1. Freeze the GAMEEMO tables and use them in the AAAI draft as preliminary evidence.
2. Add DREAMER only as a secondary benchmark if time allows.
3. Keep the main paper story focused on protocol sensitivity, representation sensitivity, and leakage-controlled evaluation.

## Paper Audit Artifacts

The current AAAI student-abstract draft is backed by explicit provenance notes:

- `docs/paper_claim_provenance.md` maps each major paper claim to the exact repository result files.
- `docs/citation_verification.md` records which local PDFs were checked for the GAMEEMO dataset, prior GAMEEMO results, chance-level interpretation, and evaluation-reliability citations.

These files are intended to make the draft easier to review and to prevent unsupported claims from entering the final submission.
