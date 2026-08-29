# Do Modern EEG Emotion Models Generalize?

A protocol-sensitivity audit for EEG emotion recognition on GAMEEMO, with DREAMER as a follow-up benchmark if access is approved.

## Project Question

Do EEG emotion-recognition model conclusions change when evaluation moves from subject-dependent splits to subject-independent / leave-one-subject-out (LOSO) splits?

## Current Dataset Plan

### GAMEEMO: first experiment dataset

- 28 subjects
- 14-channel Emotiv EPOC+ EEG
- 128 Hz sampling rate
- Four game-condition emotion labels: boring, calm, horror, funny
- Local dataset downloaded
- Used first because it is accessible now and subject-organized

## Current Dataset Inspection

The local GAMEEMO inspection script confirms 28 subject folders, 112 raw EEG CSV files, 112 raw EEG MAT files, 112 preprocessed EEG CSV files, and 112 preprocessed EEG MAT files. A sample preprocessed CSV has 38,252 samples x 14 channels at 128 Hz, or about 298.84 seconds; planned 2-second windows will have shape 256 samples x 14 channels.

See `docs/gameemo_metadata.md` for the concise file structure and metadata summary.

See `docs/gameemo_prior_methods.md` for a conservative GAMEEMO prior-method comparison table. Published accuracies are treated as literature context, not direct comparisons, unless split, labels, preprocessing, and feature extraction match this repo's protocol.

## Current Baseline Status

The first working pipeline is complete for GAMEEMO preprocessed CSV files:

- Dataset inspected and preprocessed CSV selected for the first pass.
- GAMEEMO loader parses subject IDs, game IDs, and labels.
- Recordings are converted into 2-second non-overlapping windows at 128 Hz.
- Each window is represented with simple statistical features: mean, standard deviation, min, max, and mean squared value per channel.
- Subject-dependent logistic regression baseline is saved in `results/subject_dependent_baseline.csv`.
- Subject-dependent linear SVM baseline is saved in `results/subject_dependent_svm_baseline.csv`.
- LOSO logistic regression baseline is saved in `results/loso_baseline.csv` and summarized in `results/loso_summary.csv`.
- LOSO linear SVM baseline is saved in `results/loso_svm_baseline.csv` and summarized in `results/loso_svm_summary.csv`.

Current first-pass results with simple statistical window features:

| Protocol | Model | Windows | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| Random chance baseline | Uniform 4-class guess | 16,688 | 0.250000 | 0.250000 |
| Subject-dependent random split | Logistic regression | 16,688 | 0.376199 | 0.374935 |
| Subject-dependent random split | Linear SVM | 16,688 | 0.374700 | 0.372721 |
| LOSO mean across 28 folds | Logistic regression | 16,688 | 0.321309 | 0.299302 |
| LOSO mean across 28 folds | Linear SVM | 16,688 | 0.320709 | 0.298395 |

GAMEEMO is a 4-class task, so a uniform random baseline is 25%. Current results should be interpreted relative to this chance level and against internal protocol-controlled comparisons, not directly against published GAMEEMO numbers until preprocessing and split details are verified.

Both baseline model families are using the same rough feature representation. Stronger EEG-specific features, especially bandpower-style features, are the next step before drawing conclusions about model quality.

### EEGNet protocol gap

| Model | Input | Protocol | Accuracy | Macro-F1 | Chance | Accuracy drop vs subject-dependent | Macro-F1 drop vs subject-dependent |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | Raw EEG windows | Subject-dependent | 0.552458 | 0.543291 | 0.250000 | 0.234625 | 0.278458 |
| EEGNet | Raw EEG windows | LOSO | 0.317833 | 0.264833 | 0.250000 | N/A | N/A |

EEGNet improves subject-dependent performance over the current classical baselines. The LOSO result determines whether that gain transfers to unseen subjects; the current short-trained CPU LOSO run shows a large protocol gap, so cross-subject generalization remains the main bottleneck.

### DREAMER: follow-up benchmark

- 23 subjects
- EEG plus peripheral signals
- Valence, arousal, dominance labels
- Access request pending
- More recognized benchmark for follow-up reliability checks

## Planned Evaluation

| Protocol | Description | Purpose |
| --- | --- | --- |
| Subject-dependent | Windows from the same subjects can appear in train and test | Easier within-subject setting |
| LOSO | One entire subject is held out for test | Unseen-subject generalization |

Main quantity:

```text
protocol drop = subject-dependent score - LOSO score
```

## Planned Models

1. Classical baseline: logistic regression or SVM using window-level EEG features
2. EEG neural baseline: EEGNet or lightweight CNN
3. Modern model candidate: Transformer / Vision Transformer-style EEG model if feasible

## Planned Metrics

- Accuracy
- Macro-F1
- Per-subject LOSO performance
- Per-subject variance / standard deviation
- Protocol drop

## Known Limitations

- These are first-pass baseline results, not final model conclusions.
- Statistical features are a simple baseline and are not final EEG bandpower features.
- GAMEEMO preprocessing details still need verification against the dataset documentation.
- EEGNet and Transformer / ViT-style models are next model tiers after the baseline pipeline is stable.

## Next Steps

1. Stabilize the baseline pipeline and result reporting.
2. Add an EEGNet or lightweight CNN baseline.
3. Evaluate whether a Transformer / ViT-style EEG model is feasible for this dataset size and protocol.

## Notes

This project is evaluation-focused, not primarily a new architecture paper. The central audit asks whether model gains remain stable under stricter unseen-subject evaluation.
