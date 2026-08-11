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

## Immediate TODO

- [ ] Inspect GAMEEMO subject folders and file formats
- [ ] Choose first file type: preprocessed CSV or MAT
- [ ] Load one EEG recording and print shape
- [ ] Implement 2-second windowing
- [ ] Map labels: boring, calm, horror, funny
- [ ] Implement subject-dependent split
- [ ] Implement LOSO split
- [ ] Run first logistic regression / SVM baseline
- [ ] Save first result table in `results/`

## Notes

This project is evaluation-focused, not primarily a new architecture paper. The central audit asks whether model gains remain stable under stricter unseen-subject evaluation.
