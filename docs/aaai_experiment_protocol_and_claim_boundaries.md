# Experiment protocol and scope

Current evidence boundary for the draft and lab discussion.

## Dataset scope

- Main dataset: GAMEEMO
- DREAMER status: future/secondary benchmark, not part of the current main evidence
- Task: four-class GAMEEMO game-condition classification
- Labels: boring, calm, horror, funny
- Chance baseline: 25% uniform random chance

## Evaluation protocol

- Subject-dependent: random window split where windows from the same subjects may appear in train and test
- LOSO: leave-one-subject-out evaluation where each fold holds out one full subject for test
- Matched-budget subject-dependent control: subject-dependent split with train/validation/test window counts matched to the mean LOSO budget
- Metrics: accuracy and macro-F1
- Stability: repeated seeds are reported where available
- LOSO reporting: full 28 subject folds are reported for completed LOSO experiments

## Included final models

- Logistic regression on time-statistical features
- Linear SVM on log-relative bandpower features
- EEGNet on raw EEG windows
- Channel-token Transformer on raw EEG windows
- Temporal-patch Transformer on raw EEG windows

## Scope

No SOTA claim is made.

No broad EEG generalization claim is made.

No direct comparison to published GAMEEMO papers is claimed unless the task, preprocessing, split protocol, labels, and evaluation unit match.

DREAMER is not used as main evidence yet.

## Main claim

The main claim is protocol and representation sensitivity:

> On GAMEEMO, conclusions depend strongly on evaluation protocol and raw EEG representation. Temporal-patch Transformer tokenization is the strongest tested representation under the current controlled setup, and matched-budget plus permutation checks help defend the result against obvious split-size and label-independent artifact explanations.

## Current evidence package

- Final result table: `docs/final_gameemo_results.md`
- Final model ablation table: `docs/final_model_ablation_table.md`
- Transformer tokenization ablation: `docs/transformer_tokenization_ablation.md`
- Protocol-composition control: `docs/transformer_protocol_composition_effects.md`
- Transformer verification report: `docs/transformer_verification_report.md`
- GAMEEMO prior-method caveats: `docs/gameemo_prior_methods.md`
