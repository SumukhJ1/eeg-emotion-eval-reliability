# Transformer split composition analysis

Comparison of the GAMEEMO 4-second temporal-patch Transformer subject-dependent split against LOSO folds.

## Train size

| Protocol | Fit windows | Val windows | Test windows | Note |
| --- | ---: | ---: | ---: | --- |
| Subject-dependent | 5305 | 1327 | 1656 | One random stratified window split across all subjects. |
| LOSO per fold | 6393 | 1599 | 296 | Holds out one subject; trains on the other 27 subjects. |

The biggest structural difference is train size: LOSO fits on many more windows per fold because only one subject is held out before the validation split.

## Class and game balance

The subject-dependent test split is exactly balanced by label and game: 414 windows per class/game. The lower subject-dependent Transformer score is not explained by a harder class-balance distribution.

## Subject coverage

The subject-dependent test split contains windows from 28 subjects. Each LOSO test fold contains exactly one subject, with all windows from that subject held out.

## LOSO subject distribution

| Hardest held-out subjects | Mean acc | Mean F1 |
| --- | ---: | ---: |
| S09 | 0.740991 | 0.739085 |
| S26 | 0.742117 | 0.744300 |
| S24 | 0.745496 | 0.744731 |
| S12 | 0.747748 | 0.748555 |
| S14 | 0.768018 | 0.768350 |

| Easiest held-out subjects | Mean acc | Mean F1 |
| --- | ---: | ---: |
| S03 | 0.881757 | 0.880815 |
| S04 | 0.876126 | 0.875212 |
| S21 | 0.850225 | 0.850935 |
| S22 | 0.850225 | 0.849662 |
| S20 | 0.849099 | 0.847878 |

## Class and game accuracy gap

Current Transformer outputs save aggregate test accuracy/F1 only. They do not save per-window predictions, so true per-game or per-class Transformer accuracy cannot be reconstructed from the existing result files. The saved files cover per-class and per-game split composition; per-class/per-game accuracy requires prediction logging.

## Interpretation

- LOSO is not inherently easier, but in this pipeline it trains on a larger fit set per fold than the subject-dependent split.
- GAMEEMO labels are tied to game conditions, so broad training coverage from 27 subjects may make held-out-subject recognition easier than expected.
- Subject-dependent testing samples windows from every subject, so it may include a more mixed collection of easy and hard subject/window cases in one test set.
- The stable LOSO > subject-dependent result should be reported as a protocol-composition finding, not overclaimed as a model breakthrough.

## Files

- `results/transformer_subject_dependent_repeated_seed_results.csv`
- `results/transformer_loso_repeated_seed_results.csv`
- `results/transformer_split_composition/split_counts.csv`
- `results/transformer_split_composition/train_size_comparison.csv`
- `results/transformer_split_composition/loso_subject_distribution.csv`
- `results/transformer_split_composition/per_game_class_accuracy_availability.csv`
