# Normalization Leakage Audit

This audit checks that GAMEEMO neural normalization statistics are fit on training windows only, then reused for validation and test windows.

Rows audited: 300
Leakage failures: 0

| Protocol | Fold/seed | Normalization source | Test data used in fit? |
| --- | --- | --- | --- |
| subject_dependent | seed_0 | train only | False |
| subject_dependent_matched_budget | seed_0 | train only | False |
| loso | loso_S01_seed_0 | train only | False |
| loso | loso_S02_seed_0 | train only | False |
| loso | loso_S03_seed_0 | train only | False |
| loso | loso_S04_seed_0 | train only | False |
| loso | loso_S05_seed_0 | train only | False |
| loso | loso_S06_seed_0 | train only | False |
| ... | 292 more rows in CSV | train only | False |

The detailed CSV also records whether validation windows overlap with fit windows and whether each LOSO held-out subject appears in the normalization fit set.

Expected result: `normalization_source=train only`, `test_data_used_in_fit=False`, and `held_out_subject_used_in_fit=False` for all LOSO folds.
