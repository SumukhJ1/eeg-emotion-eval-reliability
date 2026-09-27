# Transformer LOSO Repeated-Seed Runs

This folder stores fold-level and seed-level outputs for the repeated-seed temporal-patch Transformer LOSO check. Full seed-0 fold rows live at top level in `../transformer_loso_subject_analysis.csv`; this folder contains seed 1 and seed 2 fold rows plus seed summary files.

| File | Description |
| --- | --- |
| `loso_seed_0_summary.csv` | Seed-0 LOSO mean accuracy, macro-F1, fold standard deviations, and fixed model config. |
| `loso_seed_1.csv` | Seed-1 fold-level LOSO rows, one held-out subject per row. |
| `loso_seed_1_summary.csv` | Seed-1 LOSO mean accuracy, macro-F1, fold standard deviations, and fixed model config. |
| `loso_seed_2.csv` | Seed-2 fold-level LOSO rows, one held-out subject per row. |
| `loso_seed_2_summary.csv` | Seed-2 LOSO mean accuracy, macro-F1, fold standard deviations, and fixed model config. |
