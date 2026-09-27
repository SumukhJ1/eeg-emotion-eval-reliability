# Does Cross-Subject EEG Evaluation Measure Generalization or Data Budget? A Controlled Study on GAMEEMO

This is the code for my controlled GAMEEMO EEG emotion-recognition study for a 2-page paper. I put the dataset inspection scripts, windowing code, classical feature baselines, EEGNet baselines, Transformer baselines, leakage audits, permutation checks, and result summaries in this repo. The main task is four-class game-condition classification on GAMEEMO: boring, calm, horror, and funny. The main finding is that the apparent LOSO advantage over subject-dependent evaluation collapses to about 0.25 percentage points after matching the train/validation/test window budget. My current claim is narrow: on GAMEEMO, conclusions change when evaluation budget, split protocol, and EEG representation are controlled. All numbers reported in the paper trace to files under `results/`, with the main claim map in [`docs/paper_claim_provenance.md`](docs/paper_claim_provenance.md).

## Quick start

Create an environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-eegnet.txt
```

Run the key repeated-seed Transformer checks:

```bash
python scripts/run_repeated_seed_transformer_experiments.py --root "<path-to-GAMEEMO-root>" --protocol subject_dependent --seeds 0,1,2 --output results/transformer_subject_dependent_repeated_seed_results.csv --summary-output results/transformer_subject_dependent_repeated_seed_summary.csv --run-dir results/transformer_subject_dependent_repeated_seed_runs

python scripts/run_repeated_seed_transformer_experiments.py --root "<path-to-GAMEEMO-root>" --protocol subject_dependent --split-protocol subject_dependent_matched_budget --seeds 0,1,2 --output results/transformer_matched_budget_repeated_seed_results.csv --summary-output results/transformer_matched_budget_repeated_seed_summary.csv --run-dir results/transformer_matched_budget_repeated_seed_runs

python scripts/run_repeated_seed_transformer_experiments.py --root "<path-to-GAMEEMO-root>" --protocol loso --seeds 0,1,2 --output results/transformer_loso_repeated_seed_results.csv --summary-output results/transformer_loso_repeated_seed_summary.csv --run-dir results/transformer_loso_repeated_seed_runs
```

These commands write per-seed outputs and summary CSVs under `results/`. The CSVs in `results/` are frozen; I don't overwrite them during cleanup passes.

## Main results

All rows use the 4-second temporal-patch Transformer on GAMEEMO with train-only channel standardization and seeds `0,1,2`.

| Protocol | Train windows | Validation windows | Test design | Accuracy | Macro-F1 | Notes |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Original subject-dependent | 5,305 | 1,327 | Random mixed-subject test windows | 76.7 +/- 0.5 | 76.7 +/- 0.5 | From `results/transformer_subject_dependent_repeated_seed_summary.csv`. |
| Matched-budget subject-dependent | 6,393 | 1,599 | Random mixed-subject test windows, LOSO-sized test set | 80.4 +/- 0.9 | 80.5 +/- 1.0 | From `results/transformer_matched_budget_repeated_seed_summary.csv`. |
| LOSO | 6,393 | 1,599 | One held-out subject per fold | 80.2 | 80.1 | Seed s.d. 0.4; fold s.d. 4.3; fold range 70.3-88.5. |

The matched-budget subject-dependent accuracy is 80.4054%, and the LOSO accuracy is 80.1520%. Their difference is 0.2534 percentage points.

## Directory map

- `src/`: Dataset loading, windowing, split logic, features, EEGNet, Transformer, normalization, and quality-filter helpers.
- `scripts/`: Dataset inspection, training runners, audits, summaries, and analysis scripts.
- `results/`: Frozen CSV outputs, verification notes, and result README files.
- `docs/`: Dataset notes, claim provenance, protocol notes, prior-method caveats, and summaries for the CSVs reported in the paper.
- `notebooks/`: Working notebooks, if used for local exploration.
- `data/`: Placeholder location only; GAMEEMO files are not copied into the repo.

## Scope

I'm not claiming this beats published GAMEEMO methods. I'm not claiming broad EEG emotion-recognition generalization across datasets. My current claim is narrower: on GAMEEMO, reported model conclusions change when evaluation budget, split protocol, and EEG representation are controlled carefully.
