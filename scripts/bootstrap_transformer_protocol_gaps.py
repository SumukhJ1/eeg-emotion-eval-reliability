"""Subject-level cluster bootstrap for temporal-patch Transformer protocol gaps."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR


DEFAULT_SEEDS = list(range(10))
LOSO_SEEDS = [0, 1, 2]
N_SUBJECTS = 28
N_RESAMPLES = 10_000
RANDOM_SEED = 1729


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in raw.split(",") if seed.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def read_prediction_subject_accuracies(pattern: str, seeds: list[int]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for seed in seeds:
        path = RESULTS_DIR / "predictions" / pattern.format(seed=seed)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                grouped[row["subject_id"]].append(row["correct"] == "True")
    return {subject: sum(values) / len(values) for subject, values in grouped.items()}


def read_loso_subject_accuracies() -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    seed_paths = {
        0: RESULTS_DIR / "loso_transformer_baseline.csv",
        1: RESULTS_DIR / "transformer_loso_repeated_seed_runs" / "loso_seed_1.csv",
        2: RESULTS_DIR / "transformer_loso_repeated_seed_runs" / "loso_seed_2.csv",
    }
    for seed in LOSO_SEEDS:
        path = seed_paths[seed]
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                grouped[row["subject"]].append(float(row["test_accuracy"]))
    return {subject: sum(values) / len(values) for subject, values in grouped.items()}


def paired_wilcoxon(values: list[float]) -> float:
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise SystemExit("scipy is required for the paired Wilcoxon signed-rank test.") from exc

    result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.pvalue)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(path: Path, values, title: str, xlabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to save bootstrap histograms.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=40, color="#4c78a8", edgecolor="white")
    ax.axvline(float(values.mean()), color="#f58518", linewidth=2, label="mean")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("bootstrap resamples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap subject-level Transformer protocol accuracy gaps.")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RESULTS_DIR / "subject_level_bootstrap_summary.csv",
    )
    parser.add_argument(
        "--resamples-output",
        type=Path,
        default=RESULTS_DIR / "subject_level_bootstrap_resamples.csv",
    )
    parser.add_argument(
        "--per-subject-output",
        type=Path,
        default=RESULTS_DIR / "subject_level_protocol_accuracies.csv",
    )
    args = parser.parse_args()

    if args.n_resamples <= 0:
        raise SystemExit(f"n-resamples must be positive, got {args.n_resamples}")

    import numpy as np

    sd = read_prediction_subject_accuracies(
        "TemporalPatchTransformer_subject_dependent_seed{seed}_predictions.csv",
        args.seeds,
    )
    matched = read_prediction_subject_accuracies(
        "TemporalPatchTransformer_subject_dependent_matched_budget_seed{seed}_predictions.csv",
        args.seeds,
    )
    loso = read_loso_subject_accuracies()

    subjects = sorted(set(sd) & set(matched) & set(loso))
    if len(subjects) != N_SUBJECTS:
        raise SystemExit(f"Expected {N_SUBJECTS} subjects, got {len(subjects)}")

    sd_values = np.array([sd[subject] for subject in subjects], dtype=float)
    matched_values = np.array([matched[subject] for subject in subjects], dtype=float)
    loso_values = np.array([loso[subject] for subject in subjects], dtype=float)
    loso_minus_sd = loso_values - sd_values
    matched_minus_loso = matched_values - loso_values

    rng = np.random.default_rng(args.random_seed)
    resample_rows: list[dict[str, object]] = []
    loso_minus_sd_samples = np.empty(args.n_resamples, dtype=float)
    matched_minus_loso_samples = np.empty(args.n_resamples, dtype=float)
    for idx in range(args.n_resamples):
        sampled = rng.integers(0, len(subjects), size=len(subjects))
        loso_minus_sd_samples[idx] = float(loso_minus_sd[sampled].mean())
        matched_minus_loso_samples[idx] = float(matched_minus_loso[sampled].mean())
        resample_rows.append(
            {
                "resample": idx,
                "loso_minus_original_sd_accuracy_gap": f"{loso_minus_sd_samples[idx]:.8f}",
                "matched_budget_sd_minus_loso_accuracy_gap": f"{matched_minus_loso_samples[idx]:.8f}",
            }
        )

    per_subject_rows = []
    for subject, sd_acc, matched_acc, loso_acc in zip(subjects, sd_values, matched_values, loso_values):
        per_subject_rows.append(
            {
                "subject": subject,
                "subject_dependent_accuracy": f"{sd_acc:.8f}",
                "matched_budget_accuracy": f"{matched_acc:.8f}",
                "loso_accuracy": f"{loso_acc:.8f}",
                "loso_minus_subject_dependent": f"{loso_acc - sd_acc:.8f}",
                "matched_budget_minus_loso": f"{matched_acc - loso_acc:.8f}",
            }
        )

    summary_rows = [
        {
            "comparison": "loso_minus_original_sd_accuracy_gap",
            "point_estimate": f"{float(loso_minus_sd.mean()):.8f}",
            "ci_2_5": f"{float(np.percentile(loso_minus_sd_samples, 2.5)):.8f}",
            "ci_97_5": f"{float(np.percentile(loso_minus_sd_samples, 97.5)):.8f}",
            "wilcoxon_p_value": f"{paired_wilcoxon(loso_minus_sd):.8f}",
            "n_subjects": len(subjects),
            "n_resamples": args.n_resamples,
            "rng_seed": args.random_seed,
        },
        {
            "comparison": "matched_budget_sd_minus_loso_accuracy_gap",
            "point_estimate": f"{float(matched_minus_loso.mean()):.8f}",
            "ci_2_5": f"{float(np.percentile(matched_minus_loso_samples, 2.5)):.8f}",
            "ci_97_5": f"{float(np.percentile(matched_minus_loso_samples, 97.5)):.8f}",
            "wilcoxon_p_value": f"{paired_wilcoxon(matched_minus_loso):.8f}",
            "n_subjects": len(subjects),
            "n_resamples": args.n_resamples,
            "rng_seed": args.random_seed,
        },
    ]

    write_csv(
        args.per_subject_output,
        per_subject_rows,
        [
            "subject",
            "subject_dependent_accuracy",
            "matched_budget_accuracy",
            "loso_accuracy",
            "loso_minus_subject_dependent",
            "matched_budget_minus_loso",
        ],
    )
    write_csv(
        args.resamples_output,
        resample_rows,
        [
            "resample",
            "loso_minus_original_sd_accuracy_gap",
            "matched_budget_sd_minus_loso_accuracy_gap",
        ],
    )
    write_csv(
        args.summary_output,
        summary_rows,
        [
            "comparison",
            "point_estimate",
            "ci_2_5",
            "ci_97_5",
            "wilcoxon_p_value",
            "n_subjects",
            "n_resamples",
            "rng_seed",
        ],
    )

    save_histogram(
        RESULTS_DIR / "subject_level_bootstrap_loso_minus_sd_histogram.png",
        loso_minus_sd_samples,
        "LOSO minus original SD accuracy gap",
        "accuracy gap",
    )
    save_histogram(
        RESULTS_DIR / "subject_level_bootstrap_matched_minus_loso_histogram.png",
        matched_minus_loso_samples,
        "Matched-budget SD minus LOSO accuracy gap",
        "accuracy gap",
    )

    print(f"subjects: {len(subjects)}")
    print(f"subject_dependent_mean_accuracy: {sd_values.mean():.6f}")
    print(f"matched_budget_mean_accuracy: {matched_values.mean():.6f}")
    print(f"loso_mean_accuracy: {loso_values.mean():.6f}")
    for row in summary_rows:
        print(
            f"{row['comparison']}: point={row['point_estimate']} "
            f"ci=[{row['ci_2_5']}, {row['ci_97_5']}] "
            f"wilcoxon_p={row['wilcoxon_p_value']}"
        )
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.resamples_output}")
    print(f"wrote: {args.per_subject_output}")


if __name__ == "__main__":
    main()
