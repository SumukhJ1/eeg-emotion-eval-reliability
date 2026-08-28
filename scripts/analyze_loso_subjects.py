"""Analyze per-subject LOSO baseline variation across feature/model settings."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR


STAT_LR_LOSO = RESULTS_DIR / "loso_baseline.csv"
STAT_SVM_LOSO = RESULTS_DIR / "loso_svm_baseline.csv"
BANDPOWER_LOSO = RESULTS_DIR / "loso_bandpower_baselines.csv"
STAT_LR_SUBJECT = RESULTS_DIR / "subject_dependent_baseline.csv"
STAT_SVM_SUBJECT = RESULTS_DIR / "subject_dependent_svm_baseline.csv"
BANDPOWER_SUBJECT = RESULTS_DIR / "subject_dependent_bandpower_baselines.csv"

SUBJECT_ANALYSIS_OUTPUT = RESULTS_DIR / "loso_subject_analysis.csv"
PROTOCOL_GAP_OUTPUT = RESULTS_DIR / "protocol_gap_summary.csv"
NOTES_OUTPUT = RESULTS_DIR / "loso_failure_analysis.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def std(values: list[float]) -> float:
    return statistics.pstdev(values)


def load_loso_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for row in read_csv(STAT_LR_LOSO):
        rows.append(
            {
                "feature_set": "statistical",
                "model_name": "LogisticRegression",
                "subject": row["subject"],
                "test_windows": int(row["test_windows"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )

    for row in read_csv(STAT_SVM_LOSO):
        rows.append(
            {
                "feature_set": "statistical",
                "model_name": "LinearSVM",
                "subject": row["subject"],
                "test_windows": int(row["test_windows"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )

    for row in read_csv(BANDPOWER_LOSO):
        rows.append(
            {
                "feature_set": "bandpower",
                "model_name": row["model_name"],
                "subject": row["subject"],
                "test_windows": int(row["test_windows"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )

    return rows


def load_subject_dependent_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    stat_lr = read_csv(STAT_LR_SUBJECT)[0]
    rows.append(
        {
            "feature_set": "statistical",
            "model_name": "LogisticRegression",
            "accuracy": float(stat_lr["accuracy"]),
            "macro_f1": float(stat_lr["macro_f1"]),
        }
    )

    stat_svm = read_csv(STAT_SVM_SUBJECT)[0]
    rows.append(
        {
            "feature_set": "statistical",
            "model_name": "LinearSVM",
            "accuracy": float(stat_svm["accuracy"]),
            "macro_f1": float(stat_svm["macro_f1"]),
        }
    )

    for row in read_csv(BANDPOWER_SUBJECT):
        rows.append(
            {
                "feature_set": "bandpower",
                "model_name": row["model_name"],
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )

    return rows


def add_ranks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["feature_set"]), str(row["model_name"]))].append(row)

    ranked_rows: list[dict[str, object]] = []
    for (_, _), group_rows in grouped.items():
        ranked = sorted(group_rows, key=lambda row: (float(row["macro_f1"]), float(row["accuracy"])))
        for rank, row in enumerate(ranked, start=1):
            ranked_rows.append(
                {
                    **row,
                    "difficulty_rank": rank,
                    "difficulty_rank_note": "1=lowest macro_f1 within feature/model setting",
                }
            )

    return sorted(
        ranked_rows,
        key=lambda row: (
            str(row["feature_set"]),
            str(row["model_name"]),
            int(row["difficulty_rank"]),
        ),
    )


def build_protocol_gap_rows(loso_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    subject_rows = load_subject_dependent_rows()
    loso_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in loso_rows:
        loso_grouped[(str(row["feature_set"]), str(row["model_name"]))].append(row)

    gap_rows: list[dict[str, object]] = []
    for subject_row in subject_rows:
        key = (str(subject_row["feature_set"]), str(subject_row["model_name"]))
        folds = loso_grouped[key]
        loso_acc = mean([float(row["accuracy"]) for row in folds])
        loso_f1 = mean([float(row["macro_f1"]) for row in folds])
        subject_acc = float(subject_row["accuracy"])
        subject_f1 = float(subject_row["macro_f1"])
        gap_rows.append(
            {
                "feature_set": key[0],
                "model_name": key[1],
                "subject_dependent_accuracy": f"{subject_acc:.6f}",
                "loso_mean_accuracy": f"{loso_acc:.6f}",
                "accuracy_protocol_drop": f"{subject_acc - loso_acc:.6f}",
                "subject_dependent_macro_f1": f"{subject_f1:.6f}",
                "loso_mean_macro_f1": f"{loso_f1:.6f}",
                "macro_f1_protocol_drop": f"{subject_f1 - loso_f1:.6f}",
                "loso_accuracy_std": f"{std([float(row['accuracy']) for row in folds]):.6f}",
                "loso_macro_f1_std": f"{std([float(row['macro_f1']) for row in folds]):.6f}",
            }
        )

    return gap_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_subjects(rows: list[dict[str, object]], n: int = 5) -> str:
    lines = []
    for row in rows[:n]:
        lines.append(
            f"- {row['subject']} ({row['feature_set']}, {row['model_name']}): "
            f"accuracy={float(row['accuracy']):.3f}, macro-F1={float(row['macro_f1']):.3f}"
        )
    return "\n".join(lines)


def write_notes(path: Path, ranked_rows: list[dict[str, object]], gap_rows: list[dict[str, object]]) -> None:
    hardest_bandpower_lr = [
        row
        for row in ranked_rows
        if row["feature_set"] == "bandpower" and row["model_name"] == "LogisticRegression"
    ]
    best_bandpower_lr = sorted(hardest_bandpower_lr, key=lambda row: float(row["macro_f1"]), reverse=True)

    lines = [
        "# LOSO Subject Failure Analysis",
        "",
        "This analysis summarizes per-subject leave-one-subject-out variation across the current GAMEEMO baselines.",
        "",
        "## Hardest Held-Out Subjects",
        "",
        "Using bandpower + logistic regression, the lowest macro-F1 subjects are:",
        "",
        format_subjects(hardest_bandpower_lr),
        "",
        "## Strongest Held-Out Subjects",
        "",
        "Using bandpower + logistic regression, the highest macro-F1 subjects are:",
        "",
        format_subjects(best_bandpower_lr),
        "",
        "## Protocol Gap",
        "",
        "| Feature Set | Model | Subject-Dependent Macro-F1 | LOSO Mean Macro-F1 | Drop | LOSO Macro-F1 Std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in gap_rows:
        lines.append(
            f"| {row['feature_set']} | {row['model_name']} | "
            f"{row['subject_dependent_macro_f1']} | {row['loso_mean_macro_f1']} | "
            f"{row['macro_f1_protocol_drop']} | {row['loso_macro_f1_std']} |"
        )

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "- LOSO performance varies substantially by held-out subject, so aggregate scores hide subject-level failure modes.",
            "- Bandpower improves the overall baseline, but several held-out subjects remain difficult.",
            "- The next analysis target is to inspect whether difficult subjects share label confusions, signal-quality issues, or preprocessing differences.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    loso_rows = load_loso_rows()
    ranked_rows = add_ranks(loso_rows)
    gap_rows = build_protocol_gap_rows(loso_rows)

    write_csv(SUBJECT_ANALYSIS_OUTPUT, ranked_rows)
    write_csv(PROTOCOL_GAP_OUTPUT, gap_rows)
    write_notes(NOTES_OUTPUT, ranked_rows, gap_rows)

    print(f"wrote: {SUBJECT_ANALYSIS_OUTPUT}")
    print(f"wrote: {PROTOCOL_GAP_OUTPUT}")
    print(f"wrote: {NOTES_OUTPUT}")


if __name__ == "__main__":
    main()
