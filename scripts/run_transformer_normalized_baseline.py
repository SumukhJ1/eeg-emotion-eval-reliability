"""Run the best current train-normalized Transformer setting on GAMEEMO."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "transformer_normalized_baseline.csv"


def main() -> None:
    command = [
        sys.executable,
        "-u",
        "-B",
        str(PROJECT_ROOT / "scripts" / "run_subject_dependent_transformer_baseline.py"),
        "--output",
        str(OUTPUT_PATH),
        "--epochs",
        "30",
        "--batch-size",
        "64",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--patience",
        "8",
        "--dropout",
        "0.1",
        "--d-model",
        "64",
        "--n-heads",
        "4",
        "--n-layers",
        "2",
        "--dim-feedforward",
        "128",
        "--input-mode",
        "channel",
        "--patch-samples",
        "32",
        "--device",
        "cpu",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
