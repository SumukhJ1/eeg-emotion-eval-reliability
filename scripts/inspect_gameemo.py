"""Inspect local GAMEEMO files and summarize subject/condition structure.

Update GAMEEMO_ROOT below or pass a path once the data is copied/linked into data/external.
"""

from pathlib import Path

GAMEEMO_ROOT = Path("data/external/GAMEEMO")


def main() -> None:
    root = GAMEEMO_ROOT
    if not root.exists():
        print(f"GAMEEMO root not found: {root.resolve()}")
        print("Place or link the dataset at data/external/GAMEEMO, then rerun.")
        return

    subject_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith("S")])
    print(f"Subjects found: {len(subject_dirs)}")

    for subject in subject_dirs[:5]:
        files = [p for p in subject.rglob("*") if p.is_file()]
        suffixes = sorted(set(p.suffix.lower() for p in files))
        print(f"{subject.name}: {len(files)} files, suffixes={suffixes}")

    if len(subject_dirs) > 5:
        print(f"... {len(subject_dirs) - 5} more subjects not shown")


if __name__ == "__main__":
    main()
