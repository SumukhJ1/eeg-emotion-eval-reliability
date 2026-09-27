# GAMEEMO dataset metadata

Based on `python scripts/inspect_gameemo.py` run against the local GAMEEMO dataset root. The dataset root is outside the repo and is not committed.

```text
<path-to-GAMEEMO-root>
```

No dataset files are copied into the repo.

## Dataset layout

- Subject folders: 28 top-level folders, named `(S01)` through `(S28)`.
- Each subject folder contains:
  - `Raw EEG Data/.csv format/`
  - `Raw EEG Data/.mat format/`
  - `Preprocessed EEG Data/.csv format/`
  - `Preprocessed EEG Data/.mat format/`
  - `SAM Ratings/`
- The dataset root also contains `Gameplays/` with MP4 gameplay videos.

## File counts

| Scope | CSV | MAT | PDF | MP4 |
| --- | ---: | ---: | ---: | ---: |
| Raw EEG files | 112 | 112 | 0 | 0 |
| Preprocessed EEG files | 112 | 112 | 0 | 0 |
| All dataset files | 232 | 232 | 116 | 4 |

## EEG channels

The inspected preprocessed CSV header contains 14 channels:

```text
AF3, AF4, F3, F4, F7, F8, FC5, FC6, O1, O2, P7, P8, T7, T8
```

## Sampling and shape

- Sampling rate: 128 Hz.
- Sample inspected recording:
  - Path: `...\(S01)\Preprocessed EEG Data\.csv format\S01G1AllChannels.csv`
  - Shape: 38,252 samples x 14 channels.
  - Approximate duration: 298.84 seconds at 128 Hz.
- Planned 2-second window shape:
  - 256 samples x 14 channels.

## Raw vs preprocessed organization

For each subject, raw and preprocessed EEG are stored separately. Each representation is available in both CSV and MAT formats, with one file per game condition. The first analysis pass should use the preprocessed CSV files because they can be inspected with the Python standard library and already expose a clean 14-channel EEG matrix.
