# EEG Walk/Stop BCI Pipeline

An offline BCI pipeline that classifies Walk/Stop command periods from C3/C4/Cz
EEG recordings (you can choose more channels in the last versions), controlling for EOG/movement artifact effects. The pipeline
uses CSP-based feature extraction, F-score feature selection, a
shrinkage-regularized LDA classifier, and LOOCV / cross-session validation;
it ultimately produces a symbolic joystick command (WALK/STOP/IDLE) from the
held-out predictions. Focus is producing real commands, based on Predictions.

This repository documents the complete research process from early prototypes to the current pipeline.

**Scope note:** This is an offline validation and symbolic command generation
system. It does not provide real-time EEG streaming or actual HID/vJoy
control — every "joystick command" produced is a label derived from held-out
(LOOCV) predictions on a fixed recording, written to the terminal/CSV.

EEG (.edf)
      │
      ▼
Preprocessing
      │
      ▼
CSP
      │
      ▼
Feature Selection
      │
      ▼
Shrinkage LDA
      │
      ▼
Sliding Window
      │
      ▼
Temporal Smoothing
      │
      ▼
WALK / STOP / IDLE

## Currents Best Resluts

| Evaluation                | Configuration        |   Accuracy | Balanced Acc. |
| -----------------------   | -------------------- | ---------: | ------------: |
| **Seen session**          | Best configuration   | **90.00%** |    **88.10%** |
| **Cross-session (1-4→5)** | Motor3 + z-score     | **71.73%** |    **71.59%** |
| **Cross-session (1-7→8)** | Motor13 (no z-score) | **80.20%** |    **75.87%** |


## Main file
- `bci_pipeline.py` — current version (v2.8): a single file, CLI-enabled, containing the entire pipeline.

## Historical / dependency files (used in early versions)
- `modern_bci_v2.py` — the core signal processing engine (the first versions had a modular
  structure where `bci_pipeline.py` imported this; merged into a single file starting from v1.1).
- `edf_reader.py`, `parse_events.py`, `validate_full.py` — the initial real-data validation script.

## Early attempts (abandoned)
- `production_grade_bci.py`, `realtime_eeg_motor_control.py`, `joystick_output.py` —
  the first approach, real-time, multi-threaded, based on vJoy/LSL. Due to its
  complexity, this was postponed to future updates, and pivoted to the offline validation + symbolic command
  approach (see CHANGELOG).

## Usage (current version)
```bash
python bci_pipeline.py \
    --edf sub-01_ses-01_task-training_eeg.edf \
    --events sub-01_ses-01_task-training_acq-rexcommand_events.tsv \
    --output-dir results
```

See `CHANGELOG.md` for version history; each version exists as a separate git
commit (`git log --oneline`).

## Data
Raw EEG recordings (`.edf`) and trained model files (`.npz`, `.npy`) are kept
out of the repo via `.gitignore` (file size + subject data privacy).
