# EEG Walk/Stop BCI Pipeline

An offline BCI pipeline that classifies Walk/Stop command periods from C3/C4/Cz
EEG recordings, controlling for EOG/movement artifact effects. The pipeline
uses CSP-based feature extraction, F-score feature selection, a
shrinkage-regularized LDA classifier, and LOOCV / cross-session validation;
it ultimately produces a symbolic joystick command (WALK/STOP/IDLE) from the
held-out predictions.

**Scope note:** This is an offline validation and symbolic command generation
system. It does not provide real-time EEG streaming or actual HID/vJoy
control — every "joystick command" produced is a label derived from held-out
(LOOCV) predictions on a fixed recording, written to the terminal/CSV.

## Main file
- `bci_pipeline.py` — current version (v3.6): a single file, CLI-enabled, containing the entire pipeline.

## Historical / dependency files (used in early versions)
- `modern_bci_v2.py` — the core signal processing engine (the first versions had a modular
  structure where `bci_pipeline.py` imported this; merged into a single file starting from v1.1).
- `edf_reader.py`, `parse_events.py`, `validate_full.py` — the initial real-data validation script.

## Early attempts (abandoned)
- `production_grade_bci.py`, `realtime_eeg_motor_control.py`, `joystick_output.py` —
  the first approach, real-time, multi-threaded, based on vJoy/LSL. Due to its
  complexity, this was pivoted to the offline validation + symbolic command
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
