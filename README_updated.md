# EEG Walk/Stop BCI Pipeline

An offline EEG-based Brain–Computer Interface pipeline for classifying **WALK** and **STOP** command periods and producing symbolic **WALK / STOP / IDLE** outputs.

The pipeline supports configurable motor-cortex channel sets, per-channel normalization, CSP-based feature extraction, F-score feature selection, shrinkage-regularized LDA classification, full-record sliding-window validation, temporal smoothing, single-session training, and pooled multi-session or multi-subject training.

A local browser-based Web UI is included so the pipeline can be configured and started without manually writing terminal commands.

> **Scope note**
>
> This is an offline validation and symbolic command-generation system. It does not currently provide real-time EEG streaming or direct HID/vJoy control. Every command is generated from predictions on a fixed EEG recording and is written to output files and/or the terminal.

---

## Pipeline overview

```text
EEG recording (.edf)
        │
        ▼
Per-channel normalization
(none or z-score)
        │
        ▼
Preprocessing
        │
        ▼
CSP
        │
        ▼
Feature extraction
        │
        ▼
F-score feature selection
        │
        ▼
Shrinkage LDA
        │
        ▼
Full-record sliding windows
        │
        ▼
Confidence / IDLE gating
        │
        ▼
Temporal smoothing
        │
        ▼
WALK / STOP / IDLE
```

---

## Current best results

| Evaluation | Configuration | Accuracy | Balanced accuracy |
|---|---|---:|---:|
| **Seen session** | Best configuration | **90.00%** | **88.10%** |
| **Cross-session: sessions 1–4 → 5** | Motor3 + z-score | **71.73%** | **71.59%** |
| **Cross-session: sessions 1–7 → 8** | Motor13, no z-score | **80.20%** | **75.87%** |

Cross-session results refer to training and testing on different recording sessions. They should not be described as cross-subject results unless the held-out recording belongs to a subject who was not included during training.

---

## Main files

```text
bci_pipeline.py
bci_web_ui.py
modern_bci_v2.py
edf_reader.py
parse_events.py
CHANGELOG.md
```

- `bci_pipeline.py` — main CLI entry point for training, multi-session training, timeline validation, and unlabeled prediction.
- `bci_web_ui.py` — local browser interface that builds and runs the same CLI commands.
- `modern_bci_v2.py` — signal-processing and classification components.
- `edf_reader.py` — EDF loading utilities.
- `parse_events.py` — event-file parser.
- `CHANGELOG.md` — version history and experiment-related changes.

Some versions use the filename `bci_pipeline_v2.8.py`. Select the correct file in the Web UI's **Pipeline script path** field.

---

# Web UI

## What the Web UI does

The Web UI provides controls for:

- Single-session training
- Multi-session or multi-subject training
- Timeline validation with ground-truth events
- Unlabeled full-record prediction
- Channel-set selection
- Raw-channel normalization selection
- Feature-count selection
- Confidence and IDLE thresholds
- Class and subject balancing
- Temporal smoothing
- Percentage-based result visualization
- Experiment summary display
- Raw and smoothed confusion matrices
- Ground-truth / raw / smoothed prediction timeline
- Output-file downloads
- Dataset-list generation for multi-session training

The Web UI does not replace or modify the machine-learning pipeline. It starts the existing Python pipeline as a local child process using the selected settings.

EEG data is processed locally and is not uploaded by the interface.

---

## Start the Web UI

Keep the UI and pipeline files in the same project folder:

```text
project/
├── bci_web_ui.py
├── bci_pipeline.py
├── modern_bci_v2.py
├── edf_reader.py
├── parse_events.py
└── ...
```

Open PowerShell or a terminal in that folder.

### Windows

```powershell
py -X utf8 bci_web_ui.py
```

### Linux / macOS

```bash
python3 bci_web_ui.py
```

Then open:

```text
http://127.0.0.1:8765
```

To stop the server, return to the terminal and press:

```text
Ctrl+C
```

Restart the server after replacing or editing `bci_web_ui.py`.

---

## Web UI modes

| Mode | Purpose | Required inputs |
|---|---|---|
| **Train** | Train one model from one labeled recording | EDF, events, output directory |
| **Train multi-session** | Train one pooled model from multiple recording rows | Dataset-list file, output directory |
| **Validate timeline** | Evaluate a saved model on a full labeled recording | EDF, events, model directory, output directory |
| **Predict unlabeled timeline** | Run a saved model without ground-truth events | EDF, model directory, output directory |

---

# Single-session training

In the Web UI:

1. Select **Train**.
2. Select the pipeline script.
3. Enter the labeled training EDF path.
4. Enter the corresponding events-file path.
5. Choose an output directory for the model.
6. Select the channel set.
7. Select `zscore` or `none` for channel normalization.
8. Set the selected feature count, confidence threshold, balancing option, and other parameters.
9. Press **Start pipeline**.

Typical settings:

```text
Channel set: motor3 / motor5 / motor9 / motor13
Channel normalization: zscore
Selected feature count: 45
Confidence threshold: 0.45
IDLE distance threshold: 999
```

Equivalent CLI example:

```powershell
py -X utf8 bci_pipeline.py `
  --mode train `
  --edf "D:\BCI\data\sub-02_ses-01_task-training_eeg.edf" `
  --events "D:\BCI\data\sub-02_ses-01_task-training_events.tsv" `
  --output-dir "D:\BCI\results\sub-02_ses-01_model" `
  --channel-set motor9 `
  --channel-normalization zscore `
  --n-features-select 45 `
  --confidence-threshold 0.45
```

A trained model directory normally contains:

```text
trained_model.npz
model_info.json
selected_features.json
training_summary.txt
```

---

# Multi-session training

Multi-session training uses a dataset-list file containing one row for every EDF/events pair.

The required columns are exactly:

```text
subject,session,edf,events
```

## Supported dataset-list filenames

The recommended filename is:

```text
train_list.csv
```

However, this also works:

```text
train_list.txt
```

> **Note:** The filename extension is not important. The pipeline reads the file as comma-separated data. A `.txt` file works as long as it contains CSV-formatted content and the exact header `subject,session,edf,events`.

For example, both of these are valid:

```text
D:\BCI\train_list.csv
D:\BCI\train_list.txt
```

---

## 1. Create the dataset-list file manually

### Example using absolute Windows paths

```csv
subject,session,edf,events
sub-01,ses-01,D:\BCI\data\sub-01_ses-01_task-training_eeg.edf,D:\BCI\data\sub-01_ses-01_task-training_events.tsv
sub-01,ses-02,D:\BCI\data\sub-01_ses-02_task-training_eeg.edf,D:\BCI\data\sub-01_ses-02_task-training_events.tsv
sub-02,ses-01,D:\BCI\data\sub-02_ses-01_task-training_eeg.edf,D:\BCI\data\sub-02_ses-01_task-training_events.tsv
```

The `subject` and `session` values preserve provenance and are used for subject-level summaries and optional subject balancing.

Each row must point to a matching EEG recording and events file.

### Example using relative paths

Project layout:

```text
D:\BCI\
├── bci_pipeline.py
├── bci_web_ui.py
├── train_list.csv
└── data\
    ├── sub-01_ses-01_eeg.edf
    ├── sub-01_ses-01_events.tsv
    ├── sub-01_ses-02_eeg.edf
    └── sub-01_ses-02_events.tsv
```

Dataset-list content:

```csv
subject,session,edf,events
sub-01,ses-01,data\sub-01_ses-01_eeg.edf,data\sub-01_ses-01_events.tsv
sub-01,ses-02,data\sub-01_ses-02_eeg.edf,data\sub-01_ses-02_events.tsv
```

Set **Dataset base directory** to:

```text
D:\BCI
```

The pipeline combines the base directory with relative paths that cannot already be resolved.

---

## 2. Create the dataset list in the Web UI

The Web UI includes a **Dataset-list generator**.

1. Open the **Train multi-session** mode.
2. Open the dataset-list generator section.
3. Add one row per recording.
4. Enter:
   - Subject
   - Session
   - EDF path
   - Events path
5. Add more rows as needed.
6. Remove incorrect rows with the row delete control.
7. Export the generated file.
8. Save it as `train_list.csv` or `train_list.txt`.
9. Use the saved path in **Dataset list**.

Example rows:

| Subject | Session | EDF | Events |
|---|---|---|---|
| sub-01 | ses-01 | `D:\BCI\data\sub-01_ses-01_eeg.edf` | `D:\BCI\data\sub-01_ses-01_events.tsv` |
| sub-01 | ses-02 | `D:\BCI\data\sub-01_ses-02_eeg.edf` | `D:\BCI\data\sub-01_ses-02_events.tsv` |
| sub-02 | ses-01 | `D:\BCI\data\sub-02_ses-01_eeg.edf` | `D:\BCI\data\sub-02_ses-01_events.tsv` |

---

## 3. Create the file in Notepad

1. Open Notepad.
2. Paste the header and session rows.
3. Select **File → Save As**.
4. Use either:

```text
train_list.csv
```

or:

```text
train_list.txt
```

5. Set **Save as type** to:

```text
All files (*.*)
```

6. Select UTF-8 encoding.
7. Confirm that Windows did not append another extension such as `.txt`.

PowerShell check:

```powershell
Get-Content "D:\BCI\train_list.txt" -First 10
```

Expected beginning:

```text
subject,session,edf,events
sub-01,ses-01,...
```

---

## 4. Create the file in Excel

Create four columns:

| subject | session | edf | events |
|---|---|---|---|
| sub-01 | ses-01 | `D:\BCI\data\sub-01_ses-01_eeg.edf` | `D:\BCI\data\sub-01_ses-01_events.tsv` |
| sub-01 | ses-02 | `D:\BCI\data\sub-01_ses-02_eeg.edf` | `D:\BCI\data\sub-01_ses-02_events.tsv` |

Save as:

```text
CSV UTF-8 (Comma delimited) (*.csv)
```

Do not save it as an `.xlsx` workbook.

---

## 5. Start multi-session training from the Web UI

1. Select **Train multi-session**.
2. Enter the `train_list.csv` or `train_list.txt` path in **Dataset list**.
3. Set **Dataset base directory**:
   - Use `.` when every path in the list is absolute.
   - Use the parent dataset folder when the list contains relative paths.
4. Choose the channel set.
5. Choose channel normalization.
6. Select class balancing and subject balancing if required.
7. Choose the model output directory.
8. Press **Start pipeline**.

Recommended first run:

```text
Channel set: motor3
Channel normalization: zscore
Class balancing: none
Subject balancing: none
Selected feature count: 45
Seed: 42
```

Use different output directories when comparing configurations.

---

## Multi-session CLI example

Using `.csv`:

```powershell
py -X utf8 bci_pipeline.py `
  --mode train_multi `
  --dataset-list "D:\BCI\train_list.csv" `
  --dataset-dir "D:\BCI" `
  --output-dir "D:\BCI\results\multi_model" `
  --channel-set motor9 `
  --channel-normalization zscore `
  --n-features-select 45 `
  --confidence-threshold 0.45 `
  --balance-classes none `
  --balance-subjects none `
  --seed 42
```

Using `.txt`:

```powershell
py -X utf8 bci_pipeline.py `
  --mode train_multi `
  --dataset-list "D:\BCI\train_list.txt" `
  --dataset-dir "D:\BCI" `
  --output-dir "D:\BCI\results\multi_model" `
  --channel-set motor9 `
  --channel-normalization zscore
```

### Balancing options

Class balancing:

```text
none
downsample
```

Subject balancing:

```text
none
downsample
```

`class_weight` may appear as a CLI option in some versions, but it is not implemented in v2.8. Use `none` or `downsample`.

---

## How multi-session pooling works

Continuous EEG recordings are **not concatenated** across sessions.

Instead:

1. Every recording is loaded independently.
2. Per-channel normalization is calculated independently for that recording.
3. Every recording is preprocessed independently.
4. Labeled WALK/STOP command windows are extracted independently.
5. The already-epoched trials are pooled.
6. One CSP + feature-selection + LDA model is trained from the pooled trials.

This avoids continuous-signal boundaries and normalization statistics leaking across separate recordings.

---

## Multi-session requirements

For a row to be used:

- The EDF file must exist.
- The events file must exist and be readable by `parse_events.py`.
- The events file must contain usable `x5` and/or `x8` labels.
- Required channels for the selected channel set must exist.
- Sampling rate must match the other recordings used in the same model.

Current command labels:

```text
x5 → STOP
x8 → WALK
```

A row that cannot be used is skipped with an explanation in the terminal output.

Multi-session output additionally includes:

```text
train_manifest_resolved.csv
train_manifest_used.csv
training_summary.txt
```

- `train_manifest_resolved.csv` contains candidate windows before balancing.
- `train_manifest_used.csv` contains the windows that actually entered training.
- `training_summary.txt` records used files, skipped files, class counts, subject distribution, and settings.

---

# Timeline validation

Use **Validate timeline** to test a trained model on a complete recording with ground-truth events.

In the Web UI:

1. Select **Validate timeline**.
2. Select the test EDF.
3. Select the matching events file.
4. Select the trained model directory.
5. Choose a separate output directory.
6. Select smoothing window `1`, `3`, or `5`.
7. Press **Start pipeline**.

Example:

```powershell
py -X utf8 bci_pipeline.py `
  --mode validate_timeline `
  --edf "D:\BCI\data\sub-02_ses-02_task-training_eeg.edf" `
  --events "D:\BCI\data\sub-02_ses-02_task-training_events.tsv" `
  --model "D:\BCI\results\sub-02_ses-01_model" `
  --output-dir "D:\BCI\results\ses-01_to_ses-02_validation" `
  --event-overlap-threshold 0.5 `
  --smoothing-window 3
```

Typical validation outputs:

```text
validated_timeline.csv
timeline_metrics.json
timeline_confusion_matrix_raw.csv
timeline_confusion_matrix_smoothed.csv
timeline_confusion_matrix.csv
collapse_report.txt
```

Important metrics include:

- Deployment accuracy on non-IDLE ground-truth windows
- Balanced accuracy
- WALK recall
- STOP recall
- IDLE false-positive rate
- Raw and smoothed prediction distributions
- Collapse warning
- STOP/WALK/IDLE confusion matrices

---

# Unlabeled prediction

Use **Predict unlabeled timeline** when no ground-truth event file is available.

```powershell
py -X utf8 bci_pipeline.py `
  --mode predict `
  --edf "D:\BCI\data\unlabeled_recording.edf" `
  --model "D:\BCI\results\multi_model" `
  --output-dir "D:\BCI\results\unlabeled_prediction" `
  --smoothing-window 3
```

Typical outputs:

```text
predicted_timeline.csv
prediction_summary.json
collapse_report.txt
```

Accuracy cannot be calculated without ground truth.

---

# Channel sets

Available presets:

```text
motor3
motor5
motor9
motor13
all_eeg
```

The selected channel set is used during training and stored in model metadata. Validation and prediction use the channels saved in the trained model.

Changing the channel set requires training a new model because the CSP and feature spaces change.

---

# Channel normalization

Available settings:

```text
none
zscore
```

With `zscore`, every selected raw EEG channel is normalized independently for the current recording before bandpass filtering and CSP:

```text
x_normalized = (x - channel_mean) / channel_std
```

The normalization mode is stored with the model and reused during validation and prediction.

Changing normalization requires training a new model.

---

# Recommended experiment organization

Use a separate output directory for every configuration:

```text
results/
├── motor3_zscore_model/
├── motor9_zscore_model/
├── motor13_none_model/
├── ses01_to_ses02_validation/
└── unlabeled_prediction/
```

Do not overwrite the same model directory while comparing configurations.

---

# Troubleshooting

## `ModuleNotFoundError: No module named 'parse_events'`

Keep the required modules in the same project folder:

```text
bci_pipeline.py
parse_events.py
edf_reader.py
modern_bci_v2.py
```

## `No /Root object! - Is this really a PDF?`

The parser is trying to open a non-PDF file with a PDF reader.

For a TSV events file, confirm:

```powershell
Get-Content "D:\BCI\data\events.tsv" -First 5
```

Expected content:

```text
onset  duration  trial_type
0      59        x5
59     12.5      x8
```

Make sure `parse_events.py` detects `.tsv` and reads it as text instead of always calling a PDF parser.

## `No usable x5/x8 events`

Confirm that:

- The dataset list points to the intended events file.
- The file contains a `trial_type` column.
- Labels are `x5` and `x8`, or are normalized by the parser.
- The events path does not point to an HTML page, screenshot, or unrelated session.

## Unicode or `cp1252` errors on Windows

Start the UI with UTF-8 enabled:

```powershell
py -X utf8 bci_web_ui.py
```

Stop the old server with `Ctrl+C` before restarting it.

## A row is skipped during multi-session training

Read the `[!] SKIPPED` or equivalent terminal line. Common reasons:

- EDF file not found
- Events file not found
- Events parsing failed
- No usable x5/x8 events
- Missing channel
- Sampling-rate mismatch

---

# Evaluation terminology

Use precise wording when reporting results:

- **Seen-session validation:** the evaluation recording was represented during model development or training.
- **Cross-session validation:** training and test recordings come from different sessions.
- **Cross-subject validation:** the test subject was completely excluded from training.
- **Target-session calibrated performance:** the target session was used to select preprocessing or model settings.

Selecting the best channel set or normalization by repeatedly testing on the same target session makes that session part of configuration selection. A separate untouched session is required for an unbiased final test.

---

# Scientific interpretation

The model predicts command periods from EEG recordings under conditions represented in the training data.

Performance may depend on:

- Session-to-session amplitude changes
- Electrode placement
- Subject identity
- Movement and EOG artifacts
- Selected channels
- Channel normalization
- Class balance
- Window length and step size
- Confidence and IDLE thresholds
- Temporal smoothing

The results should not be interpreted as proof of artifact-free, purely brain-internal movement-intention decoding without additional controls.

---

# Historical versions

Earlier experiments included:

- `production_grade_bci.py`
- `realtime_eeg_motor_control.py`
- `joystick_output.py`

Those versions explored real-time, multi-threaded LSL/vJoy control. Development later shifted toward offline validation and symbolic command generation. See `CHANGELOG.md` and Git history for the complete research process.

---

# License and citation

Add the repository license and preferred citation information here.
