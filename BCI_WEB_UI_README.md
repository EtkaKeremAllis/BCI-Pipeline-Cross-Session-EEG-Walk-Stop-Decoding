# BCI Pipeline Web UI v2

A dependency-free local browser interface for `bci_pipeline_v2.8.py`.
It does **not** change the CSP → feature extraction → feature selection → shrinkage LDA pipeline. It only runs the existing CLI modes and visualizes their generated result files.

## New research-dashboard features

- **Experiment summary strip** showing the training file names, test file names, channel set, channel normalization, smoothing window, selected feature count, and confidence threshold.
- **Raw and smoothed 3×3 confusion matrices** for STOP, WALK, and IDLE.
- **Prediction timeline** with Ground truth, Raw prediction, and Smoothed prediction lanes.
- **One-click exports** for metrics JSON, timeline CSV, report TXT, and a combined figure PNG.
- **Dataset-list generator** for multi-session training. Add subject/session/EDF/events rows in the browser; the UI creates `train_sessions.csv` automatically before training.
- **Windows UTF-8 protection** using `-X utf8`, `PYTHONUTF8=1`, and `PYTHONIOENCODING=utf-8`.

## Required folder layout

Keep these files together:

```text
bci_web_ui.py
bci_pipeline_v2.8.py
edf_reader.py
parse_events.py
modern_bci_v2.py
```

The pipeline filename `bci_pipeline_v2.8 (1).py` is also detected automatically.

## Start the UI

Open PowerShell in the folder and run:

```powershell
py -X utf8 bci_web_ui.py
```

Open:

```text
http://127.0.0.1:8765
```

Stop the current server with `Ctrl+C` before replacing or restarting `bci_web_ui.py`.

## Dataset-list generator

Choose **Train multi-session**, then add one row per recording:

```text
Subject | Session | EDF path | Events path
```

The default generated file is:

```text
train_sessions.csv
```

Press **Generate CSV now**, or simply press **Start pipeline**. When session rows are present, the UI generates the CSV automatically and places its path in the Dataset list CSV field.

## Experiment summary source

For validation and prediction, training filenames are read from the selected model directory:

1. `train_manifest_used.csv`, when available.
2. `training_summary.txt`, as a fallback.

This lets screenshots show which training documents produced the displayed test result.

## Output visualization requirements

Confusion matrices and the three-lane Ground truth / Raw / Smoothed timeline require a labeled `validate_timeline` run that generates:

```text
timeline_metrics.json
validated_timeline.csv
timeline_confusion_matrix_raw.csv
timeline_confusion_matrix_smoothed.csv
```

Unlabeled prediction still shows Raw and Smoothed timeline lanes, but no ground-truth lane or accuracy matrix.
