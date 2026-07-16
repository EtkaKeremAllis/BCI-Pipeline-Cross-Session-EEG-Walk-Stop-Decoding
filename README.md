# EEG Walk/Stop BCI Pipeline

An offline EEG-based brain-computer interface (BCI) pipeline for classifying **WALK** and **STOP** command periods and producing symbolic **WALK / STOP / IDLE** outputs.

The current implementation is `bci_pipeline_v2.9.py`. It supports single-session and pooled multi-session training, full-record timeline validation, and prediction on unlabeled EDF recordings. A dependency-free local Web UI is also included.

## Experimental results

See [v2.9_verified_cross_session_results/VERIFIED_RESULTS.md](v2.9_verified_cross_session_results/VERIFIED_RESULTS.md) for the reported reciprocal cross-session benchmark, exact reproduction commands, environment details, machine-readable outputs, and scientific limitations.

> [!IMPORTANT]
> This repository implements an offline research pipeline. It does not provide real-time EEG streaming, HID control, or vJoy output. Predictions are generated from fixed EEG recordings and saved to files.

## Scientific scope

The pipeline predicts command periods under conditions represented in the training data. Its outputs must not be interpreted as proof of artifact-free or purely brain-internal walking-intention decoding. Performance can be affected by subject identity, session variability, electrode placement, movement and EOG artifacts, preprocessing, class balance, and model-selection choices.

**EOG artifact caveat (quantified):** see [EOG_ABLATION.md](EOG_ABLATION.md). An ablation run through this pipeline found EOG-only classification accuracy (73.47% mean balanced accuracy, `sub-02` reciprocal cross-session) far above chance, only ~2 points below EEG-only (75.59%), with EEG+EOG barely improving on EEG-only (76.91%). This confirms rather than resolves the original artifact-contamination suspicion — a meaningful share of reported EEG-only accuracy is plausibly attributable to eye-movement artifacts correlated with the WALK/STOP task structure, not solely to motor-intent signal.

Use the following terminology when reporting results:

- **Seen-session validation:** the evaluated recording was represented during training or model development.
- **Cross-session validation:** training and test recordings come from different sessions.
- **Cross-subject validation:** the test subject is completely excluded from training and model selection.
- **Target-session calibrated performance:** the target session influenced preprocessing, threshold, channel, or model selection.

Repeatedly selecting settings on a target session makes that session part of model development. Use a separate untouched recording for an unbiased final evaluation.

## Pipeline

```text
EDF recording
    -> selected EEG channels
    -> optional per-recording channel z-score normalization
    -> signal preprocessing
    -> labeled command windows (training) or sliding windows (inference)
    -> CSP
    -> feature extraction
    -> F-score feature selection
    -> shrinkage LDA
    -> confidence and IDLE gating
    -> optional temporal smoothing
    -> WALK / STOP / IDLE
```

Continuous recordings are processed independently. In multi-session training, recordings are not concatenated; only extracted command-window trials are pooled.

## Repository files

| File | Purpose |
|---|---|
| `bci_pipeline_v2.9.py` | Main command-line pipeline |
| `bci_web_ui.py` | Local browser interface for the pipeline |
| `modern_bci_v2.py` | CSP, feature extraction, feature selection, and LDA components |
| `edf_reader.py` | EDF loading utilities |
| `parse_events.py` | BIDS-style TSV and legacy PDF event parsing |
| `CHANGELOG.md` | Development history |

## Installation and quick start

Python 3.10 or newer is expected to work. The currently verified environment
uses Python 3.13.14. Other Python versions have not yet been systematically tested.

See [`requirements.txt`](requirements.txt) and [`SETUP.md`](SETUP.md) for detailed
Python PATH, and PowerShell troubleshooting instructions.

### Windows PowerShell

Run these commands from the repository directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 bci_web_ui.py
```

If `py` is unavailable, use the full path to the installed `python.exe` when
creating `.venv`. Activation is optional.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -X utf8 bci_web_ui.py
```

Then open <http://127.0.0.1:8765> and stop the server with `Ctrl+C`.

## Testing

Unit tests (`tests/`) cover the core signal-processing and ML logic (CSP,
feature selection, LDA, sliding windows, temporal smoothing, the IDLE gate)
against synthetic data with known ground truth:

```bash
pytest tests/ -v
```

`tests/test_smoke_synthetic_pipeline.py` is an end-to-end smoke test: it
generates a synthetic EEG recording (`tests/synthetic_data.py`), writes it
to a real EDF file, and runs it through the actual `train` -> `validate_timeline`
CLI flow - no real recording required. It also acts as a regression guard
against the "model collapse" bug from `CHANGELOG.md` (v3.0): if training and
prediction ever fall out of distribution sync again, this test's
`collapse_warning` assertions will fail.

**Run the full test suite (`pytest tests/ -v`) after any change to
`bci_pipeline_v2.9.py`, `modern_bci_v2.py`, `edf_reader.py`, or
`parse_events.py`** - it catches structural regressions well before you'd
notice them in a real multi-minute training run on real data.

## Command-line interface

Show the complete interface:

```bash
python bci_pipeline_v2.9.py --help
```

The available modes are:

| Mode | Required inputs | Purpose |
|---|---|---|
| `train` | `--edf`, `--events`, `--output-dir` | Train from one labeled recording |
| `train_multi` | `--dataset-list`, `--output-dir` | Train from pooled labeled recordings |
| `validate_timeline` | `--edf`, `--model`, `--output-dir` | Run full-record inference; evaluate when `--events` is supplied |
| `predict` | `--edf`, `--model`, `--output-dir` | Predict a full recording without ground truth |

### Event labels

The default mapping is:

```text
x5 -> STOP (0)
x8 -> WALK (1)
```

Use `--label-map` for another dataset. It accepts either a JSON object or the path to a JSON file:

```powershell
--label-map '{"stop":0,"walk":1}'
```

The selected mapping is saved with the model and reused during validation and prediction.

## Single-session training

```powershell
python bci_pipeline_v2.9.py `
  --mode train `
  --edf "D:\BCI\data\session01_eeg.edf" `
  --events "D:\BCI\data\session01_events.tsv" `
  --output-dir "D:\BCI\results\session01_model" `
  --channel-set motor9 `
  --channel-normalization zscore `
  --n-features-select 45 `
  --confidence-threshold 0.45 `
  --balance-classes none `
  --seed 42
```

The event file must be a BIDS-style TSV with these columns:

```text
onset    duration    trial_type
```

Legacy PDF event exports are supported when `pdfplumber` is installed.

## Multi-session training

Create a comma-separated dataset list with exactly these columns:

```csv
subject,session,edf,events
sub-01,ses-01,data/sub-01_ses-01_eeg.edf,data/sub-01_ses-01_events.tsv
sub-01,ses-02,data/sub-01_ses-02_eeg.edf,data/sub-01_ses-02_events.tsv
sub-02,ses-01,data/sub-02_ses-01_eeg.edf,data/sub-02_ses-01_events.tsv
```

Relative paths are resolved against `--dataset-dir`:

```powershell
python bci_pipeline_v2.9.py `
  --mode train_multi `
  --dataset-list "D:\BCI\train_list.csv" `
  --dataset-dir "D:\BCI" `
  --output-dir "D:\BCI\results\pooled_model" `
  --channel-set motor9 `
  --channel-normalization zscore `
  --n-features-select 45 `
  --balance-classes downsample `
  --balance-subjects downsample `
  --seed 42
```

Each usable recording must contain the selected channels and use the same sampling rate. Invalid rows are skipped with an explanation.

Supported balancing settings:

- `--balance-classes none`
- `--balance-classes downsample`
- `--balance-subjects none`
- `--balance-subjects downsample`

## Timeline validation

Supply an events file to calculate timeline metrics:

```powershell
python bci_pipeline_v2.9.py `
  --mode validate_timeline `
  --edf "D:\BCI\data\session02_eeg.edf" `
  --events "D:\BCI\data\session02_events.tsv" `
  --model "D:\BCI\results\session01_model" `
  --output-dir "D:\BCI\results\session01_to_session02" `
  --event-overlap-threshold 0.5 `
  --smoothing-window 3
```

`--events` is optional in this mode. Without it, the pipeline saves predictions but cannot calculate accuracy.

Validation and prediction always reuse the channel list, normalization mode, sampling rate, and label mapping stored with the trained model.

## Unlabeled prediction

```powershell
python bci_pipeline_v2.9.py `
  --mode predict `
  --edf "D:\BCI\data\unlabeled_eeg.edf" `
  --model "D:\BCI\results\pooled_model" `
  --output-dir "D:\BCI\results\prediction" `
  --smoothing-window 3
```

Accuracy cannot be calculated without ground-truth events.

## Model and result files

A trained model directory contains model state and metadata such as:

```text
trained_model.npz
model_info.json
selected_features.json
training_summary.txt
```

Multi-session training also writes provenance files:

```text
train_manifest_resolved.csv
train_manifest_used.csv
```

Timeline modes can produce:

```text
validated_timeline.csv
timeline_metrics.json
timeline_confusion_matrix_raw.csv
timeline_confusion_matrix_smoothed.csv
timeline_confusion_matrix.csv
predicted_timeline.csv
prediction_summary.json
collapse_report.txt
```

Exact outputs depend on the selected mode and whether ground truth is available.

## Channel and preprocessing options

Channel presets:

```text
motor3
motor5
motor9
motor13
all_eeg
```

Normalization modes:

```text
none
zscore
```

With `zscore`, each selected raw EEG channel is normalized independently for its recording before filtering and CSP:

```text
x_normalized = (x - channel_mean) / channel_std
```

Changing the channel set or normalization mode requires training a new model.

Temporal smoothing accepts windows `1`, `3`, or `5`; `1` disables smoothing.

## Web UI

Start the local interface from the repository directory:

```powershell
python bci_web_ui.py
```

Then open <http://127.0.0.1:8765>.

The UI runs `bci_pipeline_v2.9.py` as a local child process. EEG data is processed locally and is not uploaded by the interface. Stop the server with `Ctrl+C`.

## Reporting results

For every reported result, record at least:

- training and test subjects and sessions;
- whether the test data influenced configuration selection;
- channel set and normalization mode;
- label mapping;
- balancing settings and seed;
- confidence, IDLE, overlap, and smoothing settings;
- number of evaluated windows and class supports;
- accuracy, balanced accuracy, per-class recall, and IDLE false-positive rate;
- the source commit and generated result files.

Do not describe cross-session results as cross-subject results unless the held-out subject was excluded from all training and model-selection steps.

## Dataset acknowledgement

This project was developed using the following OpenNeuro dataset:

Sarkar, S., Nathan, K., & Contreras-Vidal, J. L. (2026). *EEG-Controlled Exoskeleton for Walking and Standing: A Longitudinal Study of Healthy Individuals* (Version 1.0.1) [Data set]. OpenNeuro. <https://doi.org/10.18112/openneuro.ds007788.v1.0.1>

Users are responsible for reviewing and complying with the dataset's current terms, documentation, and citation requirements.

## Development history

Earlier prototypes explored real-time LSL and joystick-control components. Those prototypes are historical and are not part of the current offline v2.9 workflow. See `CHANGELOG.md` and the Git history for development context.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE.txt](LICENSE.txt) for the full license text.
