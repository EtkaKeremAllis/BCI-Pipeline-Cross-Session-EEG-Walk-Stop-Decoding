# EEG Walk/Stop BCI — Experimental Results

> **Current evidence level:** exploratory same-subject cross-session development benchmark.  
> **Cross-subject performance:** not yet evaluated or claimed.

## Reproducibility metadata

- **Code commit used for the reported benchmark:** `d90700b6346e1fe2c1fdd37d180eeb4c67a1bcad`
- **Operating system:** Windows 11 Pro
- **Python:** 3.13.14
- **Python executable:** `C:\Users\kerem\AppData\Local\Programs\Python\Python313\python.exe`
- **NumPy:** 2.5.0
- **SciPy:** 1.18.0
- **MNE:** not installed
- **scikit-learn:** not installed

The reported benchmark was executed with the Python interpreter shown above. MNE and scikit-learn were not installed and were not required by the pipeline.

> Replace `d90700b6346e1fe2c1fdd37d180eeb4c67a1bcad` with the output of `git rev-parse HEAD` from the exact repository state used to generate the committed result files.

## Headline result

The current best low-cost configuration uses only **three motor-area electrodes (C3, Cz, C4)** and reached **77.95% mean balanced accuracy** in reciprocal cross-session validation on `sub-02`.

| Metric | Result |
|---|---:|
| Mean window-level accuracy | **79.39%** |
| Mean balanced accuracy | **77.95%** |
| Worst-direction balanced accuracy | **77.03%** |
| Mean WALK recall | **72.27%** |
| Mean STOP recall | **83.63%** |
| Collapse events | **0 / 2** |

The two held-out recordings produced approximately **980 overlapping evaluation windows per direction**. These windows are **not statistically independent samples**: adjacent windows share EEG data and are temporally correlated because a 3.0 s window was advanced in 0.25 s steps. The reported metrics therefore describe window-level decoding performance on the held-out recordings, not performance across 980 independent trials. No confidence interval or significance test assuming independent windows is reported.

## Frozen configuration

| Component | Value |
|---|---|
| EEG channels | `C3`, `Cz`, `C4` (`motor3`) |
| Raw-channel normalization | Per-recording, per-channel z-score |
| Classifier | CSP + feature selection + shrinkage LDA |
| Selected features | 25 |
| LDA shrinkage | 0.10 |
| Window / step | 3.0 s / 0.25 s |
| Temporal smoothing | Centered majority vote, window 5 |
| Class balancing | None |
| Confidence threshold | 0.45 |
| IDLE distance threshold | 999 (effectively disabled in this benchmark) |

## Reciprocal cross-session results

| Train | Test | Accuracy | Balanced accuracy | WALK recall | STOP recall | Collapse |
|---|---|---:|---:|---:|---:|---|
| `sub-02/ses-01` | `sub-02/ses-02` | 75.92% | **77.03%** | 81.42% | 72.64% | No |
| `sub-02/ses-02` | `sub-02/ses-01` | 82.86% | **78.87%** | 63.11% | 94.63% | No |
| **Mean** | — | **79.39%** | **77.95%** | **72.27%** | **83.63%** | **0 / 2** |

### Evaluation scope

This frozen benchmark evaluates only labelled **WALK** and **STOP** command intervals. `IDLE`/background periods were excluded from the reported accuracy, balanced accuracy, and class-recall calculations. The `IDLE` distance threshold was effectively disabled by setting it to `999`.

Consequently, these results do **not** measure:

- IDLE rejection performance;
- false activations during unlabelled or background periods;
- three-class WALK/STOP/IDLE performance;
- end-to-end continuous-control safety or reliability.

## Channel-cost comparison

This comparison used `sub-02/ses-01 → sub-02/ses-02` with the earlier baseline settings (45 features, smoothing 3). It is included to show the performance/cost trade-off; it is **not** the final frozen benchmark configuration.

| Channel set | Channels | Smoothed accuracy | Balanced accuracy | WALK recall | STOP recall |
|---|---:|---:|---:|---:|---:|
| `all_eeg (reference only)` | 60 | 81.94% | 81.89% | 81.69% | 82.08% |
| `motor9` | 9 | 73.78% | 71.90% | 64.48% | 79.32% |
| `motor3` | 3 | 70.20% | 71.09% | 74.59% | 67.59% |
| `motor13` | 13 | 70.00% | 70.71% | 73.50% | 67.92% |
| `motor5` | 5 | 63.27% | 63.40% | 63.93% | 62.87% |

Although `all_eeg` was strongest in this single direction, it uses 60 scalp channels and may exploit broader movement-, ocular-, muscle-, or cable-related signals. The optimized `motor3` model is the current practical choice because it is cheaper, simpler, and performed best after feature/shrinkage optimization.

## Highest exploratory configurations

All rows below are means across the two reciprocal `sub-02` directions. These are tuning results, not independent final estimates.

| Rank | Set | Norm. | Features | Shrinkage | Smoothing | Mean accuracy | Mean balanced | Worst balanced | Collapse |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `motor3` | `zscore` | 25 | 0.10 | 5 | 79.39% | **77.95%** | 77.03% | False |
| 2 | `motor3` | `zscore` | 25 | 0.20 | 5 | 78.83% | **77.42%** | 76.70% | False |
| 3 | `motor3` | `zscore` | 25 | 0.10 | 3 | 78.72% | **77.20%** | 76.59% | False |
| 4 | `motor3` | `zscore` | 25 | 0.00 | 5 | 78.42% | **76.90%** | 76.34% | False |
| 5 | `motor3` | `zscore` | 25 | 0.20 | 3 | 78.21% | **76.79%** | 75.86% | False |
| 6 | `motor3` | `zscore` | 25 | 0.10 | 1 | 78.21% | **76.74%** | 76.24% | False |
| 7 | `motor3` | `zscore` | 25 | 0.35 | 5 | 78.16% | **76.64%** | 75.74% | False |
| 8 | `motor3` | `zscore` | 25 | 0.00 | 3 | 77.96% | **76.56%** | 76.20% | False |
| 9 | `motor3` | `zscore` | 25 | 0.35 | 3 | 77.96% | **76.42%** | 75.72% | False |
| 10 | `motor3` | `zscore` | 25 | 0.20 | 1 | 77.76% | **76.32%** | 75.29% | False |

## Reproduction

Only the two reciprocal `sub-02` commands below are documented here because the repository currently contains the corresponding EDF/TSV inputs and example outputs. No unavailable benchmark scripts are referenced.

### Exact commands for the reported frozen reciprocal benchmark

Run these commands from the repository root in **Windows PowerShell**. They reproduce the two reported directions using the frozen configuration shown above.

#### 1. Train on Session 01

```powershell
py .\bci_pipeline_v2.8.py `
  --mode train `
  --edf ".\sub-02_ses-01_task-training_eeg.edf" `
  --events ".\sub-02_ses-01_task-training_acq-rexcommand_events.tsv" `
  --output-dir ".\results\session01_to_session02\model" `
  --channel-set motor3 `
  --channel-normalization zscore `
  --n-features-select 25 `
  --lda-shrinkage 0.1 `
  --balance-classes none `
  --confidence-threshold 0.45 `
  --idle-distance-threshold 999 `
  --seed 42
```

#### 2. Validate Session 01 model on Session 02

```powershell
py .\bci_pipeline_v2.8.py `
  --mode validate_timeline `
  --edf ".\sub-02_ses-02_task-training_eeg.edf" `
  --events ".\sub-02_ses-02_task-training_acq-rexcommand_events.tsv" `
  --model ".\results\session01_to_session02\model" `
  --output-dir ".\results\session01_to_session02" `
  --smoothing-window 5 `
  --confidence-threshold 0.45 `
  --idle-distance-threshold 999
```

#### 3. Train on Session 02

```powershell
py .\bci_pipeline_v2.8.py `
  --mode train `
  --edf ".\sub-02_ses-02_task-training_eeg.edf" `
  --events ".\sub-02_ses-02_task-training_acq-rexcommand_events.tsv" `
  --output-dir ".\results\session02_to_session01\model" `
  --channel-set motor3 `
  --channel-normalization zscore `
  --n-features-select 25 `
  --lda-shrinkage 0.1 `
  --balance-classes none `
  --confidence-threshold 0.45 `
  --idle-distance-threshold 999 `
  --seed 42
```

#### 4. Validate Session 02 model on Session 01

```powershell
py .\bci_pipeline_v2.8.py `
  --mode validate_timeline `
  --edf ".\sub-02_ses-01_task-training_eeg.edf" `
  --events ".\sub-02_ses-01_task-training_acq-rexcommand_events.tsv" `
  --model ".\results\session02_to_session01\model" `
  --output-dir ".\results\session02_to_session01" `
  --smoothing-window 5 `
  --confidence-threshold 0.45 `
  --idle-distance-threshold 999
```

The validation command writes the original pipeline output names, including `timeline_metrics.json`, `validated_timeline.csv`, and the raw/smoothed confusion matrices. The committed example folders may use clearer presentation names such as `metrics.json` and `timeline_predictions.csv`; their numerical contents come from these validation runs.

## Interpretation and limitations

- The 77.95% figure is an **exploratory development estimate** because feature count, shrinkage, smoothing, and normalization were selected using the same two sessions.
- The approximately 980 overlapping windows per direction are temporally correlated and must not be interpreted as independent observations.
- Same-subject cross-session validation does not establish cross-person generalization.
- High performance can reflect neural activity, but also movement, ocular, muscle, cable, feedback, or session-correlated artifacts.
- `IDLE` was not evaluated in the headline benchmark. Background periods were excluded, and the `IDLE` gate was effectively disabled with a threshold of `999`.
- The reported numbers do not characterize false activations during background periods or full continuous-control performance.
- The model should not be described as a proven decoder of pure brain-internal walking intention.

## Data and machine-readable files

- [`results/verified_sub02_cross_session.csv`](results/verified_sub02_cross_session.csv)
- [`results/channel_set_comparison.csv`](results/channel_set_comparison.csv)
- [`results/optimization_top10.csv`](results/optimization_top10.csv)
- [`results/verified_results.json`](results/verified_results.json)

## Dataset attribution

- **Dataset:** EEG-Controlled Exoskeleton for Walking and Standing — A Longitudinal Study of Healthy Individuals
- **OpenNeuro:** `ds007788`, version `1.0.1`
- **DOI:** https://doi.org/10.18112/openneuro.ds007788.v1.0.1
- **License:** CC0
- **Authors:** Shantanu Sarkar, Kevin Nathan, Jose L. Contreras-Vidal

---

Last verified: 2026-07-11.
