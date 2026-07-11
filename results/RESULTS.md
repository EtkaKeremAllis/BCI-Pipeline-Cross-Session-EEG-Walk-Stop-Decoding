# EEG Walk/Stop BCI — Experimental Results

> **Current evidence level:** exploratory cross-session development benchmark.  
> **Cross-subject performance:** not yet claimed; the frozen evaluation runner is included in this repository.

## Headline result

The current best low-cost configuration uses only **three motor-area electrodes (C3, Cz, C4)** and reached **77.95% mean balanced accuracy** in reciprocal cross-session validation on `sub-02`.

| Metric | Result |
|---|---:|
| Mean deployment accuracy | **79.39%** |
| Mean balanced accuracy | **77.95%** |
| Worst-direction balanced accuracy | **77.03%** |
| Mean WALK recall | **72.27%** |
| Mean STOP recall | **83.63%** |
| Collapse events | **0 / 2** |

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

## Cross-subject and broader cross-session evaluation

The dataset contains seven participants and nine longitudinal sessions per participant. The public data are from OpenNeuro dataset `ds007788` (version `1.0.1`, CC0). The repository includes a fixed-protocol runner that discovers all available training recordings and evaluates:

1. **Within-subject leave-one-session-out:** train on the other sessions of one participant and test the held-out session.
2. **Cross-subject leave-one-subject-out:** train on all other participants and test every session of the held-out participant.

| Protocol | Intended folds | Hyperparameter tuning on test folds | Current status |
|---|---:|---|---|
| Reciprocal development pair (`sub-02`, sessions 01/02) | 2 | Yes — development only | **Completed** |
| Within-subject leave-one-session-out | Up to 63 | No; configuration frozen | Runner ready; raw EDF subset required |
| Cross-subject leave-one-subject-out | Up to 63 test sessions / 7 shared models | No; configuration frozen | Runner ready; raw EDF subset required |

No cross-subject number is shown here until those folds have actually completed. This avoids presenting a planned benchmark as measured evidence.

## Interpretation and limitations

- The 77.95% figure is an **exploratory development estimate** because feature count, shrinkage, smoothing, and normalization were selected using the same two sessions.
- Same-subject cross-session validation does not establish cross-person generalization.
- High performance can reflect neural activity, but also movement, ocular, muscle, cable, feedback, or session-correlated artifacts.
- `IDLE` gating was effectively disabled (`999`) in the headline benchmark; the reported task is primarily STOP-versus-WALK command-period classification.
- The model should not be described as a proven decoder of pure brain-internal walking intention.

## Data and machine-readable files

You can find detailed results in results file.

## Dataset attribution

- **Dataset:** EEG-Controlled Exoskeleton for Walking and Standing — A Longitudinal Study of Healthy Individuals
- **OpenNeuro:** `ds007788`, version `1.0.1`
- **DOI:** https://doi.org/10.18112/openneuro.ds007788.v1.0.1
- **License:** CC0
- **Authors:** Shantanu Sarkar, Kevin Nathan, Jose L. Contreras-Vidal

---

Last verified: 2026-07-11.
