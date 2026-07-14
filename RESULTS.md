# BCI Pipeline v2.9.1 — Cross-Session Validation

The uploaded `bci_pipeline_v2.9.1.py` was copied into the repository as
`bci_pipeline_v2.9.1.py` and executed directly from the command line.

## Frozen configuration

- channel set: `motor3` (`C3`, `Cz`, `C4`)
- channel normalization: `zscore`
- selected features: `25`
- LDA shrinkage: `0.1`
- confidence threshold: `0.45`
- IDLE distance threshold: `999`
- class balancing: `none`
- subject balancing: `none`
- seed: `42`
- event overlap threshold: `0.5`
- centered smoothing window: `5`

For `sub-01`, the previously best overall training-session count was selected:
**7 training sessions**. Each of the eight sessions was held out exactly once,
and the remaining seven sessions were pooled using `train_multi`.

## sub-01: eight-fold leave-one-session-out

| Held-out session | Accuracy | Balanced accuracy | WALK recall | WALK support | STOP recall | STOP support | Collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| `ses-01` | 82.96% | **83.53%** | 85.79% | 366 | 81.27% | 614 | No |
| `ses-02` | 86.43% | **87.68%** | 92.62% | 366 | 82.74% | 614 | No |
| `ses-03` | 88.98% | **89.55%** | 91.80% | 366 | 87.30% | 614 | No |
| `ses-04` | 87.04% | **87.23%** | 87.98% | 366 | 86.48% | 614 | No |
| `ses-05` | 82.24% | **77.33%** | 57.92% | 366 | 96.74% | 614 | No |
| `ses-06` | 77.35% | **80.21%** | 91.53% | 366 | 68.89% | 614 | No |
| `ses-07` | 82.76% | **81.05%** | 74.32% | 366 | 87.79% | 614 | No |
| `ses-08` | 83.98% | **81.31%** | 70.77% | 366 | 91.86% | 614 | No |
| **Session macro mean** | **83.97%** | **83.49%** | — | — | — | — | **0 / 8** |

The macro mean is calculated across the eight held-out sessions, giving each
session equal weight.

## sub-02: reciprocal two-session validation

| Train | Test | Accuracy | Balanced accuracy | WALK recall | WALK support | STOP recall | STOP support | Collapse |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `ses-01` | `ses-02` | 66.12% | **71.70%** | 93.72% | 366 | 49.67% | 614 | No |
| `ses-02` | `ses-01` | 75.71% | **80.51%** | 99.45% | 366 | 61.56% | 614 | No |
| **Direction mean** | — | **70.92%** | **76.10%** | — | — | — | — | **0 / 2** |

## Important v2.9.1 behavior

During `validate_timeline`, v2.9.1 computes z-score normalization statistics
from only the first 60 seconds of the held-out EDF. Training recordings still
use their full-record per-channel statistics. This changes the test-time
distribution and produces substantially different `sub-02` results from v2.9.

## Scientific scope

- Every held-out session was excluded from its training list.
- Sliding windows are 3.0 seconds long with a 0.25-second step, so neighboring
  windows overlap and are not independent trials.
- `IDLE` distance threshold `999` effectively disables the IDLE distance gate.
  The reported results assess labelled WALK/STOP windows, not background
  rejection.
- Smoothing window 5 is centered majority voting. It uses future neighboring
  windows and is therefore non-causal and not a real-time result.
- This is same-subject cross-session validation, not cross-subject validation.

## Testing

The file passed syntax compilation and all repository tests after the stale
test fixture reference was changed from `bci_pipeline_v2.8.py` to
`bci_pipeline_v2.9.1.py`:

```text
24 passed, 2 warnings
```

The two warnings are NumPy overflow warnings in the existing logistic
probability conversion in `modern_bci_v2.py`.

The repository's unmodified tests still point to `bci_pipeline_v2.8.py`, so
they initially report 17 setup errors until that filename is updated.

## Web UI

A compatible Web UI copy is included as `bci_web_ui_v2.9.1.py`. Its pipeline
discovery prioritizes `bci_pipeline_v2.9.1.py`.
