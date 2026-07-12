# Verified v2.9 reciprocal cross-session rerun

This report was generated from a fresh extraction of the ZIP supplied by the user. No pre-existing models or result folders were used.

## Source and runtime

- ZIP archive comment / source commit: `25047c459adb0205705cd17abc37bdd952343e45`
- ZIP SHA-256: `6889c9f22a6bcba9c03cd4d7dade99c5bd69b7c409e0066ea59e7acc169f77b4`
- `bci_pipeline_v2.9.py` SHA-256: `1c8f70e4ef50d99fc5aa18a48c94f87bb2847d642c13426458d4f3ad7e831cfa`
- Python: `3.13.5`
- NumPy: `2.3.5`
- SciPy: `1.17.0`
- pdfplumber: `0.11.9`

## Configuration

- Same subject, different sessions: `sub-02`
- Channels: `C3`, `Cz`, `C4` (`motor3`)
- Channel normalization: `zscore`
- Selected features: `25`
- LDA shrinkage: `0.1`
- Window / step: `3.0 s / 0.25 s`
- Smoothing: centered majority vote, window `5`
- Class balancing: `none`
- Confidence threshold: `0.45`
- IDLE distance threshold: `999`
- Seed: `42`
- Event overlap threshold: `0.5`

## Actual rerun results

| Train | Test | Raw acc. | Raw bal. acc. | Smoothed acc. | Smoothed bal. acc. | WALK recall | STOP recall | Windows | Collapse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `sub-02/ses-01` | `sub-02/ses-02` | 78.06% | 80.40% | **78.67%** | **81.10%** | 90.71% | 71.50% | 980 | No |
| `sub-02/ses-02` | `sub-02/ses-01` | 87.86% | 86.83% | **90.00%** | **89.10%** | 85.52% | 92.67% | 980 | No |
| **Mean** | — | — | — | **84.34%** | **85.10%** | **88.11%** | **82.08%** | — | **0 / 2** |

The `90.00%` value is therefore real for the **single direction** `ses-02 -> ses-01`; it is not the two-direction mean. The reciprocal mean smoothed accuracy is `84.34%`, and the reciprocal mean balanced accuracy is `85.10%`.

## Determinism check

Both complete train -> validate directions were run a second time in separate fresh output directories. The repeated `timeline_metrics.json` and `validated_timeline.csv` files matched byte-for-byte / value-for-value for both directions.

## Test-suite check

- As shipped in the ZIP: `7 passed, 17 errors`.
- Cause: `tests/conftest.py` still hardcodes `bci_pipeline_v2.8.py`, which no longer exists.
- After changing only that fixture filename to `bci_pipeline_v2.9.py` in the extracted test copy: `24 passed, 2 warnings`.
- The two warnings are NumPy overflow warnings in the logistic probability conversion inside `modern_bci_v2.py`; the tests still pass.
- `python bci_pipeline_v2.9.py --help` also still describes itself as `pipeline v2.8`. This is a stale version string, not a benchmark failure.

## Scope

The 980 windows in each direction overlap strongly and are not independent trials. The benchmark evaluates labelled WALK/STOP windows only. IDLE/background rejection and false activations during unlabelled periods are not measured because the IDLE gate is effectively disabled with threshold `999`.
