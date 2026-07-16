# EOG Artifact Ablation Study

## Background

An early prototype script (`validate_full.py`, since removed from the repo)
raised a suspicion on `sub-01`: EOG-only classification accuracy was
unexpectedly high, suggesting the model might be partially learning from
eye-movement artifacts rather than pure motor-cortex signal. That finding
predates the current CSP + shrinkage-LDA production pipeline
(`bci_pipeline_v2.9.py` / `modern_bci_v2.py`), so it was never re-checked
against it. This ablation closes that gap.

None of the existing `motor3`/`motor5`/`motor9`/`motor13` channel-set presets
ever included EOG channels, so a normal run of this pipeline was never at
risk of silently mixing EOG into the "EEG" feature set. Two new presets were
added to `CHANNEL_SETS` (`bci_pipeline_v2.9.py`) specifically to run this
study through the real production pipeline instead of a one-off script:

- `eog_only`: `EOG_HL`, `EOG_HR`, `EOG_VA`, `EOG_VB`
- `motor3_eog`: `motor3` (`C3`, `Cz`, `C4`) plus the same four EOG channels

## Method

Reciprocal same-subject cross-session validation on `sub-02` (`ses-01` <->
`ses-02`), the same design used in `v2.9_verified_cross_session_results/`:
train on one session with `--mode train`, validate on the other with
`--mode validate_timeline`, for both directions and all three channel sets.
Default configuration otherwise (`channel-normalization=zscore`,
`n-features-select=45`, `lda-shrinkage=0.0`, smoothing window 3).

## Results

| Channel set | Train -> Test | Balanced accuracy | WALK recall | STOP recall |
|---|---|---:|---:|---:|
| `motor3` (EEG only) | ses-01 -> ses-02 | 74.02% | 89.89% | 58.14% |
| `motor3` (EEG only) | ses-02 -> ses-01 | 77.17% | 57.92% | 96.42% |
| **`motor3` mean** | — | **75.59%** | — | — |
| `eog_only` | ses-01 -> ses-02 | 70.58% | 85.79% | 55.37% |
| `eog_only` | ses-02 -> ses-01 | 76.35% | 90.16% | 62.54% |
| **`eog_only` mean** | — | **73.47%** | — | — |
| `motor3_eog` (EEG+EOG) | ses-01 -> ses-02 | 83.06% | 82.24% | 83.88% |
| `motor3_eog` (EEG+EOG) | ses-02 -> ses-01 | 70.77% | 100.00% | 41.53% |
| **`motor3_eog` mean** | — | **76.91%** | — | — |

No run collapsed (`collapse_warning: false` in every `timeline_metrics.json`).
Raw per-run outputs are reproducible with the commands in "Reproduction"
below; aggregated with `aggregate_results.py`.

## Interpretation

- **EOG-only accuracy (73.47%) is far above chance (50%)** — eye-movement
  channels alone carry substantial WALK/STOP-discriminative signal in this
  dataset, through the current production pipeline, not just the old
  prototype script. This **confirms** the original suspicion rather than
  resolving it.
- **EEG-only (75.59%) beats EOG-only by only ~2.1 points.** If the model
  were cleanly separating motor-cortex signal from eye-movement artifact,
  a much larger gap would be expected.
- **Adding EOG on top of EEG (76.91%) improves over EEG-only by only ~1.3
  points.** EOG is not adding much *independent* information beyond what
  EEG-only already captures — consistent with EEG-only already being
  partly driven by the same eye-movement-correlated signal that `eog_only`
  picks up directly, rather than EOG and EEG contributing separate,
  additive information.

**Practical implication:** headline WALK/STOP accuracy numbers reported
elsewhere in this repo for `motor3`/`motor5`/`motor9`/`motor13` (EEG-only
configurations) should be read with this caveat — some fraction of that
accuracy is plausibly attributable to eye-movement artifacts correlated
with the WALK/STOP task structure (e.g. gaze direction or blink pattern
differences between walking and standing), not solely to motor-intent
signal. This does not mean the model is *only* picking up EOG — EEG-only
still outperforms EOG-only — but the gap is too small to rule out
substantial artifact contamination.

This ablation does not attempt to *remove* the artifact (e.g. EOG
regression/cleaning, as the old prototype's "EOG-cleaned EEG" condition
did) — it only quantifies how much signal is present in EOG alone and how
much EEG+EOG gains over EEG-only. Artifact removal, if pursued, is separate
follow-up work.

## Reproduction

```bash
for cset in motor3 eog_only motor3_eog; do
  python bci_pipeline_v2.9.py --mode train \
    --edf sub-02/ses-01/eeg/sub-02_ses-01_task-training_eeg.edf \
    --events sub-02/ses-01/eeg/sub-02_ses-01_task-training_acq-rexcommand_events.tsv \
    --output-dir "out/${cset}_train-ses-01/model" --channel-set "$cset"

  python bci_pipeline_v2.9.py --mode validate_timeline \
    --edf sub-02/ses-02/eeg/sub-02_ses-02_task-training_eeg.edf \
    --events sub-02/ses-02/eeg/sub-02_ses-02_task-training_acq-rexcommand_events.tsv \
    --model "out/${cset}_train-ses-01/model" --output-dir "out/${cset}_train-ses-01/validate"
done
# repeat with ses-01/ses-02 swapped for the other direction, then:
python aggregate_results.py out --output-dir out
```
