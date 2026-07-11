# Dataset: OpenNeuro ds007788

## Source

**Title:** EEG-Controlled Exoskeleton for Walking and Standing — A Longitudinal Study of Healthy Individuals
**Authors:** Shantanu Sarkar, Kevin Nathan, Jose L. Contreras-Vidal
**OpenNeuro accession:** [ds007788](https://openneuro.org/datasets/ds007788) (DOI: `10.18112/openneuro.ds007788.v1.0.1`)
**License:** CC0 (public domain). The repository asks that the dataset be cited — see `Acknowledgements` in README.md.

Identified from the data itself: the EDF header `recording_id` field reads `MOVE System` and events are split into `acq-rexcommand` / `acq-rexstate` files, which exactly match this dataset's own `eeg.json` (`"Manufacturer": "MOVE System"`) and events sidecar (`"LongName": "Event category for rexCommand"`). Confirmed against the dataset's GitHub mirror ([OpenNeuroDatasets/ds007788](https://github.com/OpenNeuroDatasets/ds007788)).

## Study design

- 7 healthy adults (4 male, 3 female), ages 20-30.
- 9 sessions per participant, spanning 15-81 days.
- Each session: an open-loop `training` task (motor-imagery calibration, no feedback), 12 closed-loop `trial01`-`trial12` tasks, and two 6-minute blocks (`walk6min`, `stop6min`).
- Device under control: a Rex Bionics lower-limb exoskeleton ("Rex").

## Signal specs (from `eeg.json` / verified against local files)

- **EEG:** 60 scalp channels, 10-20 placement, `actiCAP`.
- **EOG:** 4 channels — `EOG_HL`, `EOG_HR`, `EOG_VA`, `EOG_VB` (horizontal left/right, vertical A/B).
- **Sampling rate:** 100 Hz (native, not downsampled by this pipeline).
- **Power line frequency:** 60 Hz.
- Full 64-channel list (order as stored in the EDF):
  `Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, EOG_HL, CP5, CP1, CP2, CP6, EOG_HR, P7, P3, Pz, P4, P8, EOG_VA, O1, Oz, O2, EOG_VB, AF7, AF3, AF4, AF8, F5, F1, F2, F6, AFz, FT7, FC3, FC4, FT8, FCz, C5, C1, C2, C6, TP7, CP3, CPz, CP4, TP8, P5, P1, P2, P6, PO7, PO3, POz, PO4, PO8`

This pipeline's `motor3` channel set (`C3`, `Cz`, `C4`) and other `motorN` presets in `resolve_channels()` are all present verbatim in this list, no renaming needed.

## Event scheme (from `*_acq-rexcommand_events.json`)

`events.tsv` columns: `onset` (s), `duration` (s), `trial_type`.

| `trial_type` | Meaning | Pipeline label |
|---|---|---|
| `x5` | Stop command sent to Rex | `STOP` (0) |
| `x8` | Walk command sent to Rex | `WALK` (1) |
| `x99` | Idle | dropped (not in `DEFAULT_LABEL_MAP`) |

This confirms `DEFAULT_LABEL_MAP = {'x5': 0, 'x8': 1}` in `bci_pipeline_v2.8.py` matches the dataset's own documented scheme exactly.

`acq-rexstate` files log the exoskeleton's actual state feedback (not currently consumed by this pipeline); `acq-infoclosedloop` and `recording-*_stim` files (closed-loop trials only) are BMI-prediction/beep/fail-counter logs, also not currently used.

## File format quirk: some `events` files are PDF, not TSV

The raw dataset on OpenNeuro ships `events.tsv` as plain tab-separated text. Some files in this repo were re-exported as PDF at some point before being added here (`sub-01/*/eeg/*_events.pdf`), and one was a plain-text TSV mislabeled with a `.pdf` extension (`sub-02_ses-02`, fixed to `.tsv` in this commit — its content was always plain text). `parse_events.py` auto-detects by extension (`.tsv` vs `.pdf`); if a new file turns out to be plain text with a `.pdf` name, rename it to `.tsv` rather than adding a new code path.

## Local data inventory (this repo)

Only a subset of the full dataset is kept here — see "Getting more data" below for the rest.

| Subject | Sessions present | Task | Events format |
|---|---|---|---|
| sub-01 | ses-01 .. ses-08 | training | `.pdf` |
| sub-02 | ses-01, ses-02 | training | `.tsv` |
| sub-03 | ses-01 | training | `.tsv` |

Layout follows BIDS: `sub-<ID>/ses-<ID>/eeg/sub-<ID>_ses-<ID>_task-<task>[_acq-<label>]_<suffix>.<ext>`. `train_list.txt` at the repo root lists all sessions above in the `--dataset-list` CSV format consumed by `bci_pipeline_v2.8.py --mode train_multi` (paths are relative; run with `--dataset-dir .` from the repo root).

Only `training`-task recordings were pulled in; `trial01-12`, `walk6min`, and `stop6min` (closed-loop and extended blocks) exist upstream but are not in this repo.

## Getting more data

The full dataset (7 subjects x 9 sessions x ~15 files/session) is not checked into this repo to avoid bloating it. To fetch a specific file, OpenNeuro serves individual files directly (not just via `git-annex`, which the GitHub mirror uses and which only returns pointer files over plain `git clone`/`raw.githubusercontent.com`):

```bash
curl -sL -o <local_filename> \
  "https://openneuro.org/crn/datasets/ds007788/snapshots/1.0.1/files/sub-<ID>:ses-<ID>:eeg:<filename>"
```

(Note the `:` separators in the URL path, not `/`.) Browse the file tree at [github.com/OpenNeuroDatasets/ds007788](https://github.com/OpenNeuroDatasets/ds007788) to find exact filenames, then substitute into the URL above.
