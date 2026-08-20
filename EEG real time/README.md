# EEG Real-Time Subsystem

A causal, streaming-capable WALK/STOP classifier, built separately from
the offline `bci_pipeline_v2.9.py` pipeline at the repo root.

> **Scope note:** this has been validated by replaying a recorded EEG file
> as if it were arriving live - genuine real-time pacing, latency, and CPU
> headroom were measured this way (see `REALTIME_E2E.md`), but **none of
> it has been tested against real EEG hardware**, since no hardware is
> available to this project. See `ARCHITECTURE.md` for the full design
> and scope discussion.

## Why this is a separate subsystem

The root pipeline is offline-only by design (non-causal centered
smoothing, whole-recording z-score normalization, no streaming concept).
Rather than retrofit that in place, `fast_causal_bci.py` is a from-scratch
causal classifier - stateful IIR filters (`scipy.signal.lfilter`, no
`filtfilt`), multi-scale energy/correlation features, and the same
shrinkage-LDA core (`modern_bci_v2.SimpleLDA`) reused from the root
pipeline - independently built and independently verified.

## Layout

```
fast_causal_bci.py       causal classifier: train / validate / replay CLI
fast_causal_web_ui.py     local web UI driving fast_causal_bci.py directly
realtime/                 layered streaming architecture on top of it
  eeg_source.py             EEGSource protocol + LiveDeviceSource placeholder
  file_replay_source.py     replays a recorded EDF at real wall-clock pace
  decision_engine.py        source -> features -> model -> smoothing
  online_smoothing.py       causal counterpart to the offline smoothing
  output_sink.py            console / CSV / websocket-queue / hardware / fan-out
  web_ui_live.py            web UI driven through DecisionEngine + WebSocketOutput
models/                   trained model directories (per session pair)
results/                  results from the end-to-end test run
tests/                    unit + end-to-end tests for everything above
```

`edf_reader.py`, `modern_bci_v2.py`, and `parse_events.py` are **not**
duplicated here - both entry scripts and `realtime/web_ui_live.py` import
them from the repo root via an explicit `sys.path` insertion at the top of
each file, so there is one copy of each to keep in sync, not two.

## Quick start

From this folder:

```powershell
python -m pip install -r ../requirements.txt

# train + evaluate the causal classifier directly
python fast_causal_bci.py --mode train    --edf <path.edf> --events <path.tsv> --model <model_dir>
python fast_causal_bci.py --mode validate --edf <path.edf> --events <path.tsv> --model <model_dir>

# replay an EDF at real EEG speed and print timing
python fast_causal_bci.py --mode replay --edf <path.edf> --model <model_dir> --output replay_timing.csv

# simple web UI, driven directly by fast_causal_bci.py
python fast_causal_web_ui.py --port 8766

# layered web UI, driven through DecisionEngine + WebSocketOutput
python realtime/web_ui_live.py --port 8767
```

Both web UIs bind to `127.0.0.1` only by default. See `WEB_UI.md` for UI
usage details and `ARCHITECTURE.md` for what each `realtime/` module does
and how they compose.

## Testing

```powershell
python -m pip install -r ../requirements.txt
python -m pytest tests/ -v
```

`tests/test_realtime_e2e.py` is the full-chain check: it confirms the
online (`DecisionEngine`) and offline (`fast_causal_bci.validate()`) paths
agree window-for-window, that a real-time-paced replay actually takes real
wall-clock time (not a fast-forward), and reports CPU headroom - see
`REALTIME_E2E.md` for the numbers from the last run.

## What's verified vs. not

See the table in `ARCHITECTURE.md` - short version: the software chain
(source -> decision engine -> output sink) is verified end-to-end against
recorded data; behavior against a live EEG device is not, because no
device is available to test against.
