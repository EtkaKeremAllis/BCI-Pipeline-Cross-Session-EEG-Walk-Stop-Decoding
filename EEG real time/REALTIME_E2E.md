# Real-Time End-to-End Simulation (Phase B slice 7)

## What this verifies

The full chain — `FileReplaySource` -> `DecisionEngine` -> `WebSocketOutput`
(the same path `web_ui_live.py`'s `AppState` drives) — run against one real
recording (`sub-02/ses-02`, the `ses-01-to-ses-02` model), with three
distinct questions answered:

1. **Correctness**: does the chunked, online chain agree with
   `fast_causal_bci.validate()` (the existing offline/batch reference)?
2. **Real-time-ness**: does a `realtime_pace=True` replay actually take
   real wall-clock time, or does it silently fast-forward?
3. **Headroom**: is there enough CPU budget to keep up with a live 1x
   stream, and what does per-decision latency actually look like?

Test code: `tests/test_realtime_e2e.py`.

## 1. Online chain vs. offline batch reference

`validate()`'s underlying feature extraction filters out windows whose
event label isn't clean/consistent across the window
(`extract_causal_features_from_source`); `DecisionEngine` makes no such
filtering — it predicts on every ready window, the same as
`run_decision_source`. So the two outputs aren't the same *size*; matching
was done by `stream_time_s`, not by position.

**Result: all 4892 offline-batch windows matched their online counterpart
exactly** (the online chain produced 4941 total windows — 49 more than the
batch reference, all of them windows `validate()` excludes for
label-cleanliness reasons, not disagreements).

## 2. Real-time pacing

An 8-second `realtime_pace=True` replay (`FileReplaySource`) took **8.01s**
of actual wall-clock time — accurate to within 10ms, not an instant
fast-forward through the data.

## 3. CPU headroom and latency

Processing the *entire* recording (247.98s of real time) as fast as
possible (not real-time-paced) took:

| Metric | Value |
|---|---:|
| Recording duration | 247.98 s |
| Total decisions | 4941 |
| Unpaced wall-clock time | 9.95 s |
| Unpaced CPU time | 9.82 s |
| **CPU time as a fraction of real-time** | **3.96%** |

That is, the entire pipeline (feature extraction + classification, per
decision) uses under 4% of a single CPU core relative to the recording's
real duration — a live 1x-speed stream would leave enormous headroom.

Per-decision latency (from chunk arrival to decision, `end_to_end_ms`) and
total latency including the smoothing delay (`total_latency_ms`,
`smoothing_window=1` here so they're nearly identical):

| Percentile | end_to_end_ms | total_latency_ms |
|---|---:|---:|
| p50 | 1.94 | 1.94 |
| p95 | 2.19 | 2.19 |
| p99 | 2.42 | 2.42 |
| max | 16.78 | 16.80 |

The max is a single outlier (likely a GC pause or OS scheduling blip); p99
is still well under the model's 50ms decision step, so this pipeline is
nowhere near its real-time budget.

Raw numbers are also written to `results/realtime_e2e_report.json` each
time the test suite runs.

## Scope note

This is still a **replay simulation**, not a test against real EEG
hardware — see the project's Phase 5 documentation slice for the standing
scope caveat. It proves the software chain (source -> decision engine ->
output sink) behaves correctly and has real-time headroom on this machine;
it does not (and cannot, without hardware) prove behavior against a live
device's actual latency/jitter characteristics.
