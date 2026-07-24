# Real-Time Subsystem Architecture (Phase B)

## Scope note (read this first)

**This is a replay simulation, validated against recorded EEG data — not a
test against real EEG hardware.** No physical EEG device is available to
this project. Every "real-time" claim below (pacing, latency, CPU
headroom) was measured by replaying a recorded `.edf` file as if it were
arriving live (`FileReplaySource`), not by reading from a live device. The
architecture is designed so that plugging in a real device later (via
`LiveDeviceSource` in `fast_causal_bci.py`, currently an untested
placeholder) requires no changes to anything downstream of it — but until
that device exists, "real-time" here means "correctly paced replay of
recorded data," nothing more.

## Why a separate `EEG real time/` subsystem

The root pipeline (`bci_pipeline_v2.9.py`) is intentionally offline-only:
non-causal centered smoothing, whole-recording z-score normalization, no
streaming concept at all. Making it real-time-capable in place would mean
rewriting its filtering, features, and smoothing simultaneously - too much
change to verify at once. `fast_causal_bci.py` (in this folder) is instead
a from-scratch, causal-by-construction classifier (stateful IIR filters via
`scipy.signal.lfilter`, multi-scale energy/correlation features, the same
shrinkage-LDA core reused from `modern_bci_v2.SimpleLDA`), built and
verified independently, with its own documented benchmark (`WEB_UI.md`:
~73.77% cross-session balanced accuracy).

The `realtime/` package below does not modify that classifier's core
(filter bank, feature extraction, model). It wraps it in a small, layered
architecture so that streaming source, decision logic, smoothing, and
output destination can each be swapped or extended independently.

## Architecture

```text
EEGSource (Protocol: realtime/eeg_source.py)
  sampling_rate, channels, is_live, chunks(), close()
  |
  +-- FileReplaySource (realtime/file_replay_source.py)
  |     replays a recorded EDF as device-like chunks; realtime_pace=True
  |     paces itself to real wall-clock time via drift-free absolute
  |     scheduling deadlines. is_live=False.
  |
  +-- LiveDeviceSource (fast_causal_bci.py)
        placeholder adapter for a real device SDK. Untested - no hardware
        exists yet to verify it against (see scope note above).
        |
        v
DecisionEngine (realtime/decision_engine.py)
  Consumes EEGSource chunks through fast_causal_bci's untouched
  FeatureStream + FastCausalModel, then:
    predict -> confidence gate (-> IDLE) -> OnlineSmoother
  OnlineSmoother (realtime/online_smoothing.py) is a delayed, causal
  counterpart to the offline pipeline's centered apply_temporal_smoothing():
  byte-for-byte identical output given the same input, at the cost of
  window_size // 2 chunks of extra latency.
  Yields fully-resolved Decision objects (raw + smoothed label correctly
  paired to the same stream_time_s, plus per-decision timing).
        |
        v
OutputSink (Protocol: realtime/output_sink.py)
  write(decision), close(), context-manager support
  |
  +-- ConsoleOutput       - print to a stream, local debugging
  +-- CSVOutput           - one CSV row per decision
  +-- WebSocketOutput     - subscribe()/unsubscribe() plain queue.Queue
  |                         objects; write() broadcasts to all of them.
  |                         No network code here by design - see below.
  +-- HardwareOutput      - placeholder; raises NotImplementedError on
  |                         construction (no HID/vJoy device exists yet)
  +-- MultiOutput         - fan one decision stream out to several sinks
        |
        v
web_ui_live.py (realtime/web_ui_live.py)
  A local HTTP server (ThreadingHTTPServer) whose AppState subscribes to a
  WebSocketOutput exactly the way a real browser client eventually would,
  and serves the same HTML/CSS/JS as fast_causal_bci's original
  fast_causal_web_ui.py (start/stop replay, live WALK/STOP/IDLE state,
  history table, optional ground-truth column). Polling-based today
  (matches what was already deployed and verified); the WebSocketOutput
  abstraction itself doesn't care whether the transport is a real
  websocket or HTTP polling.
```

## Design principle: don't destabilize the tested core

`fast_causal_bci.py`'s filter bank, feature extraction, and model
(`CausalFilterBank`, `FeatureStream`, `FastCausalModel`) are treated as a
stable, already-verified boundary. Every module above is additive - it
wraps or reads from that core, but doesn't modify it (the one exception,
`RecordedReplaySource = FileReplaySource`, is a same-behavior import alias
for backward compatibility, not a logic change). New capabilities (a new
output destination, a new decision policy) are meant to be added as new
files, not as edits to files every other module depends on.

This mattered concretely during development: an independent effort
(outside this `realtime/` package) extended `fast_causal_bci.run_decision_source()`
itself with callback-based smoothing/latency tracking, in parallel with
this package's `DecisionEngine`/`OutputSink` covering the same ground.
Reconciling that required porting the valuable ideas (a `stop_check`
callback, correctly-paired raw/smoothed records via a pending-record queue,
richer latency accounting) into this layered architecture rather than
duplicating a second competing implementation - exactly the kind of
duplicate effort a stable, rarely-touched core is meant to avoid.

## What's verified vs. what isn't

| Claim | Verified how |
|---|---|
| Causal filtering + smoothing behavior | `tests/test_online_smoothing.py` - byte-for-byte match with the offline pipeline's `apply_temporal_smoothing()`, 45+ randomized cases |
| Decision loop matches the existing, deployed classifier | `tests/test_decision_engine.py` - 0 mismatches vs. `run_decision_source()` across 4941 real decisions |
| Output routing (console/CSV/live-queue/fan-out) | `tests/test_output_sink.py`, including a real `DecisionEngine` integration run |
| Full chain agrees with the offline batch reference | `tests/test_realtime_e2e.py` - all 4892 `validate()` windows matched their online counterpart |
| Genuine real-time pacing (not fast-forwarding) | `tests/test_realtime_e2e.py` - an 8.0s paced replay took 8.01s wall-clock |
| CPU headroom for live 1x streaming | `tests/test_realtime_e2e.py` / `REALTIME_E2E.md` - 3.96% CPU-time-to-real-time ratio on the full recording |
| Behavior against real EEG hardware | **Not verified - no hardware available.** `LiveDeviceSource` is an untested placeholder. |

See `REALTIME_E2E.md` for the full write-up of the end-to-end results, and
`WEB_UI.md` for the original `fast_causal_bci.py` classifier's own
benchmark and smoke-test history.
