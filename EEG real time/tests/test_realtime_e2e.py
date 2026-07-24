"""
End-to-end integration test for Phase B slice 7: run the full chain
(FileReplaySource -> DecisionEngine -> WebSocketOutput, the same path
web_ui_live.py's AppState drives) against one real recording, verify its
raw predictions agree with fast_causal_bci's own offline/batch reference
(validate()), confirm the replay genuinely paces itself in real time (not
just fast-forwarding through the data), and record latency + CPU cost.

Run from the "EEG real time" directory:
    pytest tests/test_realtime_e2e.py -v -s

See EEG real time/REALTIME_E2E.md for the human-readable writeup of what
this test found.
"""
import json
import os
import sys
import time

import numpy as np
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_REALTIME_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(EEG_REALTIME_DIR)

if EEG_REALTIME_DIR not in sys.path:
    sys.path.insert(0, EEG_REALTIME_DIR)

from fast_causal_bci import FastCausalModel, load_recording, validate  # noqa: E402
from realtime.decision_engine import DecisionEngine  # noqa: E402
from realtime.file_replay_source import FileReplaySource  # noqa: E402
from realtime.output_sink import WebSocketOutput  # noqa: E402

MODEL_DIR = os.path.join(EEG_REALTIME_DIR, "models", "ses-01-to-ses-02")
TEST_EDF = os.path.join(REPO_ROOT, "sub-02", "ses-02", "eeg",
                         "sub-02_ses-02_task-training_eeg.edf")
TEST_EVENTS = os.path.join(REPO_ROOT, "sub-02", "ses-02", "eeg",
                           "sub-02_ses-02_task-training_acq-rexcommand_events.tsv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_DIR) and os.path.exists(TEST_EDF) and os.path.exists(TEST_EVENTS)),
    reason="real model/recording not present in this checkout",
)


@pytest.fixture(scope="module")
def model():
    return FastCausalModel.load(MODEL_DIR)


def _run_online_full_chain(model, recording_signals, channels, fs, realtime_pace=False,
                            max_seconds=None):
    """FileReplaySource -> DecisionEngine -> WebSocketOutput, unpaced by default
    (fast) so it can process a whole multi-minute recording in a test suite."""
    source = FileReplaySource(recording_signals, channels, fs,
                              realtime_pace=realtime_pace, max_seconds=max_seconds)
    engine = DecisionEngine(model, smoothing_window=1, confidence_threshold=0.0)
    sink = WebSocketOutput()
    live_q = sink.subscribe()

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    decisions = []
    with sink:
        for decision in engine.run(source):
            sink.write(decision)
            decisions.append(decision)
        for decision in engine.flush():
            sink.write(decision)
            decisions.append(decision)
    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    assert live_q.qsize() == len(decisions)  # WebSocketOutput really received every one
    return decisions, wall_elapsed, cpu_elapsed


def test_online_chain_matches_offline_batch_validate(model):
    """The full FileReplaySource -> DecisionEngine chain and fast_causal_bci's
    own offline validate() must agree on every raw prediction they both make.

    validate()'s underlying feature extraction additionally filters out
    windows whose event label isn't clean/consistent (see
    extract_causal_features_from_source); DecisionEngine makes no such
    filtering (it predicts on every ready window, same as
    run_decision_source). So the two output sets aren't the same *size* -
    matching must be done by stream_time_s, not by position.
    """
    signals, channels, fs = load_recording(TEST_EDF, model.channels)
    decisions, _, _ = _run_online_full_chain(model, signals, channels, fs)

    online_by_time = {round(d.stream_time_s, 6): d.raw_label for d in decisions}

    batch = validate(model, TEST_EDF, TEST_EVENTS)
    assert batch["n_windows"] > 0

    checked = 0
    for t, batch_pred in zip(batch["times"], batch["predictions"]):
        key = round(float(t), 6)
        assert key in online_by_time, f"validate() window at t={key}s has no online counterpart"
        assert online_by_time[key] == int(batch_pred), f"mismatch at t={key}s"
        checked += 1

    assert checked == batch["n_windows"]
    print(f"\n[e2e] {checked} offline-batch windows all matched their online counterpart "
          f"(online chain produced {len(decisions)} total windows).")


def test_realtime_pacing_matches_wall_clock(model):
    """A realtime_pace=True replay must take approximately real wall-clock
    time, not process the data instantly - proof this is a genuine
    real-time simulation, not just a fast-forwarded computation."""
    signals, channels, fs = load_recording(TEST_EDF, model.channels)
    max_seconds = 8.0

    decisions, wall_elapsed, _ = _run_online_full_chain(
        model, signals, channels, fs, realtime_pace=True, max_seconds=max_seconds
    )

    assert len(decisions) > 0
    # Generous tolerance for CI/shared-machine scheduling jitter, but tight
    # enough that "instant" (wall_elapsed << max_seconds) would still fail.
    assert max_seconds * 0.85 <= wall_elapsed <= max_seconds * 1.5, (
        f"expected ~{max_seconds}s wall-clock for a {max_seconds}s real-time-paced "
        f"replay, got {wall_elapsed:.2f}s"
    )
    print(f"\n[e2e] {max_seconds}s real-time-paced replay took {wall_elapsed:.2f}s wall-clock.")


def test_cpu_cost_has_realtime_headroom(model):
    """Full-recording, unpaced (as fast as possible) run: total CPU time
    spent must be a small fraction of the recording's real duration, proving
    this pipeline could keep up with a live 1x-speed stream with room to
    spare - and writes the measured numbers to REALTIME_E2E.md's companion
    JSON for the report."""
    signals, channels, fs = load_recording(TEST_EDF, model.channels)
    recording_duration_s = min(len(signals[ch]) for ch in channels) / fs

    decisions, wall_elapsed, cpu_elapsed = _run_online_full_chain(model, signals, channels, fs)

    assert len(decisions) > 0
    cpu_fraction_of_realtime = cpu_elapsed / recording_duration_s
    assert cpu_fraction_of_realtime < 0.5, (
        f"CPU time ({cpu_elapsed:.2f}s) is {cpu_fraction_of_realtime:.1%} of the "
        f"recording's real duration ({recording_duration_s:.1f}s) - too close to "
        "1x to call this real-time-capable with headroom"
    )

    end_to_end_ms = np.array([d.end_to_end_ms for d in decisions])
    total_latency_ms = np.array([d.total_latency_ms for d in decisions])
    report = {
        "recording_duration_s": recording_duration_s,
        "n_decisions": len(decisions),
        "unpaced_wall_elapsed_s": wall_elapsed,
        "unpaced_cpu_elapsed_s": cpu_elapsed,
        "cpu_fraction_of_realtime": cpu_fraction_of_realtime,
        "end_to_end_ms": {
            "p50": float(np.percentile(end_to_end_ms, 50)),
            "p95": float(np.percentile(end_to_end_ms, 95)),
            "p99": float(np.percentile(end_to_end_ms, 99)),
            "max": float(end_to_end_ms.max()),
        },
        "total_latency_ms": {
            "p50": float(np.percentile(total_latency_ms, 50)),
            "p95": float(np.percentile(total_latency_ms, 95)),
            "p99": float(np.percentile(total_latency_ms, 99)),
            "max": float(total_latency_ms.max()),
        },
    }
    report_path = os.path.join(THIS_DIR, "..", "results", "realtime_e2e_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[e2e] CPU cost report: {json.dumps(report, indent=2)}")
