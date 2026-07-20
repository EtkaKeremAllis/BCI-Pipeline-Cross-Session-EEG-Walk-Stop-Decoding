"""
Verify the OutputSink implementations (realtime/output_sink.py), including
an integration run against real DecisionEngine output.

Run from the "EEG real time" directory:
    pytest tests/test_output_sink.py -v
"""
import csv
import io
import os
import queue
import sys

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_REALTIME_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(EEG_REALTIME_DIR)

if EEG_REALTIME_DIR not in sys.path:
    sys.path.insert(0, EEG_REALTIME_DIR)

from fast_causal_bci import FastCausalModel, load_recording  # noqa: E402
from realtime.decision_engine import Decision, DecisionEngine  # noqa: E402
from realtime.file_replay_source import FileReplaySource  # noqa: E402
from realtime.output_sink import (  # noqa: E402
    ConsoleOutput,
    CSVOutput,
    HardwareOutput,
    MultiOutput,
    WebSocketOutput,
)

MODEL_DIR = os.path.join(EEG_REALTIME_DIR, "models", "ses-01-to-ses-02")
TEST_EDF = os.path.join(REPO_ROOT, "sub-02", "ses-02", "eeg",
                         "sub-02_ses-02_task-training_eeg.edf")


def _sample_decision(smoothed_label=1):
    return Decision(
        stream_time_s=12.5,
        raw_label=1,
        confidence=0.83,
        smoothed_label=smoothed_label,
        feature_ms=2.1,
        decision_ms=0.05,
        end_to_end_ms=2.15,
        smoothing_wall_delay_ms=0.3,
        total_latency_ms=2.45,
    )


def test_console_output_writes_readable_line():
    stream = io.StringIO()
    with ConsoleOutput(stream=stream) as sink:
        sink.write(_sample_decision())
    line = stream.getvalue()
    assert "WALK" in line
    assert "12.50" in line
    assert "0.830" in line


def test_console_output_handles_unresolved_smoothed_label():
    # DecisionEngine itself never yields smoothed_label=None any more (see
    # decision_engine.py's pending-queue re-pairing) - this is a defensive
    # robustness check for any other OutputSink caller that might.
    stream = io.StringIO()
    with ConsoleOutput(stream=stream) as sink:
        sink.write(_sample_decision(smoothed_label=None))
    assert "smoothed=-" in stream.getvalue()


def test_csv_output_round_trips(tmp_path):
    path = tmp_path / "decisions.csv"
    decisions = [_sample_decision(smoothed_label=1), _sample_decision(smoothed_label=None)]
    with CSVOutput(path) as sink:
        for d in decisions:
            sink.write(d)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["raw_label"] == "WALK"
    assert rows[0]["smoothed_label"] == "WALK"
    assert rows[1]["smoothed_label"] == ""
    assert float(rows[0]["confidence"]) == pytest.approx(0.83)
    assert float(rows[0]["total_latency_ms"]) == pytest.approx(2.45)


def test_websocket_output_broadcasts_to_all_subscribers():
    sink = WebSocketOutput()
    q1 = sink.subscribe()
    q2 = sink.subscribe()

    sink.write(_sample_decision())

    msg1 = q1.get_nowait()
    msg2 = q2.get_nowait()
    assert msg1 == msg2
    assert msg1["raw_label"] == "WALK"
    with pytest.raises(queue.Empty):
        q1.get_nowait()


def test_websocket_output_unsubscribe_stops_delivery():
    sink = WebSocketOutput()
    q1 = sink.subscribe()
    q2 = sink.subscribe()
    sink.unsubscribe(q1)

    sink.write(_sample_decision())

    with pytest.raises(queue.Empty):
        q1.get_nowait()
    assert q2.get_nowait()["raw_label"] == "WALK"


def test_websocket_output_close_clears_subscribers():
    sink = WebSocketOutput()
    q = sink.subscribe()
    sink.close()
    assert q not in sink._subscribers


def test_multi_output_fans_out_to_all_sinks(tmp_path):
    console_stream = io.StringIO()
    csv_path = tmp_path / "multi.csv"
    with MultiOutput([ConsoleOutput(stream=console_stream), CSVOutput(csv_path)]) as sink:
        sink.write(_sample_decision())

    assert "WALK" in console_stream.getvalue()
    with open(csv_path, newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_hardware_output_is_an_unimplemented_placeholder():
    with pytest.raises(NotImplementedError):
        HardwareOutput()


@pytest.mark.skipif(
    not (os.path.exists(MODEL_DIR) and os.path.exists(TEST_EDF)),
    reason="real model/recording not present in this checkout",
)
def test_integration_with_real_decision_engine(tmp_path):
    model = FastCausalModel.load(MODEL_DIR)
    signals, channels, fs = load_recording(TEST_EDF, model.channels)
    source = FileReplaySource(signals, channels, fs, realtime_pace=False)

    engine = DecisionEngine(model, smoothing_window=3, confidence_threshold=0.0)
    ws_sink = WebSocketOutput()
    live_q = ws_sink.subscribe()
    csv_path = tmp_path / "session.csv"

    n_written = 0
    with MultiOutput([CSVOutput(csv_path), ws_sink]) as sink:
        for decision in engine.run(source):
            sink.write(decision)
            n_written += 1
        for decision in engine.flush():  # trailing pending tail, easy to forget
            sink.write(decision)
            n_written += 1

    assert n_written > 4000  # sanity: real sub-02/ses-02 recording is long
    with open(csv_path, newline="") as f:
        assert len(list(csv.DictReader(f))) == n_written
    assert live_q.qsize() == n_written
