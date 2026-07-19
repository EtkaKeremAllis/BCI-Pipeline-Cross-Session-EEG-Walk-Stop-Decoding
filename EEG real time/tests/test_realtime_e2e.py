import csv
import os
import sys

import numpy as np
import pytest


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_REALTIME_DIR = os.path.dirname(THIS_DIR)
if EEG_REALTIME_DIR not in sys.path:
    sys.path.insert(0, EEG_REALTIME_DIR)

from fast_causal_bci import run_decision_source  # noqa: E402
from fast_causal_web_ui import AppState, HTML  # noqa: E402


class FakeSource:
    def __init__(self, n_chunks=4):
        self.channels = ["C3"]
        self.sampling_rate = 100.0
        self.last_lateness_ms = 0.0
        self.n_chunks = n_chunks

    def chunks(self, chunk_samples):
        for index in range(self.n_chunks):
            yield {"C3": np.arange(chunk_samples, dtype=float) + index}


class FakeModel:
    channels = ["C3"]
    fs = 100.0
    window_seconds = 0.05
    step_seconds = 0.05
    context_seconds = 0.05

    def __init__(self):
        self.labels = [0, 1, 1, 0]
        self.index = 0

    def predict_features(self, features):
        label = self.labels[self.index % len(self.labels)]
        self.index += 1
        probabilities = np.array([[0.8, 0.2] if label == 0 else [0.1, 0.9]])
        return np.array([label]), probabilities


def test_on_decision_matches_csv_records_in_order(tmp_path):
    callback_records = []
    output = tmp_path / "records.csv"

    summary = run_decision_source(
        FakeModel(), FakeSource(), output_csv=output,
        on_decision=lambda record: callback_records.append(record.copy()),
        smoothing_window=3,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        csv_records = list(csv.DictReader(handle))

    assert summary["n_decisions"] == len(callback_records) == len(csv_records) == 4
    assert [record["stream_time_s"] for record in callback_records] == [0.05, 0.1, 0.15, 0.2]
    assert [record["prediction"] for record in callback_records] == ["STOP", "WALK", "WALK", "STOP"]
    assert [record["smoothed_prediction"] for record in callback_records] == [
        row["smoothed_prediction"] for row in csv_records
    ]
    assert [str(record["stream_time_s"]) for record in callback_records] == [
        row["stream_time_s"] for row in csv_records
    ]
    for record in callback_records:
        assert record["smoothing_delay_ms"] == 50.0
        assert record["smoothing_wall_delay_ms"] >= 0.0
        assert record["total_latency_ms"] == pytest.approx(
            record["end_to_end_ms"] + record["smoothing_wall_delay_ms"]
        )
    assert "smoothing_wall_delay_ms" in summary


def test_stop_check_ends_source_before_next_chunk():
    callback_records = []
    summary = run_decision_source(
        FakeModel(), FakeSource(n_chunks=10),
        on_decision=callback_records.append,
        stop_check=lambda: len(callback_records) >= 2,
        smoothing_window=1,
    )

    assert summary["n_decisions"] == len(callback_records) == 2
    assert [record["stream_time_s"] for record in callback_records] == [0.05, 0.1]


def test_source_model_channel_and_sampling_rate_validation_is_inherited():
    source = FakeSource()
    source.channels = ["C4"]
    with pytest.raises(ValueError, match="Channel/order mismatch"):
        run_decision_source(FakeModel(), source)

    source = FakeSource()
    source.sampling_rate = 200.0
    with pytest.raises(ValueError, match="Sampling-rate mismatch"):
        run_decision_source(FakeModel(), source)


def test_app_state_callback_updates_live_snapshot():
    state = AppState()
    state.total = 2.0
    record = {"stream_time_s": 0.5, "prediction": "STOP",
              "smoothed_prediction": "WALK"}
    state._on_record(record)

    snapshot = state.snapshot()
    assert snapshot["latest"] == record
    assert snapshot["history"] == [record]
    assert snapshot["progress"] == 0.25
    assert "Smoothed" in HTML
    assert "total_latency_ms" in HTML
