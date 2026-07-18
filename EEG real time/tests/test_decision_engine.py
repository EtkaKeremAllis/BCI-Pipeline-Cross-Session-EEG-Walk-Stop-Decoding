"""
Verify DecisionEngine (realtime/decision_engine.py) against real data and
against fast_causal_bci's existing, untouched run_decision_source().

Run from the "EEG real time" directory:
    pytest tests/test_decision_engine.py -v

Requires the real sub-02 recordings and a trained model to already be in
the repo (sub-02/ses-02/eeg/... at the repo root, models/ses-01-to-ses-02
in this folder) - both already exist from prior slices, so no extra setup.
"""
import csv
import os
import sys
import tempfile

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_REALTIME_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(EEG_REALTIME_DIR)

if EEG_REALTIME_DIR not in sys.path:
    sys.path.insert(0, EEG_REALTIME_DIR)

from fast_causal_bci import FastCausalModel, LABELS, load_recording, run_decision_source  # noqa: E402
from realtime.decision_engine import DecisionEngine, IDLE_LABEL  # noqa: E402
from realtime.eeg_source import EEGSource  # noqa: E402
from realtime.file_replay_source import FileReplaySource  # noqa: E402

MODEL_DIR = os.path.join(EEG_REALTIME_DIR, "models", "ses-01-to-ses-02")
TEST_EDF = os.path.join(REPO_ROOT, "sub-02", "ses-02", "eeg",
                         "sub-02_ses-02_task-training_eeg.edf")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_DIR) and os.path.exists(TEST_EDF)),
    reason="real model/recording not present in this checkout",
)


@pytest.fixture(scope="module")
def model():
    return FastCausalModel.load(MODEL_DIR)


@pytest.fixture(scope="module")
def recording(model):
    signals, channels, fs = load_recording(TEST_EDF, model.channels)
    return signals, channels, fs


def _replay_source(recording):
    signals, channels, fs = recording
    return FileReplaySource(signals, channels, fs, realtime_pace=False)


def test_source_satisfies_eeg_source_protocol(recording):
    assert isinstance(_replay_source(recording), EEGSource)


def test_matches_existing_run_decision_source(model, recording):
    """With smoothing_window=1 (no smoothing) and confidence_threshold=0.0
    (no gating), DecisionEngine must reproduce fast_causal_bci's own,
    already-verified run_decision_source() exactly - same raw label and
    confidence for every decision."""
    with tempfile.TemporaryDirectory() as tmp:
        ref_csv = os.path.join(tmp, "ref.csv")
        run_decision_source(model, _replay_source(recording), output_csv=ref_csv, events=None)
        with open(ref_csv) as f:
            ref_rows = list(csv.DictReader(f))

    engine = DecisionEngine(model, smoothing_window=1, confidence_threshold=0.0)
    decisions = list(engine.run(_replay_source(recording)))

    assert len(decisions) == len(ref_rows)
    for ref_row, decision in zip(ref_rows, decisions):
        assert LABELS[decision.raw_label] == ref_row["prediction"]
        assert abs(float(ref_row["confidence"]) - decision.confidence) < 1e-9


@pytest.mark.parametrize("smoothing_window", [1, 3, 5])
def test_smoothed_stream_is_fully_resolved_after_flush(model, recording, smoothing_window):
    engine = DecisionEngine(model, smoothing_window=smoothing_window, confidence_threshold=0.0)
    decisions = list(engine.run(_replay_source(recording)))
    tail = engine.flush()

    n_pending_at_end = sum(1 for d in decisions if d.smoothed_label is None)
    assert n_pending_at_end == smoothing_window // 2
    assert len(tail) == n_pending_at_end


def test_extreme_confidence_threshold_gates_most_decisions_to_idle(model, recording):
    lenient = DecisionEngine(model, smoothing_window=1, confidence_threshold=0.0)
    n_idle_lenient = sum(1 for d in lenient.run(_replay_source(recording)) if d.raw_label == IDLE_LABEL)
    assert n_idle_lenient == 0  # a binary classifier's confidence is always >= 0.5

    strict = DecisionEngine(model, smoothing_window=1, confidence_threshold=0.99)
    decisions_strict = list(strict.run(_replay_source(recording)))
    n_idle_strict = sum(1 for d in decisions_strict if d.raw_label == IDLE_LABEL)
    assert n_idle_strict > len(decisions_strict) * 0.5  # most decisions should be gated


def test_channel_mismatch_raises(model, recording):
    signals, channels, fs = recording
    wrong_order = FileReplaySource(signals, list(reversed(channels)), fs)
    with pytest.raises(ValueError, match="Channel"):
        list(DecisionEngine(model).run(wrong_order))


def test_sampling_rate_mismatch_raises(model, recording):
    signals, channels, fs = recording
    wrong_rate = FileReplaySource(signals, channels, fs * 2)
    with pytest.raises(ValueError, match="Sampling-rate"):
        list(DecisionEngine(model).run(wrong_rate))
