"""
Consistency test: OnlineSmoother (realtime/online_smoothing.py) must produce
byte-for-byte the same output as the offline pipeline's
apply_temporal_smoothing(), given the same raw-label sequence, for every
supported window size (1, 3, 5) - see the module docstring in
online_smoothing.py for why this is expected to hold exactly, not just
approximately.

Run from the "EEG real time" directory:
    pytest tests/test_online_smoothing.py -v
"""
import glob
import importlib.util
import os
import random
import sys

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_REALTIME_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(EEG_REALTIME_DIR)

if EEG_REALTIME_DIR not in sys.path:
    sys.path.insert(0, EEG_REALTIME_DIR)

from realtime.online_smoothing import OnlineSmoother  # noqa: E402


def _load_offline_apply_temporal_smoothing():
    candidates = sorted(glob.glob(os.path.join(REPO_ROOT, "bci_pipeline_v*.py")))
    if not candidates:
        raise FileNotFoundError(f"No bci_pipeline_v*.py found in {REPO_ROOT}.")
    path = candidates[-1]  # highest version, same convention as tests/conftest.py
    spec = importlib.util.spec_from_file_location("bci_pipeline_offline_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_temporal_smoothing


@pytest.fixture(scope="module")
def apply_temporal_smoothing():
    return _load_offline_apply_temporal_smoothing()


def _run_online(raw_labels, window_size):
    smoother = OnlineSmoother(window_size)
    result = []
    for label in raw_labels:
        result.extend(smoother.push(label))
    result.extend(smoother.flush())
    return result


@pytest.mark.parametrize("window_size", [1, 3, 5])
@pytest.mark.parametrize("seed", range(15))
def test_online_matches_offline_random_sequences(apply_temporal_smoothing, window_size, seed):
    rng = random.Random(seed)
    n = rng.randint(0, 50)
    raw = [rng.choice([0, 1, 2]) for _ in range(n)]

    expected = apply_temporal_smoothing(raw, window_size)
    actual = _run_online(raw, window_size)

    assert actual == expected


@pytest.mark.parametrize("window_size", [1, 3, 5])
def test_online_matches_offline_empty(apply_temporal_smoothing, window_size):
    assert _run_online([], window_size) == apply_temporal_smoothing([], window_size)


@pytest.mark.parametrize("window_size", [1, 3, 5])
def test_online_matches_offline_single_element(apply_temporal_smoothing, window_size):
    for label in (0, 1, 2):
        assert _run_online([label], window_size) == apply_temporal_smoothing([label], window_size)


def test_online_matches_offline_forced_tie():
    # window_size=3, sequence 0,1,0,1,... forces a tie at every interior
    # index (one 0 and one 1 in the window, symmetric), which offline
    # resolves by keeping the raw prediction at that index.
    raw = [0, 1, 0, 1, 0, 1, 0, 1]
    apply_temporal_smoothing = _load_offline_apply_temporal_smoothing()
    assert _run_online(raw, 3) == apply_temporal_smoothing(raw, 3)


def test_invalid_window_size_rejected():
    for bad in (0, 2, 4, 6, -1):
        with pytest.raises(ValueError):
            OnlineSmoother(bad)


def test_push_returns_ready_labels_incrementally():
    # window_size=5 (half=2): index i is only ready once i+half+1 raw labels
    # have arrived, i.e. index 0 needs 3 pushes (indices 0,1,2 present),
    # after which every further push keeps pace one-for-one.
    smoother = OnlineSmoother(5)
    assert smoother.push(1) == []       # index 0 needs raw[0:3], only have raw[0:1]
    assert smoother.push(1) == []       # only have raw[0:2]
    assert smoother.push(1) == [1]      # index 0's window is raw[0:3] = [1,1,1]
    assert smoother.push(1) == [1]      # index 1's window is raw[0:4] = [1,1,1,1]
    assert smoother.push(1) == [1]      # index 2's window is raw[0:5] = [1,1,1,1,1]
    assert smoother.flush() == [1, 1]   # remaining indices 3, 4 (truncated windows)
