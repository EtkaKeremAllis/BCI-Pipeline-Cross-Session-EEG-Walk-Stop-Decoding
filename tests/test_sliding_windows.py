def test_exact_fit_no_remainder(bp):
    # signal_len exactly fits N windows with no leftover samples
    windows = bp.build_sliding_windows(signal_len=1000, fs=100, window_len=3.0, step_len=1.0)
    # window=300 samples, step=100 samples: starts at 0,100,...,700 (700+300=1000)
    assert [w['start_idx'] for w in windows] == list(range(0, 701, 100))
    assert all(w['end_idx'] - w['start_idx'] == 300 for w in windows)


def test_partial_trailing_window_is_dropped(bp):
    # 1050 samples: last full window would need to start at 750 (750+300=1050 fits),
    # but a window starting at 800 (800+300=1100) would not fit and must be dropped.
    windows = bp.build_sliding_windows(signal_len=1099, fs=100, window_len=3.0, step_len=1.0)
    assert windows[-1]['end_idx'] <= 1099
    assert windows[-1]['start_idx'] + 300 <= 1099


def test_signal_shorter_than_one_window_returns_empty(bp):
    windows = bp.build_sliding_windows(signal_len=100, fs=100, window_len=3.0, step_len=1.0)
    assert windows == []


def test_signal_length_exactly_one_window(bp):
    windows = bp.build_sliding_windows(signal_len=300, fs=100, window_len=3.0, step_len=1.0)
    assert len(windows) == 1
    assert windows[0]['start_idx'] == 0
    assert windows[0]['end_idx'] == 300


def test_start_end_time_match_fs(bp):
    windows = bp.build_sliding_windows(signal_len=1000, fs=100, window_len=3.0, step_len=2.5)
    for w in windows:
        assert w['start_time'] == w['start_idx'] / 100
        assert w['end_time'] == w['end_idx'] / 100


def test_production_defaults_step_smaller_than_window_overlap(bp):
    # production defaults: window_len=3.0s, step_len=0.25s @ 100Hz -> 300/25 samples
    windows = bp.build_sliding_windows(signal_len=1000, fs=100, window_len=3.0, step_len=0.25)
    assert windows[0]['start_idx'] == 0
    assert windows[1]['start_idx'] == 25
    # windows overlap heavily since step (25) << window (300)
    assert windows[1]['start_idx'] < windows[0]['end_idx']
