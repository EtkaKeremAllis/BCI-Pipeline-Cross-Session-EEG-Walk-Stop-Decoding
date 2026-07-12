def test_window_size_1_is_noop(bp):
    raw = [0, 1, 2, 0, 1]
    assert bp.apply_temporal_smoothing(raw, window_size=1) == raw


def test_empty_input(bp):
    assert bp.apply_temporal_smoothing([], window_size=3) == []


def test_majority_vote_clear_winner(bp):
    # window=3, centered: at i=2 the window is [1,2,3] -> indices 1,2,3 = [0,0,1] -> majority 0
    raw = [1, 0, 0, 1, 1]
    smoothed = bp.apply_temporal_smoothing(raw, window_size=3)
    assert smoothed[2] == 0


def test_tie_keeps_raw_prediction_no_walk_priority(bp):
    # window=3 centered at i=1: indices [0,1,2] = [0, 1, 2] -> STOP/WALK/IDLE all count 1 -> 3-way tie
    # must keep the raw value at i=1 (WALK=1), not silently prefer WALK for any other reason
    raw = [0, 1, 2]
    smoothed = bp.apply_temporal_smoothing(raw, window_size=3)
    assert smoothed[1] == raw[1] == 1

    # Same tie structure but raw center value is STOP (0) - must NOT be overridden to WALK.
    raw2 = [1, 0, 2]
    smoothed2 = bp.apply_temporal_smoothing(raw2, window_size=3)
    assert smoothed2[1] == raw2[1] == 0


def test_edges_use_truncated_window(bp):
    # at i=0 with window_size=3, half=1: window is raw[0:2] (no negative index), not padded
    raw = [0, 0, 1, 1, 1]
    smoothed = bp.apply_temporal_smoothing(raw, window_size=3)
    assert smoothed[0] == 0  # window [0,0] -> majority 0
    assert smoothed[-1] == 1  # window [1,1] -> majority 1


def test_window_size_5_wider_smoothing(bp):
    raw = [0, 0, 1, 0, 0, 0, 0]
    smoothed = bp.apply_temporal_smoothing(raw, window_size=5)
    # the single WALK=1 spike surrounded by STOP=0 should be smoothed away
    assert smoothed[2] == 0


def test_output_length_matches_input(bp):
    raw = [0, 1, 2, 0, 1, 2, 0]
    for ws in (1, 3, 5):
        assert len(bp.apply_temporal_smoothing(raw, window_size=ws)) == len(raw)
