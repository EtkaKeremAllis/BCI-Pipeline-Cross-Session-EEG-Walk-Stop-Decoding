import numpy as np


def _make_trial(walk, rng, n_samples=300, fs=100):
    """Synthetic epoch: WALK trials carry a clear 10Hz oscillation on top of
    the baseline noise that STOP trials consist of, so CSP+LDA has an
    unambiguous, learnable signal to separate on."""
    t = np.arange(n_samples) / fs
    c3 = rng.normal(0, 0.5, n_samples)
    c4 = rng.normal(0, 0.5, n_samples)
    if walk:
        c3 = c3 + 4.0 * np.sin(2 * np.pi * 10 * t)
        c4 = c4 + 4.0 * np.sin(2 * np.pi * 10 * t + 0.3)
    return {'C3': c3, 'C4': c4}


def _build_trained_model(bp, rng, idle_distance_threshold, confidence_threshold=0.45):
    model = bp.DeployableBCIModel(
        channels=['C3', 'C4'], sampling_rate=100,
        window_len=3.0, step_len=0.25,
        idle_distance_threshold=idle_distance_threshold,
        confidence_threshold=confidence_threshold,
        n_features_select=5, lda_shrinkage=0.0,
    )
    raw_trials, labels = [], []
    for _ in range(15):
        raw_trials.append(_make_trial(walk=False, rng=rng))
        labels.append(0)
        raw_trials.append(_make_trial(walk=True, rng=rng))
        labels.append(1)
    model.train_from_trials(raw_trials, np.array(labels))
    return model


def test_idle_gate_triggers_when_distance_threshold_is_unreachable(bp):
    # z_distance is a mean of absolute values, so it can never be negative -
    # a threshold of -1.0 makes the IDLE gate fire unconditionally.
    rng = np.random.default_rng(3)
    model = _build_trained_model(bp, rng, idle_distance_threshold=-1.0)

    walk_like = _make_trial(walk=True, rng=rng)
    label, confidence, z_distance, raw_pred = model.predict_window(walk_like, 0, 300)

    assert label == 2
    assert confidence is None
    assert raw_pred is None
    assert z_distance > -1.0


def test_predict_window_classifies_correctly_when_gates_are_open(bp):
    rng = np.random.default_rng(4)
    model = _build_trained_model(
        bp, rng, idle_distance_threshold=1e6, confidence_threshold=0.5
    )

    walk_like = _make_trial(walk=True, rng=rng)
    stop_like = _make_trial(walk=False, rng=rng)

    label_walk, conf_walk, _, raw_walk = model.predict_window(walk_like, 0, 300)
    label_stop, conf_stop, _, raw_stop = model.predict_window(stop_like, 0, 300)

    assert label_walk == 1
    assert raw_walk == 1
    assert conf_walk > 0.5

    assert label_stop == 0
    assert raw_stop == 0
    assert conf_stop > 0.5


def test_idle_gate_triggers_on_unreachable_confidence_threshold(bp):
    # predict_proba's output is a probability, mathematically bounded to
    # (0, 1) - a threshold above 1.0 is guaranteed to force the IDLE path
    # regardless of how separable the synthetic classes happen to be
    # (unlike e.g. 0.999999, which very confident models can still exceed).
    rng = np.random.default_rng(5)
    model = _build_trained_model(
        bp, rng, idle_distance_threshold=1e6, confidence_threshold=1.5
    )

    walk_like = _make_trial(walk=True, rng=rng)
    label, confidence, z_distance, raw_pred = model.predict_window(walk_like, 0, 300)

    assert label == 2  # forced to IDLE by the confidence gate, not the distance gate
    assert confidence is not None
    assert confidence <= 1.0
    assert raw_pred in (0, 1)  # underlying raw classifier decision is still reported
