"""
End-to-end smoke test: train -> validate_timeline through the real CLI entry
points (run_train / run_validate_timeline), using only synthetic data
(tests/synthetic_data.py) - no real EEG recording required.

Run this after any significant change to the core pipeline
(the main bci_pipeline_v*.py script, modern_bci_v2.py, edf_reader.py, parse_events.py):

    pytest tests/test_smoke_synthetic_pipeline.py -v

It is slower than the other unit tests (trains a real model, writes/reads
real EDF files) but exercises the full file-I/O + train + predict path in
one shot, including a direct regression guard against the "model collapse"
bug documented in CHANGELOG.md (v3.0): a training/prediction window
distribution mismatch that used to make the model always predict a single
class.
"""
import json
import os

from synthetic_data import generate_synthetic_recording, write_edf, write_events_tsv


def test_synthetic_train_and_validate_end_to_end(bp, tmp_path):
    channels = ['C3', 'Cz', 'C4']  # matches the 'motor3' channel_set preset
    fs = 100

    train_signals, train_events = generate_synthetic_recording(
        channels, fs=fs, n_blocks_per_class=6, stop_duration=8.0, walk_duration=8.0, seed=1)
    test_signals, test_events = generate_synthetic_recording(
        channels, fs=fs, n_blocks_per_class=4, stop_duration=8.0, walk_duration=8.0, seed=2)

    train_edf = str(tmp_path / 'train.edf')
    train_tsv = str(tmp_path / 'train_events.tsv')
    test_edf = str(tmp_path / 'test.edf')
    test_tsv = str(tmp_path / 'test_events.tsv')
    write_edf(train_edf, train_signals, fs=fs)
    write_events_tsv(train_tsv, train_events)
    write_edf(test_edf, test_signals, fs=fs)
    write_events_tsv(test_tsv, test_events)

    model_dir = str(tmp_path / 'model')
    output_dir = str(tmp_path / 'validate_out')

    bp.run_train(
        train_edf, train_tsv, model_dir,
        channel_set='motor3', n_features_select=10,
    )
    assert os.path.exists(os.path.join(model_dir, 'model_info.json'))
    assert os.path.exists(os.path.join(model_dir, 'trained_model.npz'))

    bp.run_validate_timeline(test_edf, test_tsv, model_dir, output_dir)

    metrics_path = os.path.join(output_dir, 'timeline_metrics.json')
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Regression guard against the v3.0 "model collapse" bug (CHANGELOG.md):
    # training-window distribution vs. sliding-window prediction distribution
    # mismatch used to make the model always predict a single class.
    assert metrics['smoothed']['collapse_warning'] is False, metrics['smoothed']
    assert metrics['raw']['collapse_warning'] is False, metrics['raw']

    # The synthetic classes are clearly separable by design (mu-band power
    # asymmetry between channels), so the trained model should do
    # meaningfully better than chance, not merely avoid collapsing.
    assert metrics['balanced_accuracy'] > 0.6, metrics['balanced_accuracy']
