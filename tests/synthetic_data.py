"""
Synthetic motor-imagery EEG data generation, for smoke-testing the full CLI
pipeline (train -> validate_timeline) without needing real EEG recordings.

The trial generator is adapted from the synthetic motor-imagery generator
that existed in modern_bci_v2.py in v0.2 of this project (git commit
28c3ed8), before the v0.3 refactor dropped it. It has been generalized to
an arbitrary channel list instead of a hardcoded C3/C4/Cz, and gained a
minimal EDF writer (the write-side counterpart to edf_reader.read_edf) so
the generated recording can be fed through the real CLI exactly as a real
recording would be.
"""
import numpy as np


def generate_motor_imagery_trial(label, channels, fs=100, duration=3.0, noise_std=0.4):
    """
    Generate one synthetic motor-imagery EEG trial.

    label=0 (STOP) -> ERD (mu-band, 8-12Hz, suppression) on the second half
                       of `channels`.
    label=1 (WALK) -> ERD on the first half of `channels` instead.

    Not real data - a controlled simulation producing a mu-band power
    difference between classes for the pipeline to learn, so that a trained
    model's behavior (does it collapse to one class? does validate_timeline
    report sane accuracy?) can be smoke-tested without real recordings.
    """
    t = np.arange(0, duration, 1 / fs)
    mid = max(1, len(channels) // 2)
    high_amp_channels = set(channels[:mid] if label == 1 else channels[mid:])

    trial = {}
    for ch in channels:
        amp = 0.7 if ch in high_amp_channels else 2.0
        phase = np.random.uniform(0, 2 * np.pi)
        trial[ch] = amp * np.sin(2 * np.pi * 10 * t + phase) + noise_std * np.random.randn(len(t))
    return trial


def generate_synthetic_recording(channels, fs=100, n_blocks_per_class=5,
                                  stop_duration=8.0, walk_duration=8.0, seed=None):
    """
    Build one continuous synthetic recording (STOP/WALK blocks back to back,
    like a real training-task session) plus its matching event list.

    Returns (signals, events):
      signals: dict[channel] -> 1D float array, concatenated across all blocks.
      events: list of (onset_seconds, duration_seconds, trial_type) tuples,
              using the same x5=STOP/x8=WALK scheme as DEFAULT_LABEL_MAP.
    """
    if seed is not None:
        np.random.seed(seed)

    signals = {ch: [] for ch in channels}
    events = []
    t_cursor = 0.0

    for _ in range(n_blocks_per_class):
        for label, trial_type, block_duration in ((0, 'x5', stop_duration), (1, 'x8', walk_duration)):
            trial = generate_motor_imagery_trial(label, channels, fs=fs, duration=block_duration)
            for ch in channels:
                signals[ch].append(trial[ch])
            events.append((t_cursor, block_duration, trial_type))
            t_cursor += block_duration

    signals = {ch: np.concatenate(arrs) for ch, arrs in signals.items()}
    return signals, events


def write_edf(path, signals, fs, patient_id='SYNTH TEST', recording_id='SYNTHETIC'):
    """
    Minimal single-record plain-EDF writer - the write-side counterpart to
    edf_reader.read_edf(). Only supports what this project's read_edf()
    supports (16-bit samples, one data record spanning the whole recording),
    matching the real dataset's own layout (see DATASET.md).
    """
    labels = list(signals.keys())
    n_signals = len(labels)
    n_samples = len(next(iter(signals.values())))
    duration = n_samples / fs

    def pad(value, n_bytes):
        return str(value).encode('ascii')[:n_bytes].ljust(n_bytes)

    n_header_bytes = 256 + n_signals * 256
    header = b''.join([
        pad('0', 8),
        pad(patient_id, 80),
        pad(recording_id, 80),
        pad('01.01.20', 8),
        pad('00.00.00', 8),
        pad(n_header_bytes, 8),
        pad('', 44),
        pad('1', 8),                    # n_records
        pad(f'{duration:.6g}', 8),      # record_duration
        pad(n_signals, 4),
    ])

    dig_min, dig_max = -32768, 32767
    phys_min, phys_max = -200.0, 200.0
    scale = (phys_max - phys_min) / (dig_max - dig_min)

    header += b''.join(pad(l, 16) for l in labels)
    header += b''.join(pad('AgAgCl', 80) for _ in labels)
    header += b''.join(pad('uV', 8) for _ in labels)
    header += b''.join(pad(phys_min, 8) for _ in labels)
    header += b''.join(pad(phys_max, 8) for _ in labels)
    header += b''.join(pad(dig_min, 8) for _ in labels)
    header += b''.join(pad(dig_max, 8) for _ in labels)
    header += b''.join(pad('', 80) for _ in labels)
    header += b''.join(pad(n_samples, 8) for _ in labels)
    header += b''.join(pad('', 32) for _ in labels)

    assert len(header) == n_header_bytes, (len(header), n_header_bytes)

    data = np.zeros((n_signals, n_samples), dtype='<i2')
    for i, label in enumerate(labels):
        phys = np.clip(np.asarray(signals[label], dtype=np.float64), phys_min, phys_max)
        digital = np.round((phys - phys_min) / scale + dig_min).astype('<i2')
        data[i] = digital

    with open(path, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())


def write_events_tsv(path, events):
    with open(path, 'w') as f:
        f.write('onset\tduration\ttrial_type\n')
        for onset, duration, trial_type in events:
            f.write(f'{onset}\t{duration}\t{trial_type}\n')
