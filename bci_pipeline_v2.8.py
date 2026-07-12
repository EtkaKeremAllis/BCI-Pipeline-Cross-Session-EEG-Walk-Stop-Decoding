"""
bci_pipeline_v2.8.py
====================
EEG Walk/Stop BCI - v2.8

Critical design-error fix from the brief:
  Training data (labeled event windows) != Prediction data (sliding windows
  over the entire continuous recording) -> distribution mismatch -> collapse.

This file separates four modes:
  train             : train and save a model from labeled event windows
  train_multi       : train and save a pooled model from multiple labeled recordings.

  validate_timeline : test a trained model with sliding windows on a FULLY
                      labeled session and compare it with the ground-truth timeline
  predict           : run inference on unlabeled raw EEG (no event file required)

The classifier is UNCHANGED: CSP + feature selection + shrinkage LDA
(reused directly from modern_bci_v2.py). No new ML classifier was added;
only normalization, an IDLE gate, model persistence, and the correct
confidence definition were added.

Scientific integrity note: This system uses a pretrained model to PREDICT
Walk/Stop command periods from EEG. It is NOT a proven decoder of pure
brain-internal walking intention. Performance depends on the training data,
artifacts, subject/session similarity, and deployment conditions. A model
trained on one subject/session may not generalize well to another.
"""
import argparse
import json
import logging
import os
import numpy as np

from edf_reader import read_edf
from parse_events import parse_events
from modern_bci_v2 import BCIConfig, ModernBCIPipeline, logger

EPS = 1e-8
LABEL_NAMES = {0: 'STOP', 1: 'WALK', 2: 'IDLE', -1: 'IGNORE'}
# Default trial_type -> label mapping (0=STOP, 1=WALK), used when --label-map
# is not given. Matches the "REX system" event scheme (x5=STOP, x8=WALK)
# used by the datasets this pipeline was originally built against.
DEFAULT_LABEL_MAP = {'x5': 0, 'x8': 1}


def resolve_label_map(label_map_arg):
    """
    Resolve the --label-map CLI value into a dict[str, int] mapping each
    events-file trial_type to 0 (STOP) or 1 (WALK).

    label_map_arg: None (-> DEFAULT_LABEL_MAP), a JSON object string
    (e.g. '{"x5":0,"x8":1}'), or a path to a JSON file with the same shape.
    Different datasets use different trial_type vocabularies, so this must
    stay configurable rather than hardcoded.
    """
    if not label_map_arg:
        return dict(DEFAULT_LABEL_MAP)

    if os.path.exists(label_map_arg):
        with open(label_map_arg) as f:
            raw = json.load(f)
    else:
        try:
            raw = json.loads(label_map_arg)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"--label-map must be a JSON object (e.g. '{{\"x5\":0,\"x8\":1}}') "
                f"or a path to a JSON file containing one. Got: {label_map_arg!r} ({e})"
            )

    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"--label-map must be a non-empty JSON object, got: {raw!r}")

    label_map = {}
    for trial_type, label in raw.items():
        if label not in (0, 1):
            raise SystemExit(
                f"--label-map values must be 0 (STOP) or 1 (WALK); got {trial_type!r}: {label!r}"
            )
        label_map[str(trial_type)] = int(label)
    return label_map


# ============================================================================
# CHANNEL SETS (for testing broader motor-area channel sets from the CLI
# without modifying the code)
# ============================================================================

CHANNEL_SETS = {
    'motor3': ['C3', 'Cz', 'C4'],
    'motor5': ['C3', 'C1', 'Cz', 'C2', 'C4'],
    'motor9': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'Cz', 'C4', 'CPz'],
    'motor13': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4',
                'CP3', 'CPz', 'CP4'],
}
# all_eeg: exclude channels whose names contain ANY of these substrings
# (case-insensitive).
_ALL_EEG_EXCLUDE_SUBSTRINGS = ['EOG', 'EMG', 'ECG', 'HEOG', 'VEOG', 'EXG',
                               'STATUS', 'TRIGGER', 'STIM']


def resolve_channels(signals, channel_set):
    """
    signals: output of read_edf() (dictionary mapping channel names to 1D arrays).
    channel_set: 'motor3' | 'motor5' | 'motor9' | 'motor13' | 'all_eeg'

    Returns: a list of selected channel names.

    For motor3/5/9/13: the complete fixed channel list must exist in the EDF.
    Missing channels raise an explicit error instead of silently producing an
    incorrect model_info.json.

    For all_eeg: all EDF channels in their original order, except channels
    containing EOG/EMG/ECG/etc.
    """
    if channel_set == 'all_eeg':
        selected = []
        for ch in signals.keys():
            upper = ch.upper()
            if any(exc in upper for exc in _ALL_EEG_EXCLUDE_SUBSTRINGS):
                continue
            selected.append(ch)
        if not selected:
            raise SystemExit(
                "channel_set=all_eeg: no channels other than EOG/EMG/ECG/etc. "
                "were found; check the EDF channel names."
            )
        return selected

    if channel_set not in CHANNEL_SETS:
        raise ValueError(
            f"Unknown channel_set: {channel_set!r}. "
            f"Valid options: {list(CHANNEL_SETS) + ['all_eeg']}"
        )

    wanted = CHANNEL_SETS[channel_set]

    # Exact match first; fall back to a case-insensitive match against the
    # EDF's actual channel names (some EEG systems export e.g. "CZ" instead
    # of "Cz"). The real channel name (with the EDF's original casing) is
    # what gets returned, since that's the key used for all later signals[]
    # lookups.
    case_insensitive_lookup = {ch.upper(): ch for ch in signals}
    resolved = []
    still_missing = []
    case_fallback_used = []
    for ch in wanted:
        if ch in signals:
            resolved.append(ch)
        elif ch.upper() in case_insensitive_lookup:
            actual = case_insensitive_lookup[ch.upper()]
            resolved.append(actual)
            case_fallback_used.append((ch, actual))
        else:
            still_missing.append(ch)

    if case_fallback_used:
        print(f"[*] Channel name case mismatch resolved via case-insensitive "
              f"fallback: {case_fallback_used}")

    if still_missing:
        raise SystemExit(
            f"Required channel(s) for channel_set={channel_set} are missing from the EDF: {still_missing}.\n"
            f"Available EDF channels: {list(signals.keys())}"
        )
    return resolved


# ============================================================================
# FEATURE NAMES (deterministic order for persistence)
# ============================================================================

STAT_FEATURE_NAMES = [
    'mean', 'std', 'var', 'min', 'max', 'range', 'rms', 'peak_to_peak',
    'kurtosis', 'skewness', 'line_length', 'hjorth_activity',
    'hjorth_mobility', 'hjorth_complexity', 'zero_crossings',
    'spectral_entropy', 'Delta_power', 'Theta_power', 'Alpha_power',
    'Mu_power', 'Beta_power', 'Gamma_power', 'mu_beta_ratio',
    'beta_mu_ratio', 'alpha_theta_ratio', 'spectral_centroid', 'total_power'
]


def get_feature_names(channels, n_csp_filters=0):
    names = []
    for ch in sorted(channels):
        for f in STAT_FEATURE_NAMES:
            names.append(f"{ch}_{f}")
    sorted_ch = sorted(channels)
    names.append(f"{sorted_ch[0]}_{sorted_ch[1]}_asymmetry" if len(sorted_ch) > 1 else "asymmetry")
    for i in range(n_csp_filters):
        names.append(f"CSP_logvar_{i}")
    return names


def apply_temporal_smoothing(raw_labels, window_size):
    """
    Deterministic, neutral temporal smoothing.

    raw_labels: list/array of ints (0=STOP, 1=WALK, 2=IDLE)
    window_size: 1, 3, or 5. 1 means no smoothing.

    Centered majority vote is applied. If there is a tie, the current
    window's raw prediction is preserved. There is no WALK priority.
    """
    raw_labels = [int(x) for x in raw_labels]
    n = len(raw_labels)
    if window_size <= 1 or n == 0:
        return list(raw_labels)

    half = window_size // 2
    smoothed_prediction = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = raw_labels[start:end]

        counts = {}
        for lbl in window:
            counts[lbl] = counts.get(lbl, 0) + 1

        max_count = max(counts.values())
        candidates = [lbl for lbl, count in counts.items() if count == max_count]

        if len(candidates) == 1:
            smoothed_prediction.append(candidates[0])
        else:
            # Tie -> keep raw prediction. No WALK priority.
            smoothed_prediction.append(raw_labels[i])

    return smoothed_prediction


# ============================================================================
# SLIDING WINDOWS (deployment style over the full recording)
# ============================================================================

def build_sliding_windows(signal_len, fs, window_len=3.0, step_len=0.25):
    """
    Builds overlapping sliding windows over the full recording (deployment style).
    MUST NOT be confused with event-anchored labeled training windows.
    """
    window_samples = int(window_len * fs)
    step_samples = int(step_len * fs)
    windows = []
    start = 0
    while start + window_samples <= signal_len:
        end = start + window_samples
        windows.append({
            'start_idx': start, 'end_idx': end,
            'start_time': start / fs, 'end_time': end / fs
        })
        start += step_samples
    return windows


def label_windows_from_events(windows, events, overlap_threshold=0.5, label_map=None):
    """
    Labels the ground-truth timeline (used ONLY by validate_timeline).
    events: list of (onset, duration, trial_type) tuples.
    label_map: dict[str, int] mapping trial_type -> 0 (STOP) or 1 (WALK);
    defaults to DEFAULT_LABEL_MAP.
    Returned label: 0=STOP, 1=WALK, 2=IDLE (ambiguous/no overlap)
    """
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP
    stop_intervals = [(o, o + d) for o, d, t in events if label_map.get(t) == 0]
    walk_intervals = [(o, o + d) for o, d, t in events if label_map.get(t) == 1]

    def overlap_duration(win_start, win_end, intervals):
        total = 0.0
        for s, e in intervals:
            total += max(0.0, min(win_end, e) - max(win_start, s))
        return total

    labels = []
    for w in windows:
        dur = w['end_time'] - w['start_time']
        stop_ov = overlap_duration(w['start_time'], w['end_time'], stop_intervals)
        walk_ov = overlap_duration(w['start_time'], w['end_time'], walk_intervals)
        stop_frac, walk_frac = stop_ov / dur, walk_ov / dur
        if walk_frac >= overlap_threshold and walk_frac >= stop_frac:
            labels.append(1)
        elif stop_frac >= overlap_threshold and stop_frac > walk_frac:
            labels.append(0)
        else:
            labels.append(2)  # IDLE (ambiguous/outside events)
    return np.array(labels)


def extract_command_training_windows(events, fs, skip_start=1.0, skip_end=1.0, window_len=3.0,
                                     label_map=None):
    """
    Event-ANCHORED, non-overlapping windows for training (STOP/WALK events ONLY,
    as defined by label_map).
    This is intentionally DIFFERENT from deployment sliding windows: training
    data always comes from labeled command intervals.
    """
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP
    windows = []
    for onset, duration, trial_type in events:
        if trial_type not in label_map:
            continue
        label = label_map[trial_type]
        usable_start = onset + skip_start
        usable_end = onset + duration - skip_end
        usable_duration = usable_end - usable_start
        n_windows = int(usable_duration // window_len)
        for w in range(max(n_windows, 0)):
            start_t = usable_start + w * window_len
            start_idx = int(start_t * fs)
            end_idx = start_idx + int(window_len * fs)
            windows.append({'start_idx': start_idx, 'end_idx': end_idx, 'label': label})
    return windows


# ============================================================================
# DEPLOYABLE MODEL: ModernBCIPipeline wrapper + normalization + IDLE gate
# ============================================================================

class DeployableBCIModel:
    """
    CSP + feature selection + shrinkage LDA (modern_bci_v2.ModernBCIPipeline)
    + feature normalization + IDLE gate + model persistence.

    There is NO new ML classifier; this is only the wrapper required for deployment.
    """

    def __init__(self, channels, sampling_rate, bandpass_low=0.5, bandpass_high=45,
                 fir_order=200, use_notch=False, notch_freq=60, use_laplacian=True,
                 use_csp=True, lda_shrinkage=0.0, n_features_select=45,
                 window_len=3.0, step_len=0.25,
                 confidence_threshold=0.45, idle_distance_threshold=999.0,
                 channel_set=None, channel_normalization='zscore', label_map=None):
        self.channels = list(channels)
        self.channel_set = channel_set  # informational only; selection was already handled by resolve_channels()
        self.channel_normalization = channel_normalization
        # trial_type -> 0 (STOP) / 1 (WALK) mapping used to train this model;
        # persisted so validate_timeline/predict reuse the same scheme automatically.
        self.label_map = dict(label_map) if label_map else dict(DEFAULT_LABEL_MAP)
        self.sampling_rate = sampling_rate
        self.window_len = window_len
        self.step_len = step_len
        self.confidence_threshold = confidence_threshold
        self.idle_distance_threshold = idle_distance_threshold

        self.config = BCIConfig(
            sampling_rate=sampling_rate, n_channels=len(channels), channels=self.channels,
            notch_freq=notch_freq, use_notch=use_notch,
            bandpass_low=bandpass_low, bandpass_high=bandpass_high, fir_order=fir_order,
            use_laplacian=use_laplacian, use_csp=use_csp,
            lda_shrinkage=lda_shrinkage, n_features_select=n_features_select,
            use_feature_selection=True,
        )
        self.pipeline = ModernBCIPipeline(self.config)

        # Populated after fitting:
        self.feature_mean = None
        self.feature_std = None
        self.command_feature_mean = None
        self.command_feature_std = None
        self.feature_names = None
        self.n_csp_filters = self.config.csp_n_filters if use_csp else 0

    def _apply_channel_normalization(self, raw_signals):
        """Normalize raw EEG channels per recording before bandpass filtering/CSP."""
        if self.channel_normalization == 'none':
            return {ch: raw_signals[ch] for ch in self.channels}

        if self.channel_normalization != 'zscore':
            raise ValueError(f"Unknown channel_normalization: {self.channel_normalization}")

        normalized = {}
        for ch in self.channels:
            x = np.asarray(raw_signals[ch], dtype=np.float64)
            mean = float(np.mean(x))
            std = max(float(np.std(x)), EPS)
            normalized[ch] = (x - mean) / std
        return normalized

    def preprocess_continuous(self, raw_signals):
        """Preprocess the full continuous recording ONCE, not per window."""
        normalized_signals = self._apply_channel_normalization(raw_signals)
        return self.pipeline.preprocess(normalized_signals)

    def _epoch_dict(self, preprocessed, start_idx, end_idx):
        return {ch: preprocessed[ch][start_idx:end_idx] for ch in self.channels}

    def train(self, preprocessed_continuous, train_windows):
        """
        preprocessed_continuous: output of preprocess_continuous()
        train_windows: output of extract_command_training_windows(), including labels
        """
        raw_trials = [self._epoch_dict(preprocessed_continuous, w['start_idx'], w['end_idx'])
                      for w in train_windows]
        y = np.array([w['label'] for w in train_windows])
        return self.train_from_trials(raw_trials, y)

    def train_from_trials(self, raw_trials, y):
        """
        Shared core of single-session train(), now accepting DIRECTLY epoched
        raw_trials (a list of eeg_dict objects) plus y, without depending on
        indices into one continuous recording. This is required for multi-session
        pooling (train_multi): each session is preprocessed and epoched separately
        within its OWN continuous recording, then the epoch DICTIONARIES
        (independent arrays that no longer carry recording-time indices) are pooled
        into one list and passed here.
        """
        y = np.asarray(y)

        # 1) Fit CSP, feature selection, and an initial throwaway classifier
        self.pipeline.train(raw_trials, y)

        # 2) CSP and feature_selector are now fitted; extract selected features
        #    at their raw scale for all training windows
        X_raw = np.array([self.pipeline.extract_full_features(t) for t in raw_trials])
        X_sel_raw = self.pipeline.feature_selector.transform(X_raw)

        # 3) Normalization statistics in the selected-feature space
        self.feature_mean = X_sel_raw.mean(axis=0)
        self.feature_std = X_sel_raw.std(axis=0) + EPS
        X_sel_norm = (X_sel_raw - self.feature_mean) / self.feature_std

        # 4) Retrain the classifier with NORMALIZED features
        self.pipeline.classifier.fit(X_sel_norm, y)

        # 5) IDLE-gate reference distribution (command-window feature distribution)
        self.command_feature_mean = X_sel_norm.mean(axis=0)
        self.command_feature_std = X_sel_norm.std(axis=0) + EPS

        self.feature_names = get_feature_names(self.channels, self.n_csp_filters)

        return {
            'n_train_windows': len(raw_trials),
            'n_stop': int(np.sum(y == 0)), 'n_walk': int(np.sum(y == 1)),
            'selected_feature_idx': self.pipeline.feature_selector.selected_idx.tolist(),
        }

    def predict_window(self, preprocessed_continuous, start_idx, end_idx):
        """
        For one window: IDLE gate -> LDA if accepted -> corrected confidence.
        Returns: (label:int[0/1/2], confidence:float or None, z_distance:float, raw_pred_or_None)
        """
        epoch = self._epoch_dict(preprocessed_continuous, start_idx, end_idx)
        x_raw = self.pipeline.extract_full_features(epoch).reshape(1, -1)
        x_sel_raw = self.pipeline.feature_selector.transform(x_raw)[0]
        x_sel_norm = (x_sel_raw - self.feature_mean) / self.feature_std

        z_distance = float(np.mean(np.abs(
            (x_sel_norm - self.command_feature_mean) / self.command_feature_std
        )))

        if z_distance > self.idle_distance_threshold:
            return 2, None, z_distance, None  # IDLE gate triggered

        proba = self.pipeline.classifier.predict_proba(x_sel_norm.reshape(1, -1))[0]
        pred = int(self.pipeline.classifier.predict(x_sel_norm.reshape(1, -1))[0])
        # FIX: confidence = P(predicted class), not always P(WALK)
        confidence = float(proba[pred])

        if confidence < self.confidence_threshold:
            return 2, confidence, z_distance, pred  # low confidence -> IDLE

        label = 1 if pred == 1 else 0
        return label, confidence, z_distance, pred

    # ---------------- persistence ----------------

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        np.savez(
            os.path.join(output_dir, 'trained_model.npz'),
            lda_coef=self.pipeline.classifier.coef,
            lda_intercept=self.pipeline.classifier.intercept,
            lda_mean_0=self.pipeline.classifier.mean_0,
            lda_mean_1=self.pipeline.classifier.mean_1,
            selected_feature_idx=self.pipeline.feature_selector.selected_idx,
            csp_filters=(self.pipeline.csp.filters if self.pipeline.csp is not None
                         and self.pipeline.csp.filters is not None else np.array([])),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            command_feature_mean=self.command_feature_mean,
            command_feature_std=self.command_feature_std,
        )

        info = {
            'channels': self.channels,
            'channel_set': self.channel_set,
            'channel_normalization': self.channel_normalization,
            'sampling_rate': self.sampling_rate,
            'window_len': self.window_len,
            'step_len': self.step_len,
            'bandpass_low': self.config.bandpass_low,
            'bandpass_high': self.config.bandpass_high,
            'fir_order': self.config.fir_order,
            'use_notch': self.config.use_notch,
            'notch_freq': self.config.notch_freq,
            'use_laplacian': self.config.use_laplacian,
            'use_csp': self.config.use_csp,
            'n_csp_filters': self.n_csp_filters,
            'lda_shrinkage': self.config.lda_shrinkage,
            'n_features_select': self.config.n_features_select,
            'confidence_threshold': self.confidence_threshold,
            'idle_distance_threshold': self.idle_distance_threshold,
            'label_map': {'STOP': 0, 'WALK': 1, 'IDLE': 2},
            'command_label_map_source': self.label_map,
        }
        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        with open(os.path.join(output_dir, 'selected_features.json'), 'w') as f:
            sel_idx = self.pipeline.feature_selector.selected_idx.tolist()
            names = [self.feature_names[i] for i in sel_idx]
            json.dump({'selected_indices': sel_idx, 'selected_names': names}, f, indent=2)

        return info

    @classmethod
    def load(cls, model_dir):
        with open(os.path.join(model_dir, 'model_info.json')) as f:
            info = json.load(f)
        arrs = np.load(os.path.join(model_dir, 'trained_model.npz'))

        model = cls(
            channels=info['channels'], sampling_rate=info['sampling_rate'],
            bandpass_low=info['bandpass_low'], bandpass_high=info['bandpass_high'],
            fir_order=info['fir_order'], use_notch=info['use_notch'],
            notch_freq=info['notch_freq'], use_laplacian=info['use_laplacian'],
            use_csp=info['use_csp'], lda_shrinkage=info['lda_shrinkage'],
            n_features_select=info['n_features_select'],
            window_len=info['window_len'], step_len=info['step_len'],
            confidence_threshold=info['confidence_threshold'],
            idle_distance_threshold=info['idle_distance_threshold'],
            # Backward compatible: older models do not contain this field -> None.
            # The channels list itself is already loaded correctly; channel_set only
            # records which preset it came from.
            channel_set=info.get('channel_set'),
            channel_normalization=info.get('channel_normalization', 'none'),
            # Backward compatible: older models were always trained with the
            # REX-system x5/x8 scheme (DEFAULT_LABEL_MAP), so fall back to it.
            label_map=info.get('command_label_map_source'),
        )

        # Restore fitted state (CSP, feature_selector, classifier)
        model.pipeline.classifier.coef = arrs['lda_coef']
        model.pipeline.classifier.intercept = arrs['lda_intercept']
        model.pipeline.classifier.mean_0 = arrs['lda_mean_0']
        model.pipeline.classifier.mean_1 = arrs['lda_mean_1']

        model.pipeline.feature_selector.selected_idx = arrs['selected_feature_idx']

        csp_filters = arrs['csp_filters']
        if model.pipeline.csp is not None and csp_filters.size > 0:
            model.pipeline.csp.filters = csp_filters

        model.feature_mean = arrs['feature_mean']
        model.feature_std = arrs['feature_std']
        model.command_feature_mean = arrs['command_feature_mean']
        model.command_feature_std = arrs['command_feature_std']
        model.feature_names = get_feature_names(model.channels, model.n_csp_filters)

        return model


# ============================================================================
# CLASS BALANCING
# ============================================================================

def downsample_classes(items, y, seed=42):
    """
    Reproducibly downsample the majority class (usually STOP) at random to the
    size of the minority class (usually WALK), using the provided seed.

    items: any list in the same order as y (window dictionaries OR already
           epoched eeg_dict objects; contents are not inspected).
    y:     a 0/1 label array with the same length as items.

    Returns: (items_balanced, y_balanced, stats)
    stats: kept/dropped counts plus the seed, written to training_summary.txt.
    """
    y = np.asarray(y)
    idx_stop = np.where(y == 0)[0]
    idx_walk = np.where(y == 1)[0]
    n_stop_before, n_walk_before = len(idx_stop), len(idx_walk)

    if n_stop_before == 0 or n_walk_before == 0:
        logger.warning("downsample_classes: one class is completely absent (STOP=%d, WALK=%d); "
                        "balancing is skipped.", n_stop_before, n_walk_before)
        stats = {
            'method': 'downsample', 'applied': False, 'seed': seed,
            'n_stop_before': n_stop_before, 'n_walk_before': n_walk_before,
            'n_stop_after': n_stop_before, 'n_walk_after': n_walk_before,
            'n_kept': n_stop_before + n_walk_before, 'n_dropped': 0,
        }
        return list(items), y, stats

    n_target = min(n_stop_before, n_walk_before)
    rng = np.random.default_rng(seed)

    if n_stop_before > n_target:
        idx_stop = rng.choice(idx_stop, size=n_target, replace=False)
    if n_walk_before > n_target:
        idx_walk = rng.choice(idx_walk, size=n_target, replace=False)

    keep_idx = np.sort(np.concatenate([idx_stop, idx_walk]))
    items_balanced = [items[i] for i in keep_idx]
    y_balanced = y[keep_idx]

    n_before_total = n_stop_before + n_walk_before
    n_after_total = len(keep_idx)
    stats = {
        'method': 'downsample', 'applied': True, 'seed': int(seed),
        'n_stop_before': int(n_stop_before), 'n_walk_before': int(n_walk_before),
        'n_stop_after': int(np.sum(y_balanced == 0)), 'n_walk_after': int(np.sum(y_balanced == 1)),
        'n_kept': int(n_after_total), 'n_dropped': int(n_before_total - n_after_total),
    }
    return items_balanced, y_balanced, stats


def apply_class_balancing(items, y, balance_classes, seed):
    """
    balance_classes: 'none' | 'downsample' | 'class_weight'
    'class_weight' is not implemented yet; raise an explicit error instead of
    silently ignoring it and creating a false impression.
    """
    if balance_classes == 'none':
        y = np.asarray(y)
        return list(items), y, {
            'method': 'none', 'applied': False, 'seed': seed,
            'n_stop_before': int(np.sum(y == 0)), 'n_walk_before': int(np.sum(y == 1)),
            'n_stop_after': int(np.sum(y == 0)), 'n_walk_after': int(np.sum(y == 1)),
            'n_kept': len(y), 'n_dropped': 0,
        }
    elif balance_classes == 'downsample':
        return downsample_classes(items, y, seed=seed)
    elif balance_classes == 'class_weight':
        raise NotImplementedError(
            "--balance-classes class_weight is not implemented yet. "
            "Use --balance-classes downsample for now."
        )
    else:
        raise ValueError(f"Unknown balance_classes value: {balance_classes}")


def _format_balance_stats(stats):
    lines = [f"Class balancing: {stats['method']} (seed={stats['seed']})"]
    if stats['applied']:
        lines.append(f"  Before: STOP={stats['n_stop_before']}, WALK={stats['n_walk_before']}")
        lines.append(f"  After:  STOP={stats['n_stop_after']}, WALK={stats['n_walk_after']}")
        lines.append(f"  Kept: {stats['n_kept']}, Dropped: {stats['n_dropped']}")
    else:
        lines.append(f"  Not applied. STOP={stats['n_stop_before']}, WALK={stats['n_walk_before']}")
    return "\n".join(lines)


def summarize_subjects(trials, labels, seed=42, method='none', applied=False, target_per_subject=None):
    """
    Build per-subject counts from trial dictionaries carrying subject/session
    provenance for train_multi. Each trial must contain at least a subject key;
    labels are the corresponding 0/1 labels in the same order.
    """
    import collections
    labels = np.asarray(labels)
    subj_to_idx = collections.defaultdict(list)
    for i, t in enumerate(trials):
        subj_to_idx[t['subject']].append(i)

    per_subject = {}
    for subj in sorted(subj_to_idx):
        idx = np.asarray(subj_to_idx[subj], dtype=int)
        subj_labels = labels[idx]
        per_subject[subj] = {
            'n_before': int(len(idx)),
            'n_after': int(len(idx)),
            'stop_before': int(np.sum(subj_labels == 0)),
            'walk_before': int(np.sum(subj_labels == 1)),
            'stop_after': int(np.sum(subj_labels == 0)),
            'walk_after': int(np.sum(subj_labels == 1)),
        }

    return {
        'method': method,
        'applied': bool(applied),
        'seed': int(seed),
        'target_per_subject': None if target_per_subject is None else int(target_per_subject),
        'per_subject': per_subject,
    }


def downsample_subjects(trials, labels, seed=42):
    """
    Perform subject-level downsampling so every subject contributes the same
    number of trials. This is not class balancing; --balance-classes manages the
    STOP/WALK balance separately.
    """
    import collections
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    if len(trials) == 0:
        return list(trials), labels, summarize_subjects(
            trials, labels, seed=seed, method='downsample_subjects', applied=False
        )

    subj_to_idx = collections.defaultdict(list)
    for i, t in enumerate(trials):
        subj_to_idx[t['subject']].append(i)

    subj_counts = {s: len(idx) for s, idx in subj_to_idx.items()}
    target = min(subj_counts.values())

    kept_idx = []
    per_subject_stats = {}
    for subj in sorted(subj_to_idx):
        idx_arr = np.asarray(subj_to_idx[subj], dtype=int)
        if len(idx_arr) > target:
            chosen = rng.choice(idx_arr, size=target, replace=False)
        else:
            chosen = idx_arr
        chosen = np.sort(chosen)
        kept_idx.extend(chosen.tolist())

        before_labels = labels[idx_arr]
        after_labels = labels[chosen]
        per_subject_stats[subj] = {
            'n_before': int(len(idx_arr)),
            'n_after': int(len(chosen)),
            'stop_before': int(np.sum(before_labels == 0)),
            'walk_before': int(np.sum(before_labels == 1)),
            'stop_after': int(np.sum(after_labels == 0)),
            'walk_after': int(np.sum(after_labels == 1)),
        }

    kept_idx = np.sort(np.asarray(kept_idx, dtype=int))
    trials_bal = [trials[i] for i in kept_idx]
    labels_bal = labels[kept_idx]

    stats = {
        'method': 'downsample_subjects',
        'applied': True,
        'seed': int(seed),
        'target_per_subject': int(target),
        'per_subject': per_subject_stats,
    }
    return trials_bal, labels_bal, stats


def apply_subject_balancing(trials, labels, balance_subjects, seed=42):
    if balance_subjects == 'none':
        return list(trials), np.asarray(labels), summarize_subjects(
            trials, labels, seed=seed, method='none', applied=False
        )
    if balance_subjects == 'downsample':
        return downsample_subjects(trials, labels, seed=seed)
    raise ValueError(f"Unknown balance_subjects value: {balance_subjects}")


def _format_subject_balance_stats(stats):
    lines = [f"Subject balancing: {stats['method']} (seed={stats['seed']})"]
    if stats.get('applied'):
        lines.append(f"  Target trials per subject: {stats.get('target_per_subject')}")
    else:
        lines.append("  Not applied.")

    per_subject = stats.get('per_subject', {})
    for subj in sorted(per_subject):
        s = per_subject[subj]
        if not s:
            lines.append(f"  [{subj}] no stats")
            continue
        lines.append(
            f"  [{subj}] before={s['n_before']} "
            f"(STOP={s['stop_before']}, WALK={s['walk_before']}), "
            f"after={s['n_after']} "
            f"(STOP={s['stop_after']}, WALK={s['walk_after']})"
        )
    return "\n".join(lines)


def _json_default(o):
    """Small converter that makes NumPy values and NaN compatible with json.dump."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if np.isnan(o):
            return None
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ============================================================================
# MODE: train
# ============================================================================

def run_train(edf_path, events_path, output_dir, channel_set='motor3',
              window_len=3.0, step_len=0.25, idle_distance_threshold=999.0,
              confidence_threshold=0.45, n_features_select=45, lda_shrinkage=0.0,
              balance_classes='none', seed=42, channel_normalization='zscore',
              label_map=None):
    print("=" * 70)
    print("MODE: train")
    print("=" * 70)

    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    signals, info = read_edf(edf_path)

    channels = resolve_channels(signals, channel_set)
    print(f"[*] Channel set {channel_set}: {len(channels)} channels -> {channels}")
    print(f"[*] Channel normalization: {channel_normalization}")
    print(f"[*] Label map: {label_map}")

    events_all = parse_events(events_path)
    events = [(o, d, t) for o, d, t in events_all if t in label_map]
    dropped = [(o, d, t) for o, d, t in events_all if t not in label_map]
    if dropped:
        print(f"[*] Skipped {len(dropped)} events outside the label map (e.g. {dropped[:3]})")

    fs = info['sampling_rate'][channels[0]]
    print(f"[*] EDF: {len(signals[channels[0]])} samples @ {fs:.0f}Hz")
    print(f"[*] Events: {len(events)} usable")

    model = DeployableBCIModel(
        channels=channels, sampling_rate=fs, window_len=window_len, step_len=step_len,
        idle_distance_threshold=idle_distance_threshold,
        confidence_threshold=confidence_threshold,
        n_features_select=n_features_select, lda_shrinkage=lda_shrinkage,
        channel_set=channel_set,
        channel_normalization=channel_normalization,
        label_map=label_map,
    )

    preprocessed = model.preprocess_continuous(signals)
    train_windows = extract_command_training_windows(events, fs, window_len=window_len,
                                                       label_map=label_map)
    print(f"[*] Extracted {len(train_windows)} training windows "
          f"(event-anchored, non-overlapping, {window_len}s)")

    labels_arr = np.array([w['label'] for w in train_windows])
    train_windows, _, balance_stats = apply_class_balancing(
        train_windows, labels_arr, balance_classes, seed
    )
    if balance_stats['applied']:
        print(f"[*] Class balancing ({balance_stats['method']}): "
              f"STOP {balance_stats['n_stop_before']}->{balance_stats['n_stop_after']}, "
              f"WALK {balance_stats['n_walk_before']}->{balance_stats['n_walk_after']}, "
              f"dropped={balance_stats['n_dropped']} (seed={balance_stats['seed']})")

    stats = model.train(preprocessed, train_windows)
    print(f"[+] Training completed: STOP={stats['n_stop']}, WALK={stats['n_walk']}")
    print(f"[+] Selected feature indices: {stats['selected_feature_idx']}")

    info_saved = model.save(output_dir)
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("BCI Walk/Stop Model - Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"EDF: {edf_path}\n")
        f.write(f"Events: {events_path}\n")
        f.write(f"Channel set: {channel_set}\n")
        f.write(f"Channel normalization: {channel_normalization}\n")
        f.write(f"Label map: {label_map}\n")
        f.write(f"Channels: {channels}\n")
        f.write(f"Sampling rate: {fs}\n")
        f.write(f"Train windows: {len(train_windows)} (STOP={stats['n_stop']}, WALK={stats['n_walk']})\n")
        f.write(f"CSP: {info_saved['use_csp']}, n_features_select: {info_saved['n_features_select']}, "
                f"lda_shrinkage: {info_saved['lda_shrinkage']}\n")
        f.write(f"idle_distance_threshold: {idle_distance_threshold}, "
                f"confidence_threshold: {confidence_threshold}\n\n")
        f.write(_format_balance_stats(balance_stats) + "\n\n")
        f.write("SCIENTIFIC INTEGRITY: This model uses a pretrained model to predict\n")
        f.write("Walk/Stop command periods from EEG. It is not a proven decoder of pure\n")
        f.write("brain-internal walking intention. Performance depends on the training\n")
        f.write("data, artifacts, and subject/session similarity.\n")

    print(f"[+] Model saved: {output_dir}/")
    return model, signals, info, events


# ============================================================================
# MODE: train_multi (multi-session pooling)
# ============================================================================

def _resolve_path(path, dataset_dir):
    """Join a relative CSV path with dataset_dir; keep it unchanged if it is
    already resolvable as an absolute path or from the current working directory."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    joined = os.path.join(dataset_dir, path)
    if os.path.exists(joined):
        return joined
    return path  # return even if absent so the caller can raise an explicit error


def run_train_multi(dataset_list_csv, output_dir, dataset_dir='.',
                     channel_set='motor3', window_len=3.0, step_len=0.25,
                     idle_distance_threshold=999.0, confidence_threshold=0.45,
                     n_features_select=45, lda_shrinkage=0.0,
                     balance_classes='none', balance_subjects='none', seed=42,
                     channel_normalization='zscore', label_map=None):
    """
    Preprocess multiple (subject, session) recordings SEPARATELY and train ONE
    DeployableBCIModel by pooling ONLY labeled command windows.

    Additional safeguards:
      - Every trial carries subject/session provenance.
      - --balance-subjects downsample makes every subject contribute equally.
      - train_manifest_resolved.csv: all candidate windows before balancing.
      - train_manifest_used.csv: windows actually used for training.
    """
    import csv as csv_mod

    if label_map is None:
        label_map = DEFAULT_LABEL_MAP

    print("=" * 70)
    print("MODE: train_multi")
    print("=" * 70)

    with open(dataset_list_csv, newline='') as f:
        rows = list(csv_mod.DictReader(f))
    if not rows:
        raise SystemExit(f"{dataset_list_csv} is empty or could not be read")

    required_cols = {'subject', 'session', 'edf', 'events'}
    missing_cols = required_cols - set(rows[0].keys())
    if missing_cols:
        raise SystemExit(f"dataset-list CSV is missing column(s): {missing_cols}")

    print(f"[*] Found {len(rows)} (subject, session) rows: {dataset_list_csv}")
    print(f"[*] Channel normalization: {channel_normalization}")

    pooled_trials = []  # dict: epoch + provenance
    pooled_labels = []
    manifest_rows = []
    per_file_stats = []
    fs_reference = None
    channels = None  # resolved from the FIRST successfully loaded file, then kept fixed
    skipped_files = []

    for row in rows:
        subject, session = row['subject'], row['session']
        edf_path = _resolve_path(row['edf'], dataset_dir)
        events_path = _resolve_path(row['events'], dataset_dir)
        tag = f"{subject}/{session}"

        if not os.path.exists(edf_path):
            print(f"[!] SKIPPED [{tag}]: EDF not found -> {edf_path}")
            skipped_files.append((tag, 'edf_missing', edf_path))
            continue
        if not os.path.exists(events_path):
            print(f"[!] SKIPPED [{tag}]: events file not found -> {events_path}")
            skipped_files.append((tag, 'events_missing', events_path))
            continue

        signals, info = read_edf(edf_path)

        if channels is None:
            # Resolve channel_set from the EDF channel names of the FIRST
            # successfully read file; validate all later files against this FIXED list.
            channels = resolve_channels(signals, channel_set)
            print(f"[*] Channel set {channel_set}: {len(channels)} channels -> {channels}")

        missing_channels = [ch for ch in channels if ch not in signals]
        if missing_channels:
            print(f"[!] SKIPPED [{tag}]: missing channel(s) {missing_channels}")
            skipped_files.append((tag, f'missing_channels:{missing_channels}', edf_path))
            continue

        fs = info['sampling_rate'][channels[0]]
        if fs_reference is None:
            fs_reference = fs
        elif abs(fs - fs_reference) > 1e-6:
            print(f"[!] SKIPPED [{tag}]: sampling-rate mismatch "
                  f"({fs}Hz != reference {fs_reference}Hz); cannot combine in one model.")
            skipped_files.append((tag, f'fs_mismatch:{fs}!={fs_reference}', edf_path))
            continue

        try:
            events_all = parse_events(events_path)
        except Exception as e:
            print(f"[!] SKIPPED [{tag}]: events could not be read ({e})")
            skipped_files.append((tag, f'events_parse_error:{e}', events_path))
            continue
        events = [(o, d, t) for o, d, t in events_all if t in label_map]
        if not events:
            print(f"[!] SKIPPED [{tag}]: no usable events for the configured label map")
            skipped_files.append((tag, 'no_usable_events', events_path))
            continue

        temp_model = DeployableBCIModel(
            channels=channels, sampling_rate=fs, window_len=window_len, step_len=step_len,
            idle_distance_threshold=idle_distance_threshold,
            confidence_threshold=confidence_threshold,
            n_features_select=n_features_select, lda_shrinkage=lda_shrinkage,
            channel_set=channel_set,
            channel_normalization=channel_normalization,
            label_map=label_map,
        )
        preprocessed = temp_model.preprocess_continuous(signals)

        file_windows = extract_command_training_windows(events, fs, window_len=window_len,
                                                          label_map=label_map)
        n_stop_file = sum(1 for w in file_windows if w['label'] == 0)
        n_walk_file = sum(1 for w in file_windows if w['label'] == 1)
        print(f"[+] [{tag}] {len(file_windows)} windows (STOP={n_stop_file}, WALK={n_walk_file})")
        per_file_stats.append({
            'subject': subject, 'session': session, 'edf': edf_path, 'events': events_path,
            'n_windows': len(file_windows), 'n_stop': n_stop_file, 'n_walk': n_walk_file,
        })

        if n_stop_file == 0 or n_walk_file == 0:
            print(f"[!] WARNING [{tag}]: one class is completely absent (STOP={n_stop_file}, WALK={n_walk_file}); "
                  f"this file cannot teach discrimination by itself and only contributes to the pool.")

        for w in file_windows:
            epoch = temp_model._epoch_dict(preprocessed, w['start_idx'], w['end_idx'])
            trial = {
                'epoch': epoch,
                'subject': subject,
                'session': session,
                'edf': edf_path,
                'events': events_path,
                'start_idx': int(w['start_idx']),
                'end_idx': int(w['end_idx']),
                'start_time': round(w['start_idx'] / fs, 3),
                'end_time': round(w['end_idx'] / fs, 3),
                'label': int(w['label']),
            }
            pooled_trials.append(trial)
            pooled_labels.append(w['label'])
            manifest_rows.append({
                'subject': subject, 'session': session, 'edf': edf_path, 'events': events_path,
                'start_idx': int(w['start_idx']), 'end_idx': int(w['end_idx']),
                'start_time': round(w['start_idx'] / fs, 3), 'end_time': round(w['end_idx'] / fs, 3),
                'label': int(w['label']), 'label_name': LABEL_NAMES[w['label']],
            })

    if fs_reference is None or len(pooled_trials) == 0:
        raise SystemExit("No files could be used; there are no windows to pool for train_multi. "
                         "See the [!] SKIPPED lines above.")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'train_manifest_resolved.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv_mod.DictWriter(
            f,
            fieldnames=['subject', 'session', 'edf', 'events', 'start_idx', 'end_idx',
                        'start_time', 'end_time', 'label', 'label_name']
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    y_pooled = np.asarray(pooled_labels)
    n_stop_candidate = int(np.sum(y_pooled == 0))
    n_walk_candidate = int(np.sum(y_pooled == 1))
    print(f"\n[*] Total candidate training windows: {len(pooled_trials)} "
          f"(STOP={n_stop_candidate}, WALK={n_walk_candidate}) from {len(per_file_stats)} files; "
          f"{len(skipped_files)} files skipped")
    print(f"[+] Candidate manifest written: {manifest_path}")

    imbalance_ratio_candidate = max(n_stop_candidate, n_walk_candidate) / max(min(n_stop_candidate, n_walk_candidate), 1)
    if imbalance_ratio_candidate >= 3.0:
        print(f"[!] WARNING: severe class imbalance; ratio {imbalance_ratio_candidate:.1f}:1 "
              f"(STOP={n_stop_candidate}, WALK={n_walk_candidate}). The model may favor the majority class.")

    # 1) Subject balancing: prevent sub-01 from dominating the pool with two sessions.
    pooled_trials, y_pooled, subject_balance_stats = apply_subject_balancing(
        pooled_trials, y_pooled, balance_subjects, seed=seed
    )
    if subject_balance_stats['applied']:
        print(f"[*] Subject balancing ({subject_balance_stats['method']}): "
              f"target={subject_balance_stats['target_per_subject']} trial/subject (seed={seed})")

    # 2) Class balancing: manage the STOP/WALK balance separately.
    pooled_trials, y_pooled, class_balance_stats = apply_class_balancing(
        pooled_trials, y_pooled, balance_classes, seed
    )
    if class_balance_stats['applied']:
        print(f"[*] Class balancing ({class_balance_stats['method']}): "
              f"STOP {class_balance_stats['n_stop_before']}->{class_balance_stats['n_stop_after']}, "
              f"WALK {class_balance_stats['n_walk_before']}->{class_balance_stats['n_walk_after']}, "
              f"dropped={class_balance_stats['n_dropped']} (seed={class_balance_stats['seed']})")

    # Class balancing may alter the subject distribution again; save the final subject summary too.
    final_subject_stats = summarize_subjects(
        pooled_trials, y_pooled, seed=seed, method='final_after_all_balancing', applied=False
    )

    used_manifest_path = os.path.join(output_dir, 'train_manifest_used.csv')
    with open(used_manifest_path, 'w', newline='') as f:
        writer = csv_mod.DictWriter(
            f,
            fieldnames=['subject', 'session', 'edf', 'events', 'start_idx', 'end_idx',
                        'start_time', 'end_time', 'label', 'label_name']
        )
        writer.writeheader()
        for t, label in zip(pooled_trials, y_pooled):
            writer.writerow({
                'subject': t['subject'], 'session': t['session'],
                'edf': t['edf'], 'events': t['events'],
                'start_idx': t['start_idx'], 'end_idx': t['end_idx'],
                'start_time': t['start_time'], 'end_time': t['end_time'],
                'label': int(label), 'label_name': LABEL_NAMES[int(label)],
            })

    raw_trials = [t['epoch'] for t in pooled_trials]
    n_stop_final = int(np.sum(y_pooled == 0))
    n_walk_final = int(np.sum(y_pooled == 1))
    print(f"[*] Final number of windows entering training: {len(raw_trials)} "
          f"(STOP={n_stop_final}, WALK={n_walk_final})")
    print(f"[+] Used manifest written: {used_manifest_path}")

    model = DeployableBCIModel(
        channels=channels, sampling_rate=fs_reference, window_len=window_len, step_len=step_len,
        idle_distance_threshold=idle_distance_threshold,
        confidence_threshold=confidence_threshold,
        n_features_select=n_features_select, lda_shrinkage=lda_shrinkage,
        channel_set=channel_set,
        channel_normalization=channel_normalization,
        label_map=label_map,
    )
    stats = model.train_from_trials(raw_trials, y_pooled)
    print(f"\n[+] Pooled training completed: STOP={stats['n_stop']}, WALK={stats['n_walk']}")
    print(f"[+] Selected feature indices: {stats['selected_feature_idx']}")

    info_saved = model.save(output_dir)

    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("BCI Walk/Stop Model - MULTI-SESSION Training Summary\n")
        f.write("=" * 55 + "\n")
        f.write(f"Dataset list: {dataset_list_csv}\n")
        f.write(f"Files used: {len(per_file_stats)}, skipped: {len(skipped_files)}\n\n")
        for s in per_file_stats:
            f.write(f"  [{s['subject']}/{s['session']}] {s['edf']}: "
                    f"{s['n_windows']} windows (STOP={s['n_stop']}, WALK={s['n_walk']})\n")
        if skipped_files:
            f.write("\nSkipped files:\n")
            for tag, reason, path in skipped_files:
                f.write(f"  [{tag}] {reason}: {path}\n")
        f.write(f"\nCandidate pooled training windows: {len(manifest_rows)} "
                f"(STOP={n_stop_candidate}, WALK={n_walk_candidate})\n")
        f.write(f"Final used training windows: {len(raw_trials)} "
                f"(STOP={stats['n_stop']}, WALK={stats['n_walk']})\n")
        f.write(f"Candidate class imbalance ratio: {imbalance_ratio_candidate:.2f}:1\n")
        f.write(f"Channel set: {channel_set}\n")
        f.write(f"Channel normalization: {channel_normalization}\n")
        f.write(f"Label map: {label_map}\n")
        f.write(f"Channels: {channels}\n")
        f.write(f"Sampling rate: {fs_reference}\n")
        f.write(f"CSP: {info_saved['use_csp']}, n_features_select: {info_saved['n_features_select']}, "
                f"lda_shrinkage: {info_saved['lda_shrinkage']}\n")
        f.write(f"idle_distance_threshold: {idle_distance_threshold}, "
                f"confidence_threshold: {confidence_threshold}\n\n")
        f.write(_format_subject_balance_stats(subject_balance_stats) + "\n\n")
        f.write(_format_balance_stats(class_balance_stats) + "\n\n")
        f.write("Final subject distribution after all balancing steps:\n")
        f.write(_format_subject_balance_stats(final_subject_stats) + "\n\n")
        f.write(f"Candidate manifest: {manifest_path}\n")
        f.write(f"Used manifest: {used_manifest_path}\n\n")
        f.write("IMPORTANT: Continuous EEG signals were NEVER concatenated across sessions.\n")
        f.write("Each recording was preprocessed independently; only already-epoched\n")
        f.write("(labeled command-window) trials were pooled before training.\n\n")
        f.write("SCIENTIFIC INTEGRITY: This model uses a pretrained model to predict\n")
        f.write("Walk/Stop command periods from EEG. It is not a proven decoder of pure\n")
        f.write("brain-internal walking intention. Multi-session pooling attempts to\n")
        f.write("reduce the risk of overfitting to ONE session, but cross-subject/session\n")
        f.write("generalization cannot be claimed without separate validation using\n")
        f.write("validate_timeline results.\n")

    print(f"\n[+] Model saved: {output_dir}/")
    print(f"[+] Provenance: {manifest_path} (candidate), {used_manifest_path} (final used)")
    return model, per_file_stats, skipped_files


# ============================================================================
# MODE: validate_timeline
# ============================================================================

def _prediction_counts(y_pred):
    classes = [0, 1, 2]
    return {LABEL_NAMES[c]: int(np.sum(y_pred == c)) for c in classes}


def _predict_full_record(edf_path, model):
    """Generate sliding-window predictions over a full EDF recording using the model."""
    signals, info = read_edf(edf_path)

    missing_channels = [ch for ch in model.channels if ch not in signals]
    if missing_channels:
        raise SystemExit(f"The EDF is missing channel(s) required by the model: {missing_channels}")

    edf_fs = info['sampling_rate'][model.channels[0]]
    if abs(edf_fs - model.sampling_rate) > 1e-6:
        raise SystemExit(
            f"Sampling-rate mismatch: EDF {edf_fs}Hz, model {model.sampling_rate}Hz. "
            "Use a model trained with the same sampling rate."
        )

    preprocessed = model.preprocess_continuous(signals)
    signal_len = len(preprocessed[model.channels[0]])
    windows = build_sliding_windows(signal_len, model.sampling_rate, model.window_len, model.step_len)

    rows = []
    for w in windows:
        label, conf, z_dist, raw_pred = model.predict_window(preprocessed, w['start_idx'], w['end_idx'])
        rows.append({
            'start_time': w['start_time'],
            'end_time': w['end_time'],
            'predicted': int(label),
            'confidence': conf,
            'z_distance': z_dist,
            'raw_prediction': None if raw_pred is None else int(raw_pred),
        })

    meta = {
        'edf': edf_path,
        'sampling_rate': model.sampling_rate,
        'channel_normalization': model.channel_normalization,
        'signal_len_samples': int(signal_len),
        'duration_seconds': float(signal_len / model.sampling_rate),
        'window_len': model.window_len,
        'step_len': model.step_len,
        'n_windows': len(windows),
    }
    return rows, meta


def _write_prediction_outputs(rows, meta, output_dir,
                              timeline_csv_name='predicted_timeline.csv',
                              summary_json_name='prediction_summary.json',
                              smoothing_window=3):
    """Write an unlabeled-prediction CSV, summary JSON, and collapse report.

    Accuracy is not calculated. The CSV contains both raw and smoothed predictions.
    raw_predicted_label = raw window label after the IDLE gate and confidence threshold.
    smoothed_predicted_label = label after temporal smoothing.
    """
    import csv
    os.makedirs(output_dir, exist_ok=True)

    raw_labels_int = [int(r['predicted']) for r in rows]
    smoothed_prediction = apply_temporal_smoothing(raw_labels_int, smoothing_window)

    timeline_path = os.path.join(output_dir, timeline_csv_name)
    with open(timeline_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'start_time', 'end_time',
            'raw_predicted_label', 'raw_confidence', 'raw_z_distance',
            'raw_classifier_prediction',
            'smoothed_predicted_label',
        ])
        for idx, r in enumerate(rows):
            writer.writerow([
                f"{r['start_time']:.2f}",
                f"{r['end_time']:.2f}",
                LABEL_NAMES[raw_labels_int[idx]],
                '' if r['confidence'] is None else f"{r['confidence']:.4f}",
                f"{r['z_distance']:.3f}",
                '' if r['raw_prediction'] is None else LABEL_NAMES[r['raw_prediction']],
                LABEL_NAMES[smoothed_prediction[idx]],
            ])

    y_raw = np.asarray(raw_labels_int)
    y_smoothed = np.asarray(smoothed_prediction)
    raw_counts = _prediction_counts(y_raw)
    smoothed_counts = _prediction_counts(y_smoothed)
    total = len(y_raw)

    dominant_frac_raw = (max(raw_counts.values()) / total) if total else float('nan')
    dominant_frac_smoothed = (max(smoothed_counts.values()) / total) if total else float('nan')
    collapse_raw = bool(total and (dominant_frac_raw > 0.90 or raw_counts['WALK'] == 0))
    collapse_smoothed = bool(total and (dominant_frac_smoothed > 0.90 or smoothed_counts['WALK'] == 0))

    summary = {
        **meta,
        'validation_available': False,
        'accuracy_computed': False,
        'reason_accuracy_not_computed': 'No ground-truth events were provided. Accuracy cannot be computed.',
        'smoothing_window': int(smoothing_window),
        'raw': {
            'prediction_counts': raw_counts,
            'dominant_prediction_fraction': dominant_frac_raw,
            'collapse_warning': collapse_raw,
        },
        'smoothed': {
            'prediction_counts': smoothed_counts,
            'dominant_prediction_fraction': dominant_frac_smoothed,
            'collapse_warning': collapse_smoothed,
        },
        # Backward-compatible aliases: use smoothed as the deployment-facing stream.
        'prediction_counts': smoothed_counts,
        'dominant_prediction_fraction': dominant_frac_smoothed,
        'collapse_warning': collapse_smoothed,
        'outputs': {
            'timeline_csv': timeline_path,
            'summary_json': os.path.join(output_dir, summary_json_name),
            'collapse_report': os.path.join(output_dir, 'collapse_report.txt'),
        }
    }

    summary_path = os.path.join(output_dir, summary_json_name)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)

    collapse_path = os.path.join(output_dir, 'collapse_report.txt')
    with open(collapse_path, 'w') as f:
        f.write("COLLAPSE CHECK - UNLABELED PREDICTION\n" + "=" * 45 + "\n")
        f.write("No ground-truth events were provided.\n")
        f.write("Raw and smoothed predictions saved.\n")
        f.write("Accuracy cannot be computed.\n")
        f.write(f"Smoothing window: {smoothing_window}\n")
        f.write(f"Total windows: {total}\n")
        f.write(f"RAW prediction counts: {raw_counts}\n")
        f.write(f"SMOOTHED prediction counts: {smoothed_counts}\n")
        if total:
            f.write(f"RAW dominant prediction fraction: {dominant_frac_raw:.2%}\n")
            f.write(f"SMOOTHED dominant prediction fraction: {dominant_frac_smoothed:.2%}\n")
        else:
            f.write("Dominant prediction fraction: N/A\n")
        f.write(f"RAW WALK count: {raw_counts.get('WALK', 0)}\n")
        f.write(f"SMOOTHED WALK count: {smoothed_counts.get('WALK', 0)}\n")
        f.write(f"RAW COLLAPSE WARNING: {'YES' if collapse_raw else 'no'}\n")
        f.write(f"SMOOTHED COLLAPSE WARNING: {'YES' if collapse_smoothed else 'no'}\n")

    return summary


def run_unlabeled_prediction(edf_path, model_dir, output_dir,
                             timeline_csv_name='predicted_timeline.csv',
                             summary_json_name='prediction_summary.json',
                             smoothing_window=3):
    print("=" * 70)
    print("MODE: predict / unlabeled timeline")
    print("=" * 70)

    model = DeployableBCIModel.load(model_dir)
    print(f"[*] Channel normalization: {model.channel_normalization} (loaded from model)")
    rows, meta = _predict_full_record(edf_path, model)
    print(f"[*] Built {len(rows)} sliding windows "
          f"(window={model.window_len}s, step={model.step_len}s, over the full recording)")
    print(f"[*] Temporal smoothing window: {smoothing_window} "
          f"({'no smoothing' if smoothing_window == 1 else 'centered majority vote'})")

    summary = _write_prediction_outputs(
        rows, meta, output_dir,
        timeline_csv_name=timeline_csv_name,
        summary_json_name=summary_json_name,
        smoothing_window=smoothing_window,
    )

    print(f"\n[+] RAW prediction distribution: {summary['raw']['prediction_counts']}")
    print(f"[+] SMOOTHED prediction distribution: {summary['smoothed']['prediction_counts']}")
    print("[+] No ground-truth events were provided.")
    print("[+] Raw and smoothed predictions saved.")
    print("[+] Accuracy cannot be computed.")
    if summary['smoothed']['collapse_warning']:
        print(f"[!!!] SMOOTHED COLLAPSE WARNING: dominant prediction = "
              f"{summary['smoothed']['dominant_prediction_fraction']:.1%} or WALK count = 0.")
    else:
        print(f"[+] No collapse: smoothed dominant prediction "
              f"{summary['smoothed']['dominant_prediction_fraction']:.1%} (<90%), WALK count>0.")
    print(f"\n[+] Outputs: {output_dir}/{timeline_csv_name}, "
          f"{summary_json_name}, collapse_report.txt")
    return summary


def _compute_timeline_metrics(y_true, y_pred):
    """Three-class confusion matrix plus STOP/WALK deployment metrics, without sklearn."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = [0, 1, 2]

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = {}
    for c in classes:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
              else float('nan'))
        per_class[LABEL_NAMES[c]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(cm[c, :].sum()),
        }

    pred_counts = _prediction_counts(y_pred)
    total = len(y_pred)
    walk_recall = per_class['WALK']['recall']
    stop_recall = per_class['STOP']['recall']
    idle_fp_rate = (cm[:, 2].sum() - cm[2, 2]) / max(total - cm[2, :].sum(), 1)

    non_idle_mask = (y_true == 0) | (y_true == 1)
    n_non_idle = int(non_idle_mask.sum())
    if n_non_idle > 0:
        deployment_accuracy_non_idle = float(np.mean(y_pred[non_idle_mask] == y_true[non_idle_mask]))
    else:
        deployment_accuracy_non_idle = float('nan')

    if not np.isnan(walk_recall) and not np.isnan(stop_recall):
        balanced_accuracy = float((walk_recall + stop_recall) / 2)
    else:
        balanced_accuracy = float('nan')

    dominant_frac = max(pred_counts.values()) / total if total else float('nan')
    collapse = bool(total and (dominant_frac > 0.90 or pred_counts['WALK'] == 0))

    return {
        'total_windows': total,
        'prediction_counts': pred_counts,
        'per_class': per_class,
        'walk_recall': walk_recall,
        'stop_recall': stop_recall,
        'deployment_accuracy_non_idle': deployment_accuracy_non_idle,
        'balanced_accuracy': balanced_accuracy,
        'n_non_idle_ground_truth_windows': n_non_idle,
        'idle_false_positive_rate': idle_fp_rate,
        'collapse_warning': collapse,
        'dominant_prediction_fraction': dominant_frac,
        'confusion_matrix': cm,
    }


def _write_confusion_matrix_csv(path, cm):
    import csv
    classes = [0, 1, 2]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + [f"pred_{LABEL_NAMES[c]}" for c in classes])
        for i, c in enumerate(classes):
            writer.writerow([f"true_{LABEL_NAMES[c]}"] + cm[i].tolist())


def run_validate_timeline(edf_path, events_path, model_dir, output_dir,
                          overlap_threshold=0.5, smoothing_window=3):
    print("=" * 70)
    print("MODE: validate_timeline")
    print("=" * 70)

    # events are optional; without them, run unlabeled prediction instead of validation.
    if not events_path or not os.path.exists(events_path):
        if not events_path:
            print("[!] --events was not provided. Only prediction will run because ground truth is unavailable.")
        else:
            print(f"[!] Events file not found: {events_path}")
            print("[!] Only prediction will run because ground truth is unavailable.")
        return run_unlabeled_prediction(
            edf_path, model_dir, output_dir,
            timeline_csv_name='predicted_timeline.csv',
            summary_json_name='prediction_summary.json',
            smoothing_window=smoothing_window,
        )

    model = DeployableBCIModel.load(model_dir)
    print(f"[*] Channel normalization: {model.channel_normalization} (loaded from model)")
    print(f"[*] Label map: {model.label_map} (loaded from model)")
    rows_pred, meta = _predict_full_record(edf_path, model)
    events_all = parse_events(events_path)

    fs = model.sampling_rate
    # Rebuild the same sliding windows used by _predict_full_record; they are in
    # exactly the same order as the prediction rows.
    signal_len = int(meta['signal_len_samples'])
    windows = build_sliding_windows(signal_len, fs, model.window_len, model.step_len)
    print(f"[*] Built {len(windows)} sliding windows "
          f"(window={model.window_len}s, step={model.step_len}s, over the full recording)")
    print(f"[*] Temporal smoothing window: {smoothing_window} "
          f"({'no smoothing' if smoothing_window == 1 else 'centered majority vote'})")

    gt_labels = label_windows_from_events(windows, events_all, overlap_threshold, label_map=model.label_map)
    raw_labels_int = [int(r['predicted']) for r in rows_pred]
    smoothed_prediction = apply_temporal_smoothing(raw_labels_int, smoothing_window)

    rows = []
    for idx, (pred_row, gt) in enumerate(zip(rows_pred, gt_labels)):
        rows.append({
            'start_time': pred_row['start_time'],
            'end_time': pred_row['end_time'],
            'ground_truth': int(gt),
            'raw_predicted': raw_labels_int[idx],
            'raw_confidence': pred_row['confidence'],
            'raw_z_distance': pred_row['z_distance'],
            'raw_classifier_prediction': pred_row['raw_prediction'],
            'smoothed_prediction': int(smoothed_prediction[idx]),
        })

    os.makedirs(output_dir, exist_ok=True)

    import csv
    with open(os.path.join(output_dir, 'validated_timeline.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'window_index', 'start_time', 'end_time', 'ground_truth_label',
            'raw_predicted_label', 'raw_confidence', 'raw_z_distance',
            'raw_classifier_prediction', 'smoothed_predicted_label',
        ])
        for idx, r in enumerate(rows):
            writer.writerow([
                idx,
                f"{r['start_time']:.2f}",
                f"{r['end_time']:.2f}",
                LABEL_NAMES[r['ground_truth']],
                LABEL_NAMES[r['raw_predicted']],
                '' if r['raw_confidence'] is None else f"{r['raw_confidence']:.4f}",
                f"{r['raw_z_distance']:.3f}",
                '' if r['raw_classifier_prediction'] is None else LABEL_NAMES[r['raw_classifier_prediction']],
                LABEL_NAMES[r['smoothed_prediction']],
            ])

    y_true = np.asarray([r['ground_truth'] for r in rows])
    y_raw = np.asarray([r['raw_predicted'] for r in rows])
    y_smoothed = np.asarray([r['smoothed_prediction'] for r in rows])

    raw_metrics = _compute_timeline_metrics(y_true, y_raw)
    smoothed_metrics = _compute_timeline_metrics(y_true, y_smoothed)

    cm_raw = raw_metrics.pop('confusion_matrix')
    cm_smoothed = smoothed_metrics.pop('confusion_matrix')

    _write_confusion_matrix_csv(os.path.join(output_dir, 'timeline_confusion_matrix_raw.csv'), cm_raw)
    _write_confusion_matrix_csv(os.path.join(output_dir, 'timeline_confusion_matrix_smoothed.csv'), cm_smoothed)
    # Backward-compatible alias: deployment-facing stream is smoothed.
    _write_confusion_matrix_csv(os.path.join(output_dir, 'timeline_confusion_matrix.csv'), cm_smoothed)

    total = int(len(y_true))
    metrics = {
        **meta,
        'events': events_path,
        'validation_available': True,
        'accuracy_computed': True,
        'smoothing_window': int(smoothing_window),
        'total_windows': total,
        'raw': raw_metrics,
        'smoothed': smoothed_metrics,
        # Backward-compatible aliases point to smoothed/deployment stream.
        'prediction_counts': smoothed_metrics['prediction_counts'],
        'per_class': smoothed_metrics['per_class'],
        'walk_recall': smoothed_metrics['walk_recall'],
        'stop_recall': smoothed_metrics['stop_recall'],
        'deployment_accuracy_non_idle': smoothed_metrics['deployment_accuracy_non_idle'],
        'balanced_accuracy': smoothed_metrics['balanced_accuracy'],
        'n_non_idle_ground_truth_windows': smoothed_metrics['n_non_idle_ground_truth_windows'],
        'idle_false_positive_rate': smoothed_metrics['idle_false_positive_rate'],
        'collapse_warning': smoothed_metrics['collapse_warning'],
        'dominant_prediction_fraction': smoothed_metrics['dominant_prediction_fraction'],
    }
    with open(os.path.join(output_dir, 'timeline_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    with open(os.path.join(output_dir, 'collapse_report.txt'), 'w') as f:
        f.write("COLLAPSE CHECK\n" + "=" * 40 + "\n")
        f.write(f"Smoothing window: {smoothing_window}\n")
        f.write(f"Total windows: {total}\n")
        f.write(f"RAW prediction counts: {raw_metrics['prediction_counts']}\n")
        f.write(f"SMOOTHED prediction counts: {smoothed_metrics['prediction_counts']}\n")
        f.write(f"RAW dominant prediction fraction: {raw_metrics['dominant_prediction_fraction']:.2%}\n")
        f.write(f"SMOOTHED dominant prediction fraction: {smoothed_metrics['dominant_prediction_fraction']:.2%}\n")
        f.write(f"RAW WALK count: {raw_metrics['prediction_counts']['WALK']}\n")
        f.write(f"SMOOTHED WALK count: {smoothed_metrics['prediction_counts']['WALK']}\n")
        f.write(f"RAW COLLAPSE WARNING: {'YES' if raw_metrics['collapse_warning'] else 'no'}\n")
        f.write(f"SMOOTHED COLLAPSE WARNING: {'YES' if smoothed_metrics['collapse_warning'] else 'no'}\n")
        f.write("\nMETRICS\n" + "-" * 40 + "\n")
        f.write(f"[RAW] Accuracy non-IDLE: {raw_metrics['deployment_accuracy_non_idle']:.4f}\n")
        f.write(f"[RAW] Balanced accuracy: {raw_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"[SMOOTHED] Accuracy non-IDLE: {smoothed_metrics['deployment_accuracy_non_idle']:.4f}\n")
        f.write(f"[SMOOTHED] Balanced accuracy: {smoothed_metrics['balanced_accuracy']:.4f}\n")

    print(f"\n[+] RAW prediction distribution: {raw_metrics['prediction_counts']}")
    print(f"[+] SMOOTHED prediction distribution: {smoothed_metrics['prediction_counts']}")
    print(f"[SMOOTHED] Accuracy: {smoothed_metrics['deployment_accuracy_non_idle']:.2%}  "
          f"Balanced: {smoothed_metrics['balanced_accuracy']:.2%}")
    print(f"[RAW]      Accuracy: {raw_metrics['deployment_accuracy_non_idle']:.2%}  "
          f"Balanced: {raw_metrics['balanced_accuracy']:.2%}")
    print(f"[+] SMOOTHED WALK recall: {smoothed_metrics['walk_recall']:.2%}" if not np.isnan(smoothed_metrics['walk_recall']) else "[+] SMOOTHED WALK recall: N/A")
    print(f"[+] SMOOTHED STOP recall: {smoothed_metrics['stop_recall']:.2%}" if not np.isnan(smoothed_metrics['stop_recall']) else "[+] SMOOTHED STOP recall: N/A")
    print(f"[+] SMOOTHED IDLE false positive rate: {smoothed_metrics['idle_false_positive_rate']:.2%}")

    print(f"\nConfusion matrix RAW (rows=true, columns=predicted):")
    print(f"{'':>12}{'pred_STOP':>12}{'pred_WALK':>12}{'pred_IDLE':>12}")
    for i, c in enumerate([0, 1, 2]):
        print(f"{'true_'+LABEL_NAMES[c]:>12}{cm_raw[i,0]:>12}{cm_raw[i,1]:>12}{cm_raw[i,2]:>12}")

    print(f"\nConfusion matrix SMOOTHED (rows=true, columns=predicted):")
    print(f"{'':>12}{'pred_STOP':>12}{'pred_WALK':>12}{'pred_IDLE':>12}")
    for i, c in enumerate([0, 1, 2]):
        print(f"{'true_'+LABEL_NAMES[c]:>12}{cm_smoothed[i,0]:>12}{cm_smoothed[i,1]:>12}{cm_smoothed[i,2]:>12}")

    if smoothed_metrics['collapse_warning']:
        print(f"\n[!!!] SMOOTHED COLLAPSE WARNING: dominant prediction = "
              f"{smoothed_metrics['dominant_prediction_fraction']:.1%} "
              f"of all windows, or WALK count = 0. MODEL NOT DEPLOYABLE YET.")
    else:
        print(f"\n[+] No collapse: smoothed dominant prediction "
              f"{smoothed_metrics['dominant_prediction_fraction']:.1%} (<90%), WALK count>0.")

    print(f"\n[+] Outputs: {output_dir}/validated_timeline.csv, "
          f"timeline_confusion_matrix_raw.csv, timeline_confusion_matrix_smoothed.csv, "
          f"timeline_confusion_matrix.csv, timeline_metrics.json, collapse_report.txt")

    return metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BCI Walk/Stop pipeline v2.8")
    parser.add_argument('--mode', required=True, choices=['train', 'train_multi', 'validate_timeline', 'predict'])
    parser.add_argument('--edf', default=None, help='(train/validate_timeline/predict)')
    parser.add_argument('--events', default=None)
    parser.add_argument('--model', default=None, help='trained_model directory (for validate_timeline/predict)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-list', default=None, help='(train_multi) subject,session,edf,events CSV')
    parser.add_argument('--dataset-dir', default='.', help='(train_multi) base directory for relative paths in the CSV')
    parser.add_argument('--event-overlap-threshold', type=float, default=0.5)
    parser.add_argument('--idle-distance-threshold', type=float, default=999.0)
    parser.add_argument('--confidence-threshold', type=float, default=0.45)
    parser.add_argument('--n-features-select', type=int, default=45)
    parser.add_argument('--lda-shrinkage', type=float, default=0.0)
    parser.add_argument('--balance-classes', choices=['none', 'downsample'],
                         default='none', help='Method for addressing class imbalance '
                                               '(default: none). Currently only downsample is implemented.')
    parser.add_argument('--balance-subjects', choices=['none', 'downsample'], default='none',
                        help='Subject-level balancing for train_multi: none | downsample')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducible class balancing')
    parser.add_argument('--smoothing-window', type=int, choices=[1, 3, 5], default=3,
                        help='Temporal smoothing window (1=no smoothing)')
    parser.add_argument('--channel-normalization', choices=['none', 'zscore'], default='zscore',
                        help='Per-channel raw EEG normalization before bandpass/CSP: none | zscore')
    parser.add_argument('--channel-set', choices=['motor3', 'motor5', 'motor9', 'motor13', 'all_eeg'],
                        default='motor3',
                        help='(train/train_multi only) EEG channel set to use. '
                             'validate_timeline/predict always use the channels saved in the model.')
    parser.add_argument('--label-map', default=None,
                        help='(train/train_multi only) trial_type -> 0 (STOP) / 1 (WALK) mapping, as a '
                             'JSON object string (e.g. \'{"x5":0,"x8":1}\') or a path to a JSON file. '
                             f'Defaults to the REX-system scheme {DEFAULT_LABEL_MAP}. '
                             'validate_timeline/predict always reuse the label map saved in the model.')
    args = parser.parse_args()

    logger.setLevel(logging.WARNING)

    if args.mode == 'train':
        if not args.edf or not args.events:
            raise SystemExit("--edf and --events are required in train mode")
        run_train(args.edf, args.events, args.output_dir,
                   channel_set=args.channel_set,
                   idle_distance_threshold=args.idle_distance_threshold,
                   confidence_threshold=args.confidence_threshold,
                   n_features_select=args.n_features_select,
                   lda_shrinkage=args.lda_shrinkage,
                   balance_classes=args.balance_classes, seed=args.seed,
                   channel_normalization=args.channel_normalization,
                   label_map=resolve_label_map(args.label_map))

    elif args.mode == 'train_multi':
        if not args.dataset_list:
            raise SystemExit("--dataset-list is required in train_multi mode")
        run_train_multi(args.dataset_list, args.output_dir, dataset_dir=args.dataset_dir,
                         channel_set=args.channel_set,
                         idle_distance_threshold=args.idle_distance_threshold,
                         confidence_threshold=args.confidence_threshold,
                         n_features_select=args.n_features_select,
                         lda_shrinkage=args.lda_shrinkage,
                         balance_classes=args.balance_classes,
                         balance_subjects=args.balance_subjects, seed=args.seed,
                         channel_normalization=args.channel_normalization,
                         label_map=resolve_label_map(args.label_map))

    elif args.mode == 'validate_timeline':
        if not args.edf or not args.model:
            raise SystemExit("--edf and --model are required in validate_timeline mode. --events is optional.")
        run_validate_timeline(args.edf, args.events, args.model, args.output_dir,
                               overlap_threshold=args.event_overlap_threshold,
                               smoothing_window=args.smoothing_window)

    elif args.mode == 'predict':
        if not args.edf or not args.model:
            raise SystemExit("--edf and --model are required in predict mode")
        run_unlabeled_prediction(args.edf, args.model, args.output_dir,
                                 smoothing_window=args.smoothing_window)


if __name__ == '__main__':
    main()
