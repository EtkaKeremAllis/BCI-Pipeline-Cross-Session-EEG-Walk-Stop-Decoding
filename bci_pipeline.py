#!/usr/bin/env python3
"""
bci_pipeline.py
===============================================================================
Offline Walk/Stop BCI validation pipeline (single-file, v1.1)

EEG (C3/C4/Cz) -> preprocessing -> CSP + feature extraction -> feature
selection -> shrinkage LDA -> LOOCV validation -> EOG artifact analysis ->
symbolic joystick command generation (console output only).

No new classifier/model is introduced here. The classification approach
(CSP + F-score feature selection + shrinkage-regularized LDA) is unchanged
from the existing modern_bci_v2.py pipeline - this file only reorganizes it
into modular functions, adds a CLI, and writes structured result files.

Usage:
    python bci_pipeline.py \
        --edf sub-01_ses-01_tasktraining_eeg.edf \
        --events sub-01_ses-01_tasktraining_acq-rexcommand_events.tsv \
        --output-dir results

-------------------------------------------------------------------------------
IMPORTANT SCOPE NOTE (read this before assuming more than the code does):

This is an OFFLINE validation and SYMBOLIC joystick command generation
script. It does not provide real-time EEG streaming or real HID/vJoy
control yet. Every "joystick command" produced here is a printed label
(WALK/STOP/IDLE) derived from held-out (LOOCV) predictions on a fixed
recording - not a live control signal.

Symbolic commands (commands.csv) are generated from the held-out LOOCV
predictions on the raw-EEG condition, never from predictions the final
deployable model makes on its own training data. This avoids leakage
between "how good does this look" and "what would this actually say".

v1.1 changes (code-review follow-up):
  - parse_events() now reads TSV directly (fast path) and only falls back
    to PDF-table extraction via pdfplumber if a real .tsv/.csv is not found.
  - Added an OutputDevice abstract base class so ConsoleOutput is one
    concrete implementation; future real HID/vJoy backends can subclass it
    without touching command-generation logic.
  - Everything else (CSP, feature selection, shrinkage LDA, LOOCV, EOG
    artifact analysis) is unchanged from v1.0.

Known future work (not implemented here, flagged deliberately rather than
silently omitted):
  - Sliding-window (overlapping) epoching instead of fixed 5s non-overlapping
    windows, for closer-to-real-time granularity.
  - A command state machine (majority vote + cooldown) sitting between
    per-window predictions and emitted commands, to avoid rapid flicker
    between WALK/STOP/IDLE in a live setting.
  - A real-time acquisition backend (e.g. LSL/BrainFlow) feeding this same
    preprocessing + classifier code path.
-------------------------------------------------------------------------------
"""
import argparse
import csv
import json
import logging
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import signal, stats
from scipy.linalg import eig

warnings.filterwarnings('ignore')


# ==============================================================================
# LOGGING
# ==============================================================================
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        return f"[{record.levelname}] {record.getMessage()}"


def setup_logging(name="BCI", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    return logger


logger = setup_logging()


# ==============================================================================
# 1) EDF READER (no external dependencies)
# ==============================================================================
def read_edf(filepath: str):
    """
    Minimal EDF (European Data Format) reader.
    Spec: https://www.edfplus.info/specs/edf.html
    Returns (signals: dict[label -> np.ndarray], info: dict).
    """
    with open(filepath, 'rb') as f:
        header = f.read(256)
        version = header[0:8].decode('ascii', errors='replace').strip()
        patient_id = header[8:88].decode('ascii', errors='replace').strip()
        recording_id = header[88:168].decode('ascii', errors='replace').strip()
        start_date = header[168:176].decode('ascii', errors='replace').strip()
        start_time = header[176:184].decode('ascii', errors='replace').strip()
        n_header_bytes = int(header[184:192].decode('ascii').strip())
        reserved = header[192:236].decode('ascii', errors='replace').strip()
        n_records = int(header[236:244].decode('ascii').strip())
        record_duration = float(header[244:252].decode('ascii').strip())
        n_signals = int(header[252:256].decode('ascii').strip())

        def read_field(n_bytes):
            return [f.read(n_bytes).decode('ascii', errors='replace').strip()
                    for _ in range(n_signals)]

        labels = read_field(16)
        transducer = read_field(80)
        phys_dim = read_field(8)
        phys_min = [float(x) for x in read_field(8)]
        phys_max = [float(x) for x in read_field(8)]
        dig_min = [int(x) for x in read_field(8)]
        dig_max = [int(x) for x in read_field(8)]
        prefiltering = read_field(80)
        n_samples_per_record = [int(x) for x in read_field(8)]
        reserved_sig = read_field(32)

        record_size = sum(n_samples_per_record)
        raw = f.read()
        n_records_actual = len(raw) // (record_size * 2)
        n_records_use = min(n_records, n_records_actual) if n_records > 0 else n_records_actual

        arr = np.frombuffer(raw[:n_records_use * record_size * 2], dtype='<i2')
        arr = arr.reshape(n_records_use, record_size)

        offset = 0
        signals_out = {}
        fs_per_channel = {}
        for i, label in enumerate(labels):
            ns = n_samples_per_record[i]
            chan_digital = arr[:, offset:offset + ns].flatten()
            offset += ns
            dmin, dmax = dig_min[i], dig_max[i]
            pmin, pmax = phys_min[i], phys_max[i]
            scale = (pmax - pmin) / (dmax - dmin) if dmax != dmin else 1.0
            phys = (chan_digital.astype(np.float64) - dmin) * scale + pmin
            signals_out[label] = phys
            fs_per_channel[label] = ns / record_duration

    info = {
        'version': version, 'patient_id': patient_id, 'recording_id': recording_id,
        'start_date': start_date, 'start_time': start_time,
        'n_records': n_records_use, 'record_duration': record_duration,
        'n_signals': n_signals, 'labels': labels,
        'phys_dim': dict(zip(labels, phys_dim)),
        'sampling_rate': fs_per_channel,
        'prefiltering': dict(zip(labels, prefiltering)),
    }
    return signals_out, info


# ==============================================================================
# 2) EVENT PARSER
# ==============================================================================
def _parse_events_tsv(path: str) -> List[Tuple[float, float, str]]:
    """Fast path: read a real BIDS-style events.tsv/.csv with a header row
    containing 'onset', 'duration', 'trial_type' columns (tab or comma sep)."""
    with open(path, 'r', newline='') as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters='\t,')
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]
        if not {'onset', 'duration', 'trial_type'}.issubset(set(fieldnames)):
            raise ValueError("TSV missing required columns onset/duration/trial_type")
        events = []
        for row in reader:
            row = {k.strip().lower(): v for k, v in row.items()}
            events.append((float(row['onset']), float(row['duration']), row['trial_type'].strip()))
    return events


def _parse_events_pdf(path: str) -> List[Tuple[float, float, str]]:
    """Fallback path: events table was exported/received as a PDF instead of a
    real TSV. Extracts the same (onset, duration, trial_type) rows from the
    text layer of the first page using pdfplumber."""
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text()
    lines = text.split('\n')
    events = []
    started = False
    pattern = re.compile(r'^([\d.]+)\s+([\d.]+)\s+(\S+)$')
    for line in lines:
        if line.strip().lower().startswith('onset'):
            started = True
            continue
        if started:
            m = pattern.match(line.strip())
            if m:
                onset, duration, trial_type = m.groups()
                events.append((float(onset), float(duration), trial_type))
            else:
                break
    return events


def parse_events(path: str) -> List[Tuple[float, float, str]]:
    """
    Parses the rexcommand / rexstate events table.
    Tries a real TSV/CSV read first (the expected BIDS format); only falls
    back to PDF-table text extraction if the file isn't a parseable TSV/CSV
    (e.g. it was handed over as a scanned/exported PDF).
    Returns: [(onset, duration, trial_type), ...]
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tsv', '.csv'):
        return _parse_events_tsv(path)
    try:
        return _parse_events_tsv(path)
    except Exception as e:
        logger.warning(f"TSV/CSV parse failed ({e}); falling back to PDF-table extraction.")
        return _parse_events_pdf(path)


# ==============================================================================
# 3) CONFIG
# ==============================================================================
@dataclass
class BCIConfig:
    sampling_rate: int = 256
    n_channels: int = 3
    channels: List[str] = None
    notch_freq: int = 50
    use_notch: bool = True  # auto-skipped if notch_freq >= Nyquist
    bandpass_low: float = 0.5
    bandpass_high: float = 50
    fir_order: int = 256
    use_laplacian: bool = True
    use_csp: bool = True
    csp_n_filters: int = 4
    classifier_type: str = 'lda'
    confidence_threshold: float = 0.6
    lda_shrinkage: float = 0.15
    use_feature_selection: bool = True
    n_features_select: int = 20

    def __post_init__(self):
        if self.channels is None:
            self.channels = ['C3', 'C4', 'Cz'][:self.n_channels]


# ==============================================================================
# 4) FEATURE EXTRACTION
# ==============================================================================
class AdvancedFeatureExtractor:
    """Time-domain + frequency-domain + spatial features."""

    def __init__(self, sampling_rate=256):
        self.fs = sampling_rate
        self.bands = {
            'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12),
            'Mu': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)
        }

    def extract_time_domain(self, signal_data: np.ndarray) -> Dict:
        features = {}
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['range'] = np.max(signal_data) - np.min(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data ** 2))
        features['peak_to_peak'] = np.ptp(signal_data)
        features['kurtosis'] = stats.kurtosis(signal_data)
        features['skewness'] = stats.skew(signal_data)
        features['line_length'] = np.sum(np.abs(np.diff(signal_data)))
        features['hjorth_activity'] = np.var(signal_data)
        features['hjorth_mobility'] = np.sqrt(
            np.var(np.diff(signal_data)) / np.var(signal_data)
        )
        diff2 = np.diff(signal_data, 2)
        if np.var(np.diff(signal_data)) > 1e-6:
            features['hjorth_complexity'] = (
                np.sqrt(np.var(diff2) / np.var(np.diff(signal_data))) /
                features['hjorth_mobility']
            )
        else:
            features['hjorth_complexity'] = 0
        features['zero_crossings'] = np.sum(np.diff(np.sign(signal_data)) != 0)
        psd = np.abs(np.fft.fft(signal_data)) ** 2
        psd_norm = psd / np.sum(psd)
        features['spectral_entropy'] = -np.sum(
            psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0] + 1e-8)
        )
        return features

    def extract_frequency_domain(self, signal_data: np.ndarray) -> Dict:
        features = {}
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=min(256, len(signal_data)),
                                   noverlap=min(128, len(signal_data) // 2))
        for band_name, (low, high) in self.bands.items():
            idx_low = np.argmin(np.abs(freqs - low))
            idx_high = np.argmin(np.abs(freqs - high))
            power = np.mean(psd[idx_low:idx_high + 1])
            features[f'{band_name}_power'] = power
        mu_power = features.get('Mu_power', 1e-6)
        beta_power = features.get('Beta_power', 1e-6)
        features['mu_beta_ratio'] = mu_power / (beta_power + 1e-6)
        features['beta_mu_ratio'] = beta_power / (mu_power + 1e-6)
        features['alpha_theta_ratio'] = (
            features.get('Alpha_power', 1e-6) / (features.get('Theta_power', 1e-6) + 1e-6)
        )
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features['total_power'] = np.sum(psd)
        return features

    def extract_all_features(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        all_features = []
        for channel in sorted(eeg_dict.keys()):
            signal_data = eeg_dict[channel]
            time_feat = self.extract_time_domain(signal_data)
            freq_feat = self.extract_frequency_domain(signal_data)
            channel_features = {**time_feat, **freq_feat}
            all_features.append(list(channel_features.values()))
        if len(eeg_dict) >= 2:
            channels = sorted(eeg_dict.keys())
            c3_mu = self.extract_frequency_domain(eeg_dict[channels[0]]).get('Mu_power', 0)
            c4_mu = self.extract_frequency_domain(eeg_dict[channels[1]]).get('Mu_power', 0)
            asymmetry = (c3_mu - c4_mu) / (c3_mu + c4_mu + 1e-6)
            all_features.append([asymmetry])
        return np.concatenate(all_features)


# ==============================================================================
# 5) LAPLACIAN REFERENCE
# ==============================================================================
class LaplacianReference:
    def __init__(self, channels: List[str]):
        self.channels = channels
        self.laplacian_config = {
            'C3': {'center': 1.0, 'neighbors': {'FC3': -0.25, 'CP3': -0.25, 'C1': -0.25, 'C5': -0.25}},
            'C4': {'center': 1.0, 'neighbors': {'FC4': -0.25, 'CP4': -0.25, 'C2': -0.25, 'C6': -0.25}},
            'Cz': {'center': 1.0, 'neighbors': {'FCz': -0.25, 'CPz': -0.25, 'C1': -0.125, 'C2': -0.125}}
        }

    def apply(self, eeg_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        laplacian_data = {}
        for ch in self.channels:
            if ch not in eeg_dict:
                continue
            signal_data = eeg_dict[ch].copy()
            if ch in self.laplacian_config:
                config = self.laplacian_config[ch]
                for neighbor_ch, weight in config.get('neighbors', {}).items():
                    if neighbor_ch in eeg_dict:
                        signal_data = signal_data + weight * eeg_dict[neighbor_ch]
            laplacian_data[ch] = signal_data
        return laplacian_data


# ==============================================================================
# 6) CSP (Common Spatial Pattern)
# ==============================================================================
class CSPFilter:
    """Class-discriminative spatial filters: S = C1 @ inv(C1 + C2)."""

    def __init__(self, n_filters: int = 4):
        self.n_filters = n_filters
        self.filters = None
        self.n_components = None

    def fit(self, X_class1: np.ndarray, X_class2: np.ndarray):
        if X_class1.ndim == 2:
            X_class1 = X_class1.reshape(1, *X_class1.shape)
        if X_class2.ndim == 2:
            X_class2 = X_class2.reshape(1, *X_class2.shape)
        C1 = np.zeros((X_class1.shape[1], X_class1.shape[1]))
        for trial in X_class1:
            C = (trial @ trial.T) / np.trace(trial @ trial.T)
            C1 += C / X_class1.shape[0]
        C2 = np.zeros((X_class2.shape[1], X_class2.shape[1]))
        for trial in X_class2:
            C = (trial @ trial.T) / np.trace(trial @ trial.T)
            C2 += C / X_class2.shape[0]
        Lambda, W = eig(C1, C1 + C2)
        idx = np.argsort(Lambda)[::-1]
        self.filters = W[:, idx[:self.n_filters]].real
        self.n_components = self.n_filters

    def transform_logvar(self, X: np.ndarray) -> np.ndarray:
        """Spatial filter + log-variance. X: (n_channels, n_samples) single trial
        or (n_trials, n_channels, n_samples) multi-trial."""
        if self.filters is None:
            return np.array([])
        if X.ndim == 2:
            Z = self.filters.T @ X
            var = np.var(Z, axis=1)
            var_norm = var / (np.sum(var) + 1e-10)
            return np.log(var_norm + 1e-10)
        else:
            return np.array([self.transform_logvar(trial) for trial in X])


# ==============================================================================
# 7) FEATURE SELECTION (F-score)
# ==============================================================================
class FeatureSelector:
    """Select the top-K most discriminative features by F-score."""

    def __init__(self, k: int = 20):
        self.k = k
        self.selected_idx = None
        self.f_scores_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X0, X1 = X[y == 0], X[y == 1]
        mean0, mean1 = X0.mean(axis=0), X1.mean(axis=0)
        var0, var1 = X0.var(axis=0), X1.var(axis=0)
        n0, n1 = len(X0), len(X1)
        pooled_var = ((n0 - 1) * var0 + (n1 - 1) * var1) / max(n0 + n1 - 2, 1)
        f_score = (mean1 - mean0) ** 2 / (pooled_var + 1e-12)
        self.f_scores_ = f_score
        k = min(self.k, X.shape[1])
        self.selected_idx = np.argsort(f_score)[::-1][:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_idx]


# ==============================================================================
# 8) SHRINKAGE LDA
# ==============================================================================
class SimpleLDA:
    """Binary LDA, no sklearn dependency. Supports shrinkage regularization."""

    def __init__(self, shrinkage: float = 0.0):
        self.mean_0 = None
        self.mean_1 = None
        self.pooled_cov = None
        self.coef = None
        self.intercept = None
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray):
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.mean_0 = np.mean(X0, axis=0)
        self.mean_1 = np.mean(X1, axis=0)
        S0 = (X0 - self.mean_0).T @ (X0 - self.mean_0)
        S1 = (X1 - self.mean_1).T @ (X1 - self.mean_1)
        raw_cov = (S0 + S1) / (len(X) - 2)
        n_features = raw_cov.shape[0]
        avg_var = np.trace(raw_cov) / n_features
        shrink_target = avg_var * np.eye(n_features)
        self.pooled_cov = (1 - self.shrinkage) * raw_cov + self.shrinkage * shrink_target
        self.pooled_cov += np.eye(self.pooled_cov.shape[0]) * 1e-6
        try:
            inv_cov = np.linalg.inv(self.pooled_cov)
        except Exception:
            inv_cov = np.linalg.pinv(self.pooled_cov)
        self.coef = inv_cov @ (self.mean_1 - self.mean_0)
        self.intercept = -0.5 * (
            self.mean_1 @ inv_cov @ self.mean_1 - self.mean_0 @ inv_cov @ self.mean_0
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        decision = self.decision_function(X)
        proba = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - proba, proba])


# ==============================================================================
# 9) FIR BANDPASS FILTER (linear phase)
# ==============================================================================
class FIRBandpassFilter:
    def __init__(self, low_freq: float, high_freq: float, sampling_rate: int, order: int = 256):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = sampling_rate
        self.order = order if order % 2 == 1 else order + 1
        nyquist = sampling_rate / 2
        self.taps = signal.firwin(
            self.order, [low_freq / nyquist, high_freq / nyquist], window='hamming'
        )

    def apply_offline(self, signal_data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(self.taps, 1, signal_data)


# ==============================================================================
# 10) METRICS
# ==============================================================================
class BCIMetrics:
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.confidence_scores = []

    def add_result(self, prediction: int, ground_truth: int, confidence: float):
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        self.confidence_scores.append(confidence)

    def compute_metrics(self) -> Dict:
        if len(self.predictions) == 0:
            return {}
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        y_conf = np.array(self.confidence_scores)
        metrics = {'accuracy': float(np.mean(y_pred == y_true)), 'n_samples': len(y_true)}
        if len(np.unique(y_true)) > 1:
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1'] = float(f1)
            try:
                pos_scores = y_conf[y_true == 1]
                neg_scores = y_conf[y_true == 0]
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    auc = np.mean([p > n for p in pos_scores for n in neg_scores])
                    metrics['roc_auc'] = float(auc)
                else:
                    metrics['roc_auc'] = 0.5
            except Exception:
                metrics['roc_auc'] = 0.5
        metrics['mean_confidence'] = float(np.mean(y_conf))
        metrics['std_confidence'] = float(np.std(y_conf))
        return metrics


# ==============================================================================
# 11) MODERN BCI PIPELINE (preprocessing + CSP + feature selection + LDA)
# ==============================================================================
class ModernBCIPipeline:
    def __init__(self, config: BCIConfig = None):
        self.config = config or BCIConfig()
        self.logger = logger
        self.feature_extractor = AdvancedFeatureExtractor(self.config.sampling_rate)
        self.laplacian = LaplacianReference(self.config.channels)
        self.bandpass_fir = FIRBandpassFilter(
            self.config.bandpass_low, self.config.bandpass_high,
            self.config.sampling_rate, order=self.config.fir_order
        )
        nyquist = self.config.sampling_rate / 2
        if self.config.use_notch and self.config.notch_freq < nyquist:
            w0 = self.config.notch_freq / nyquist
            b, a = signal.iirnotch(w0, 30)
            self.notch_sos = signal.tf2sos(b, a)
            self._apply_notch = True
        else:
            self.notch_sos = None
            self._apply_notch = False
            if self.config.use_notch:
                self.logger.warning(
                    f"notch_freq={self.config.notch_freq}Hz >= Nyquist={nyquist}Hz, "
                    f"skipping notch filter (sampling_rate too low)."
                )
        self.csp = CSPFilter(n_filters=self.config.csp_n_filters) if self.config.use_csp else None
        self.classifier = SimpleLDA(shrinkage=self.config.lda_shrinkage)
        self.feature_selector = (
            FeatureSelector(k=self.config.n_features_select)
            if self.config.use_feature_selection else None
        )
        self.metrics = BCIMetrics()

    def preprocess(self, eeg_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        notch_data = {}
        for ch in self.config.channels:
            if ch in eeg_dict:
                if self._apply_notch:
                    notch_data[ch] = signal.sosfilt(self.notch_sos, [eeg_dict[ch]])[0]
                else:
                    notch_data[ch] = eeg_dict[ch]
        ref_data = self.laplacian.apply(notch_data) if self.config.use_laplacian else notch_data
        filtered_data = {}
        for ch in self.config.channels:
            if ch in ref_data:
                filtered_data[ch] = self.bandpass_fir.apply_offline(ref_data[ch])
        return filtered_data

    def extract_features(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return self.feature_extractor.extract_all_features(eeg_dict)

    def _stack_channels(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([eeg_dict[ch] for ch in self.config.channels if ch in eeg_dict])

    def extract_full_features(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        stat_features = self.extract_features(eeg_dict)
        if self.config.use_csp and self.csp is not None and self.csp.filters is not None:
            raw_matrix = self._stack_channels(eeg_dict)
            csp_features = self.csp.transform_logvar(raw_matrix)
            return np.concatenate([stat_features, csp_features])
        return stat_features

    def train(self, raw_trials: List[Dict[str, np.ndarray]], y: np.ndarray):
        y = np.asarray(y)
        if self.config.use_csp and self.csp is not None:
            stacked = np.array([self._stack_channels(t) for t in raw_trials])
            X_class1 = stacked[y == 0]
            X_class2 = stacked[y == 1]
            if len(X_class1) > 0 and len(X_class2) > 0:
                self.csp.fit(X_class1, X_class2)
        X = np.array([self.extract_full_features(t) for t in raw_trials])
        if self.feature_selector is not None:
            self.feature_selector.fit(X, y)
            X = self.feature_selector.transform(X)
        self.classifier.fit(X, y)

    def predict(self, raw_trials: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([self.extract_full_features(t) for t in raw_trials])
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        predictions = self.classifier.predict(X)
        confidences = self.classifier.predict_proba(X)[:, 1]
        return predictions, confidences


# ==============================================================================
# 12) EPOCH / WINDOW EXTRACTION
# ==============================================================================
def build_windows(events: List[Tuple[float, float, str, int]], fs: int,
                   skip_start: float, skip_end: float, window_len: float
                   ) -> List[Tuple[int, int, int]]:
    """
    events: list of (onset, duration, trial_type, label).
    For each event, skip_start / skip_end are trimmed from the usable
    interval, and non-overlapping window_len-sized windows are cut from
    what remains.
    Returns: [(start_idx, end_idx, label), ...]
    """
    windows = []
    for onset, duration, trial_type, label in events:
        usable_start = onset + skip_start
        usable_end = onset + duration - skip_end
        usable_duration = usable_end - usable_start
        n_windows = int(usable_duration // window_len)
        if n_windows <= 0:
            continue
        for w in range(n_windows):
            start_t = usable_start + w * window_len
            start_idx = int(start_t * fs)
            end_idx = start_idx + int(window_len * fs)
            windows.append((start_idx, end_idx, label))
    return windows


def extract_epochs(preprocessed_dict: Dict[str, np.ndarray],
                    windows: List[Tuple[int, int, int]],
                    channels: List[str]) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """windows -> (X: list of per-channel epoch dicts, y: label array)"""
    X, y_out = [], []
    max_len = len(next(iter(preprocessed_dict.values())))
    for start_idx, end_idx, label in windows:
        if end_idx > max_len:
            continue
        epoch = {ch: preprocessed_dict[ch][start_idx:end_idx] for ch in channels}
        X.append(epoch)
        y_out.append(label)
    return X, np.array(y_out)


# ==============================================================================
# 13) LOOCV VALIDATION
# ==============================================================================
def _roc_auc(y_true, y_score):
    from scipy.stats import rankdata
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    n_pos, n_neg = len(pos), len(neg)
    ranks = rankdata(np.concatenate([pos, neg]))
    sum_ranks_pos = ranks[:n_pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def run_loocv(config: BCIConfig, X: List[Dict[str, np.ndarray]], y: np.ndarray) -> Dict:
    """
    Leave-one-out CV. In each fold, CSP + feature selection + shrinkage LDA
    are fit ONLY on the remaining (n-1) trials; the held-out trial is
    predicted with that fold's model. Nothing about the held-out trial's
    label or feature statistics ever touches the fold's fitting step.
    """
    y = np.asarray(y)
    n = len(X)
    y_true_all, y_pred_all, y_conf_all = [], [], []
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        for i in range(n):
            train_idx = [j for j in range(n) if j != i]
            train_trials = [X[j] for j in train_idx]
            y_train = y[train_idx]
            test_trials = [X[i]]
            pipe = ModernBCIPipeline(config)
            pipe.train(train_trials, y_train)
            pred, conf = pipe.predict(test_trials)
            y_true_all.append(y[i])
            y_pred_all.append(pred[0])
            y_conf_all.append(conf[0])
    finally:
        logger.setLevel(prev_level)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_conf_all = np.array(y_conf_all)
    acc = np.mean(y_pred_all == y_true_all)
    tp = np.sum((y_pred_all == 1) & (y_true_all == 1))
    fp = np.sum((y_pred_all == 1) & (y_true_all == 0))
    fn = np.sum((y_pred_all == 0) & (y_true_all == 1))
    tn = np.sum((y_pred_all == 0) & (y_true_all == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
    try:
        auc = _roc_auc(y_true_all, y_conf_all)
    except Exception:
        auc = float('nan')
    return {
        'accuracy': float(acc), 'precision': float(precision), 'recall': float(recall),
        'f1': float(f1), 'roc_auc': float(auc), 'confusion': (int(tn), int(fp), int(fn), int(tp)),
        'y_true': y_true_all, 'y_pred': y_pred_all, 'y_conf': y_conf_all,
    }


# ==============================================================================
# 14) EOG ARTIFACT / CORRELATION / CLEANING ANALYSIS
# ==============================================================================
def run_artifact_checks(config_factory, eeg_channels, eog_channels,
                         signals, windows) -> Dict[str, Dict]:
    """
    Runs LOOCV under three conditions: A) EEG only, B) EOG only, C) EEG+EOG.
    config_factory(channels) -> BCIConfig.
    Returns: {'raw_eeg':..., 'eog_only':..., 'eeg_plus_eog':...,
              '_preproc_eeg':..., '_preproc_eog':...}
    """
    results = {}

    config_eeg = config_factory(eeg_channels)
    pipeline_eeg = ModernBCIPipeline(config_eeg)
    raw_eeg = {ch: signals[ch] for ch in eeg_channels}
    preproc_eeg = pipeline_eeg.preprocess(raw_eeg)
    X_eeg, y_eeg = extract_epochs(preproc_eeg, windows, eeg_channels)
    results['raw_eeg'] = run_loocv(config_eeg, X_eeg, y_eeg)
    results['raw_eeg']['config'] = config_eeg

    config_eog = config_factory(eog_channels)
    pipeline_eog = ModernBCIPipeline(config_eog)
    raw_eog = {ch: signals[ch] for ch in eog_channels}
    preproc_eog = pipeline_eog.preprocess(raw_eog)
    X_eog, y_eog = extract_epochs(preproc_eog, windows, eog_channels)
    results['eog_only'] = run_loocv(config_eog, X_eog, y_eog)

    combined_channels = eeg_channels + eog_channels
    config_combined = config_factory(combined_channels)
    pipeline_combined = ModernBCIPipeline(config_combined)
    raw_combined = {ch: signals[ch] for ch in combined_channels}
    preproc_combined = pipeline_combined.preprocess(raw_combined)
    X_combined, y_combined = extract_epochs(preproc_combined, windows, combined_channels)
    results['eeg_plus_eog'] = run_loocv(config_combined, X_combined, y_combined)

    results['_preproc_eeg'] = preproc_eeg
    results['_preproc_eog'] = preproc_eog
    return results


def run_eog_correlation_analysis(preproc_eeg, preproc_eog, windows, pairs=None) -> List[Dict]:
    """Computes per-window correlation for each (EEG channel, EOG channel) pair."""
    if pairs is None:
        pairs = [('C3', 'EOG_VA'), ('C4', 'EOG_VA'), ('Cz', 'EOG_VA'),
                 ('C3', 'EOG_HL'), ('C4', 'EOG_HL')]
    all_signals_preproc = {**preproc_eeg, **preproc_eog}
    corr_rows = []
    for start_idx, end_idx, label in windows:
        row = {'label': int(label)}
        for ch1, ch2 in pairs:
            s1 = all_signals_preproc[ch1][start_idx:end_idx]
            s2 = all_signals_preproc[ch2][start_idx:end_idx]
            if np.std(s1) > 0 and np.std(s2) > 0:
                row[f"{ch1}-{ch2}"] = float(np.corrcoef(s1, s2)[0, 1])
            else:
                row[f"{ch1}-{ch2}"] = 0.0
        corr_rows.append(row)
    return corr_rows


def run_eog_cleaning_analysis(config_eeg, eeg_channels, preproc_eeg, preproc_eog,
                               eog_channels, windows) -> Dict:
    """
    EEG_clean = EEG - (portion linearly predictable from EOG).
    Regression coefficients are computed on the continuous (unlabeled)
    recording. LOOCV is re-run on the cleaned EEG.
    """
    EOG_mat = np.column_stack([preproc_eog[ch] for ch in eog_channels])
    EOG_design = np.column_stack([EOG_mat, np.ones(len(EOG_mat))])
    preproc_eeg_clean = {}
    r2_per_channel = {}
    for ch in eeg_channels:
        target = preproc_eeg[ch]
        beta, *_ = np.linalg.lstsq(EOG_design, target, rcond=None)
        predicted = EOG_design @ beta
        preproc_eeg_clean[ch] = target - predicted
        r2 = 1 - np.sum((target - predicted) ** 2) / np.sum((target - target.mean()) ** 2)
        r2_per_channel[ch] = float(r2)
    X_clean, y_clean = extract_epochs(preproc_eeg_clean, windows, eeg_channels)
    result_clean = run_loocv(config_eeg, X_clean, y_clean)
    result_clean['r2_per_channel'] = r2_per_channel
    return result_clean


# ==============================================================================
# 15) FINAL MODEL (deployable, trained on ALL usable trials)
# ==============================================================================
def train_final_model(config: BCIConfig, X: List[Dict[str, np.ndarray]],
                       y: np.ndarray) -> ModernBCIPipeline:
    """
    Returns a model trained on all usable trials. This model is NOT tested
    on its own training data (that would leak) - it's only produced for
    future use once a real-time system exists. commands.csv in this file is,
    for honesty, generated from LOOCV's held-out predictions, not from this
    model's own predictions.
    """
    pipe = ModernBCIPipeline(config)
    pipe.train(X, y)
    return pipe


# ==============================================================================
# 16) SYMBOLIC JOYSTICK COMMAND LAYER
# ==============================================================================
def prediction_to_command(prediction: int, confidence: float, threshold: float = 0.6) -> str:
    """
    prediction=1 (WALK) + confidence>=threshold -> "WALK"
    prediction=0 (STOP) + confidence>=threshold -> "STOP"
    confidence<threshold                        -> "IDLE"
    """
    if confidence < threshold:
        return "IDLE"
    return "WALK" if prediction == 1 else "STOP"


class OutputDevice(ABC):
    """Abstract output sink for symbolic joystick commands. Concrete
    real-time backends (e.g. VJoyOutput, ViGEmOutput) should subclass this
    without changing anything upstream in the command-generation logic."""

    @abstractmethod
    def send(self, command: str) -> None:
        ...


class ConsoleOutput(OutputDevice):
    """Symbolic output layer - no real vJoy/HID, just prints to the terminal."""

    def send(self, command: str) -> None:
        if command == "WALK":
            print("[JOYSTICK] FORWARD")
        elif command == "STOP":
            print("[JOYSTICK] STOP")
        else:
            print("[JOYSTICK] IDLE")


def generate_symbolic_commands(y_pred: np.ndarray, y_conf: np.ndarray,
                                threshold: float, emit: bool = True,
                                device: Optional[OutputDevice] = None) -> List[Dict]:
    """
    Generates window-by-window symbolic commands from LOOCV's held-out
    (y_pred, y_conf) pairs. If emit=True, each command is also sent to
    `device` (defaults to ConsoleOutput).
    """
    if emit and device is None:
        device = ConsoleOutput()
    rows = []
    for i, (pred, conf) in enumerate(zip(y_pred, y_conf)):
        command = prediction_to_command(int(pred), float(conf), threshold)
        rows.append({
            'window_index': i, 'predicted_label': int(pred),
            'confidence': float(conf), 'command': command,
        })
        if emit:
            device.send(command)
    return rows


# ==============================================================================
# 17) DATA LOADING / VALIDATION
# ==============================================================================
def load_data(edf_path: str):
    logger.info(f"Reading EDF: {edf_path}")
    signals, info = read_edf(edf_path)
    logger.info(f"{info['n_signals']} channels found")
    return signals, info


def validate_channels(info: Dict, required_eeg: List[str], required_eog: List[str]) -> Dict[str, bool]:
    presence = {}
    for ch in required_eeg + required_eog:
        presence[ch] = ch in info['labels']
        if not presence[ch]:
            logger.warning(f"Missing channel: {ch}")
    return presence


def load_events(events_path: str, label_map: Dict[str, int]) -> Tuple[List, List]:
    """Reads the events file; drops trial_types not present in label_map (e.g. x99)."""
    cmd_events = parse_events(events_path)
    usable, dropped = [], []
    for onset, duration, trial_type in cmd_events:
        if trial_type in label_map:
            usable.append((onset, duration, trial_type, label_map[trial_type]))
        else:
            dropped.append((onset, duration, trial_type))
    logger.info(f"Events: {len(usable)} usable, {len(dropped)} dropped (unmapped code)")
    return usable, dropped


# ==============================================================================
# 18) RESULT SAVING
# ==============================================================================
def save_results(output_dir: str, results: Dict, corr_rows: List[Dict],
                  r2_per_channel: Dict, commands: List[Dict]):
    os.makedirs(output_dir, exist_ok=True)

    metrics_out = {}
    for key in ('raw_eeg', 'eog_only', 'eeg_plus_eog', 'eog_cleaned'):
        r = results[key]
        metrics_out[key] = {
            'accuracy': r['accuracy'], 'precision': r['precision'], 'recall': r['recall'],
            'f1': r['f1'], 'roc_auc': r['roc_auc'], 'confusion': r['confusion'],
        }
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics_out, f, indent=2)

    with open(f"{output_dir}/predictions.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'fold_index', 'true_label', 'predicted_label',
                          'confidence', 'correct'])
        for condition in ('raw_eeg', 'eog_only', 'eeg_plus_eog', 'eog_cleaned'):
            r = results[condition]
            for i, (t, p, c) in enumerate(zip(r['y_true'], r['y_pred'], r['y_conf'])):
                writer.writerow([condition, i, int(t), int(p), float(c), int(t == p)])

    with open(f"{output_dir}/confusion_matrix.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'tn', 'fp', 'fn', 'tp'])
        for condition in ('raw_eeg', 'eog_only', 'eeg_plus_eog', 'eog_cleaned'):
            tn, fp, fn, tp = results[condition]['confusion']
            writer.writerow([condition, tn, fp, fn, tp])

    if corr_rows:
        with open(f"{output_dir}/eog_correlation.csv", 'w', newline='') as f:
            fieldnames = list(corr_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(corr_rows)

    with open(f"{output_dir}/eog_regression_r2.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', 'r2'])
        for ch, r2 in r2_per_channel.items():
            writer.writerow([ch, r2])

    with open(f"{output_dir}/commands.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['window_index', 'predicted_label', 'confidence', 'command'])
        for row in commands:
            writer.writerow([row['window_index'], row['predicted_label'],
                              row['confidence'], row['command']])

    logger.info(f"Results written: {output_dir}/")


def print_summary(results: Dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"{'Condition':>20}{'Accuracy':>12}{'F1':>10}{'ROC-AUC':>10}")
    lines.append("-" * 60)
    for label, key in [('A) EEG only', 'raw_eeg'), ('B) EOG only', 'eog_only'),
                        ('C) EEG+EOG', 'eeg_plus_eog'), ('D) EOG-cleaned EEG', 'eog_cleaned')]:
        r = results[key]
        lines.append(f"{label:>20}{r['accuracy']:>12.2%}{r['f1']:>10.2%}{r['roc_auc']:>10.3f}")
    lines.append("=" * 60)

    result_A = results['raw_eeg']
    result_B = results['eog_only']
    result_C = results['eeg_plus_eog']
    result_clean = results['eog_cleaned']

    lines.append("\nInterpretation:")
    if result_B['accuracy'] < 0.65:
        lines.append(f"  - EOG-only accuracy is low ({result_B['accuracy']:.2%}) -> good sign, "
                      f"eye movement alone doesn't separate Walk/Stop.")
    else:
        lines.append(f"  - EOG-only accuracy is HIGH ({result_B['accuracy']:.2%}) -> WARNING: "
                      f"the model may be learning an artifact; the EEG result is suspect.")

    diff = result_C['accuracy'] - result_A['accuracy']
    if diff > 0.05:
        lines.append(f"  - EEG+EOG is {diff:.2%} higher than EEG-only -> EOG contribution is "
                      f"strong; the EEG-only result may be inflated by artifact.")
    else:
        lines.append(f"  - EEG+EOG ({result_C['accuracy']:.2%}) vs EEG-only "
                      f"({result_A['accuracy']:.2%}) show no large gap -> adding EOG doesn't "
                      f"meaningfully raise the result.")

    drop = result_A['accuracy'] - result_clean['accuracy']
    lines.append(f"  - Accuracy change after EOG cleaning: {-drop:+.2%}")
    if result_clean['accuracy'] >= result_A['accuracy'] - 0.05:
        lines.append("  - Score stays high after cleaning - positive evidence the EEG signal "
                      "isn't dependent on EOG.")
    else:
        lines.append("  - Score dropped noticeably after cleaning - part of the earlier EEG "
                      "result may have come from an EOG artifact.")

    lines.append("\nMost accurate summary sentence:\n"
                  "\"I classified Walk/Stop command periods from C3/C4/Cz EEG recordings, "
                  "controlling for EOG/movement artifact influence.\"\n"
                  "This is different from claiming \"I decoded walking intent from pure brain "
                  "signal.\"")
    text = "\n".join(lines)
    print(text)
    return text


# ==============================================================================
# 19) CLI / MAIN
# ==============================================================================
LABEL_MAP = {'x5': 0, 'x8': 1}  # x5=STOP=0, x8=WALK=1. x99 not in map -> dropped.
REQUIRED_EEG = ['C3', 'C4', 'Cz']
REQUIRED_EOG = ['EOG_HL', 'EOG_HR', 'EOG_VA', 'EOG_VB']
BEST_SHRINKAGE = 0.0
BEST_K = 5


def make_config(channels: List[str], sampling_rate: int,
                 n_features_select: int = BEST_K, shrinkage: float = BEST_SHRINKAGE,
                 use_csp: bool = True) -> BCIConfig:
    return BCIConfig(
        sampling_rate=sampling_rate, n_channels=len(channels), channels=channels,
        notch_freq=60, use_notch=False,  # 60Hz notch impossible at 100Hz sampling (Nyquist=50Hz)
        bandpass_low=0.5, bandpass_high=45,
        fir_order=200, use_laplacian=True, use_csp=use_csp,
        lda_shrinkage=shrinkage, n_features_select=n_features_select,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Walk/Stop BCI validation + symbolic joystick command generation"
    )
    parser.add_argument('--edf', required=True, help="Path to EDF file")
    parser.add_argument('--events', required=True, help="Path to rexcommand events file (TSV preferred, PDF fallback)")
    parser.add_argument('--output-dir', default='results', help="Output directory (default: results)")
    parser.add_argument('--window-len', type=float, default=5.0, help="Window length (seconds)")
    parser.add_argument('--skip-start', type=float, default=1.0, help="Time trimmed from event start (s)")
    parser.add_argument('--skip-end', type=float, default=1.0, help="Time trimmed from event end (s)")
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                         help="Minimum confidence for WALK/STOP (below this -> IDLE)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser


def main():
    args = build_arg_parser().parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("BCI PIPELINE v1.1 - Offline Walk/Stop validation")
    print("=" * 70)

    # 1) Load data + validate channels
    signals, info = load_data(args.edf)
    fs = int(round(info['sampling_rate'][REQUIRED_EEG[0]]))
    validate_channels(info, REQUIRED_EEG, REQUIRED_EOG)

    # 2) Load events
    usable_events, dropped_events = load_events(args.events, LABEL_MAP)

    # 3) Epoch/window extraction
    windows = build_windows(usable_events, fs, args.skip_start, args.skip_end, args.window_len)
    n_stop = sum(1 for _, _, lbl in windows if lbl == 0)
    n_walk = sum(1 for _, _, lbl in windows if lbl == 1)
    logger.info(f"{len(windows)} windows extracted (STOP={n_stop}, WALK={n_walk})")

    config_factory = lambda channels: make_config(channels, fs)

    # 4) Artifact tests (A: EEG, B: EOG, C: EEG+EOG)
    artifact_results = run_artifact_checks(config_factory, REQUIRED_EEG, REQUIRED_EOG,
                                            signals, windows)
    preproc_eeg = artifact_results.pop('_preproc_eeg')
    preproc_eog = artifact_results.pop('_preproc_eog')

    # 5) EOG correlation analysis
    corr_rows = run_eog_correlation_analysis(preproc_eeg, preproc_eog, windows)

    # 6) EOG cleaning test
    config_eeg = config_factory(REQUIRED_EEG)
    result_clean = run_eog_cleaning_analysis(config_eeg, REQUIRED_EEG, preproc_eeg,
                                              preproc_eog, REQUIRED_EOG, windows)
    r2_per_channel = result_clean.pop('r2_per_channel')

    results = {
        'raw_eeg': artifact_results['raw_eeg'],
        'eog_only': artifact_results['eog_only'],
        'eeg_plus_eog': artifact_results['eeg_plus_eog'],
        'eog_cleaned': result_clean,
    }

    # 7) Final model (deployable, trained on all data - NOT used for commands.csv)
    X_eeg, y_eeg = extract_epochs(preproc_eeg, windows, REQUIRED_EEG)
    _final_pipe = train_final_model(config_eeg, X_eeg, y_eeg)

    # 8) Symbolic commands - generated from LOOCV held-out predictions, for honesty
    commands = generate_symbolic_commands(
        results['raw_eeg']['y_pred'], results['raw_eeg']['y_conf'],
        threshold=args.confidence_threshold, emit=True
    )

    # 9) Summary + save
    summary_text = print_summary(results)
    save_results(args.output_dir, results, corr_rows, r2_per_channel, commands)
    with open(f"{args.output_dir}/summary.txt", 'w') as f:
        f.write(summary_text)

    print("\n" + "=" * 70)
    print("This is an offline validation and symbolic joystick command generation")
    print("script. It does not provide real-time EEG streaming or real HID/vJoy")
    print("control yet.")
    print("=" * 70)


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------
# SCOPE DISCLAIMER (repeated here deliberately, per project requirement):
#
# This is an offline validation and symbolic joystick command generation script.
# It does not provide real-time EEG streaming or real HID/vJoy control yet.
# -------------------------------------------------------------------------------
