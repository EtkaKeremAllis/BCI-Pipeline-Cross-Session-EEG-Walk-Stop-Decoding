#!/usr/bin/env python3
"""
bci_pipeline.py
===============================================================================
Offline + deployable Walk/Stop BCI pipeline (single-file, v2.0)

EEG (C3/C4/Cz) -> preprocessing -> CSP + feature extraction -> feature
selection -> shrinkage LDA -> LOOCV validation -> EOG artifact analysis ->
symbolic/real joystick command generation.

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

This file supports two modes:
  train_validate: labeled EDF + events -> LOOCV validation + final model artifact
  predict: unlabeled raw EDF + saved model -> predicted timeline + optional joystick output
It still does not implement live EEG streaming; predict mode is sliding-window
inference/replay over a fixed EDF recording. Every "joystick command" produced here is a printed label
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

v1.2 changes (second code-review pass):
  - The final model (trained on all usable trials, config_eeg/REQUIRED_EEG)
    is now actually persisted to disk as trained_model.npz (LDA weights,
    CSP filters, selected feature indices) instead of being built and
    discarded in memory.
  - selected_features.json now records which named features the F-score
    selector kept (and their scores), not just their raw indices.
  - csp_filters.npy is written as a standalone array for anyone who wants
    the spatial filters without touching the .npz bundle.
  - model_info.json is written as a model card: sampling rate, channels,
    feature count, CSP filter count, LDA shrinkage, LOOCV accuracy, and
    basic dataset provenance (EDF path, patient/recording id, n windows).
  - roc_curve.png, confusion_matrices.png, and accuracy_comparison.png are
    generated from the same LOOCV results already computed, for anyone
    browsing the repo without running the script.
  Note: the trained model artifact is for future reuse only. commands.csv
  is still generated exclusively from LOOCV's held-out predictions, never
  from this saved model's own predictions on its training data.

Known future work (not implemented here, flagged deliberately rather than
silently omitted):
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
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import signal, stats
from scipy.linalg import eig

import matplotlib
matplotlib.use('Agg')  # headless: never try to open a GUI window
import matplotlib.pyplot as plt

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
# These two lists mirror the exact key-insertion order inside
# extract_time_domain() / extract_frequency_domain() below. They exist so we
# can label the final feature vector by name (selected_features.json,
# model_info.json) without having to re-run extraction just to recover names.
TIME_FEATURE_NAMES = [
    'mean', 'std', 'var', 'min', 'max', 'range', 'rms', 'peak_to_peak',
    'kurtosis', 'skewness', 'line_length', 'hjorth_activity',
    'hjorth_mobility', 'hjorth_complexity', 'zero_crossings', 'spectral_entropy',
]
FREQ_FEATURE_NAMES = [
    'Delta_power', 'Theta_power', 'Alpha_power', 'Mu_power', 'Beta_power',
    'Gamma_power', 'mu_beta_ratio', 'beta_mu_ratio', 'alpha_theta_ratio',
    'spectral_centroid', 'total_power',
]


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

    def feature_name_list(self, present_channels: List[str]) -> List[str]:
        """
        Returns feature names in exactly the order extract_full_features()
        concatenates them, so any downstream index (e.g. a FeatureSelector's
        selected_idx) can be turned into a human-readable label. Mirrors:
          for ch in sorted(present_channels): time-domain names, freq-domain names
          [asymmetry] if >=2 channels
          [csp_logvar_i] * csp_n_filters, if CSP is enabled
        """
        names = []
        sorted_channels = sorted(present_channels)
        for ch in sorted_channels:
            for feat_name in TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES:
                names.append(f"{ch}_{feat_name}")
        if len(sorted_channels) >= 2:
            names.append(f"asymmetry_{sorted_channels[0]}_{sorted_channels[1]}")
        if self.config.use_csp and self.csp is not None and self.csp.filters is not None:
            # Use the actual fitted filter count, not the configured target:
            # eig() can only return as many spatial filters as there are
            # channels, so csp_n_filters is silently capped when n_channels
            # < csp_n_filters. Naming must follow the real fitted shape or
            # names/vector lengths drift apart.
            n_fitted = self.csp.filters.shape[1]
            for i in range(n_fitted):
                names.append(f"csp_logvar_{i}")
        return names

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


def build_prediction_windows(signal_len: int, fs: int, window_len: float,
                             step_len: float) -> List[Tuple[int, int, float, float]]:
    """Build sliding windows for unlabeled inference.

    Returns [(start_idx, end_idx, start_time, end_time), ...].
    No event labels are used here.
    """
    win_samples = int(round(window_len * fs))
    step_samples = int(round(step_len * fs))
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("window_len and step_len must be positive")
    windows = []
    start = 0
    while start + win_samples <= signal_len:
        end = start + win_samples
        windows.append((start, end, start / fs, end / fs))
        start += step_samples
    return windows


def extract_prediction_epochs(preprocessed_dict: Dict[str, np.ndarray],
                              windows: List[Tuple[int, int, float, float]],
                              channels: List[str]) -> List[Dict[str, np.ndarray]]:
    X = []
    max_len = len(next(iter(preprocessed_dict.values())))
    for start_idx, end_idx, _, _ in windows:
        if end_idx > max_len:
            continue
        X.append({ch: preprocessed_dict[ch][start_idx:end_idx] for ch in channels})
    return X


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


def save_trained_model(output_dir: str, pipe: ModernBCIPipeline, channels: List[str]) -> Dict:
    """
    Persists the final deployable model to trained_model.npz, plus a
    standalone csp_filters.npy and a selected_features.json naming which
    features the F-score selector kept.

    Returns a small dict of what was saved, so callers (e.g. main()) can
    fold the same numbers into model_info.json without recomputing them.
    """
    clf = pipe.classifier
    feature_names = pipe.feature_name_list(channels)

    npz_payload = {
        'lda_coef': clf.coef,
        'lda_intercept': np.array(clf.intercept),
        'lda_mean_0': clf.mean_0,
        'lda_mean_1': clf.mean_1,
        'lda_shrinkage': np.array(pipe.config.lda_shrinkage),
    }

    selected_idx = None
    f_scores = None
    if pipe.feature_selector is not None and pipe.feature_selector.selected_idx is not None:
        selected_idx = pipe.feature_selector.selected_idx
        f_scores = pipe.feature_selector.f_scores_
        npz_payload['selected_feature_idx'] = selected_idx

    csp_filters = None
    if pipe.csp is not None and pipe.csp.filters is not None:
        csp_filters = pipe.csp.filters
        npz_payload['csp_filters'] = csp_filters
        np.save(f"{output_dir}/csp_filters.npy", csp_filters)

    np.savez(f"{output_dir}/trained_model.npz", **npz_payload)

    if selected_idx is not None:
        selected_features = []
        for rank, idx in enumerate(selected_idx):
            entry = {
                'rank': rank,
                'feature_index': int(idx),
                'feature_name': feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
            }
            if f_scores is not None:
                entry['f_score'] = float(f_scores[idx])
            selected_features.append(entry)
        with open(f"{output_dir}/selected_features.json", 'w') as f:
            json.dump(selected_features, f, indent=2)

    return {
        'n_features_total': len(feature_names),
        'n_features_selected': int(len(selected_idx)) if selected_idx is not None else len(feature_names),
        'csp_n_filters': int(csp_filters.shape[1]) if csp_filters is not None else 0,
    }


# ==============================================================================
# 16) MODEL LOADING + JOYSTICK COMMAND LAYER
# ==============================================================================
def config_to_dict(config: BCIConfig) -> Dict:
    return asdict(config)


def config_from_dict(data: Dict) -> BCIConfig:
    allowed = set(BCIConfig.__dataclass_fields__.keys())
    clean = {k: v for k, v in (data or {}).items() if k in allowed}
    return BCIConfig(**clean)


def prediction_confidence(prediction: int, walk_probability: float) -> float:
    """Return confidence for the predicted class, not always P(WALK)."""
    walk_probability = float(walk_probability)
    return walk_probability if int(prediction) == 1 else 1.0 - walk_probability


def prediction_to_command(prediction: int, confidence: float, threshold: float = 0.6) -> str:
    """
    prediction=1 (WALK) + confidence>=threshold -> "WALK"
    prediction=0 (STOP) + confidence>=threshold -> "STOP"
    confidence<threshold                        -> "IDLE"
    """
    if confidence < threshold:
        return "IDLE"
    return "WALK" if prediction == 1 else "STOP"


def load_trained_model(model_path: str, model_info_path: Optional[str] = None) -> ModernBCIPipeline:
    """
    Rebuilds a ModernBCIPipeline from trained_model.npz (+ model_info.json).
    No fitting is performed here; predict mode only calls predict().
    """
    if model_info_path is None:
        candidate = os.path.join(os.path.dirname(model_path), 'model_info.json')
        model_info_path = candidate if os.path.exists(candidate) else None

    info = {}
    if model_info_path and os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            info = json.load(f)

    model = np.load(model_path, allow_pickle=True)
    cfg_dict = info.get('config')
    if cfg_dict:
        config = config_from_dict(cfg_dict)
    else:
        channels = info.get('channels', REQUIRED_EEG)
        fs = int(info.get('sampling_rate_hz', 256))
        shrinkage = float(model['lda_shrinkage']) if 'lda_shrinkage' in model else BEST_SHRINKAGE
        k = int(len(model['selected_feature_idx'])) if 'selected_feature_idx' in model else BEST_K
        config = make_config(channels, fs, n_features_select=k, shrinkage=shrinkage)
        config.confidence_threshold = float(info.get('confidence_threshold', config.confidence_threshold))

    pipe = ModernBCIPipeline(config)

    if 'csp_filters' in model and pipe.csp is not None:
        pipe.csp.filters = model['csp_filters']
        pipe.csp.n_components = pipe.csp.filters.shape[1]

    if 'selected_feature_idx' in model and pipe.feature_selector is not None:
        pipe.feature_selector.selected_idx = model['selected_feature_idx'].astype(int)

    pipe.classifier.coef = model['lda_coef']
    pipe.classifier.intercept = float(np.asarray(model['lda_intercept']))
    pipe.classifier.mean_0 = model['lda_mean_0'] if 'lda_mean_0' in model else None
    pipe.classifier.mean_1 = model['lda_mean_1'] if 'lda_mean_1' in model else None
    if 'lda_shrinkage' in model:
        pipe.classifier.shrinkage = float(np.asarray(model['lda_shrinkage']))

    logger.info(
        f"Loaded trained model: channels={pipe.config.channels}, "
        f"fs={pipe.config.sampling_rate}Hz, threshold={pipe.config.confidence_threshold}"
    )
    return pipe


class OutputDevice(ABC):
    """Abstract output sink for joystick commands."""

    @abstractmethod
    def send(self, command: str) -> None:
        ...

    def close(self) -> None:
        pass


class ConsoleOutput(OutputDevice):
    """Safe default: prints command changes, without touching real devices."""

    def __init__(self):
        self.last_printed = None

    def send(self, command: str) -> None:
        # Console output is intentionally quiet on repeated states. Real joystick
        # backends refresh the axis every call.
        if command == self.last_printed:
            return
        self.last_printed = command
        if command == "WALK":
            print("[JOYSTICK] FORWARD")
        elif command == "STOP":
            print("[JOYSTICK] STOP / NEUTRAL")
        else:
            print("[JOYSTICK] IDLE / NEUTRAL")


class ViGEmOutput(OutputDevice):
    """Windows virtual Xbox controller via pyvgamepad/ViGEmBus."""

    def __init__(self):
        try:
            import pyvgamepad as vg
        except Exception as exc:
            raise RuntimeError(
                "ViGEm backend requires ViGEmBus + pyvgamepad. "
                "Install/configure them first, or use --output-backend console."
            ) from exc
        self.vg = vg
        self.gamepad = vg.VX360Gamepad()
        self.send("STOP")

    def send(self, command: str) -> None:
        if command == "WALK":
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=1.0)
        else:
            self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.update()

    def close(self) -> None:
        try:
            self.send("STOP")
        except Exception:
            pass


class VJoyOutput(OutputDevice):
    """Windows vJoy backend via pyvjoy."""

    def __init__(self, device_id: int = 1):
        try:
            import pyvjoy
        except Exception as exc:
            raise RuntimeError(
                "vJoy backend requires the vJoy driver + pyvjoy. "
                "Install/configure them first, or use --output-backend console."
            ) from exc
        self.pyvjoy = pyvjoy
        self.device = pyvjoy.VJoyDevice(device_id)
        self.neutral = 0x4000
        self.forward = 0x8000
        self.send("STOP")

    def send(self, command: str) -> None:
        # vJoy axis ranges are driver-dependent; this common 0x0000..0x8000
        # convention maps WALK to full forward on Y, otherwise neutral.
        value = self.forward if command == "WALK" else self.neutral
        self.device.set_axis(self.pyvjoy.HID_USAGE_Y, value)

    def close(self) -> None:
        try:
            self.send("STOP")
        except Exception:
            pass


def make_output_device(backend: str) -> OutputDevice:
    backend = (backend or 'console').lower()
    if backend == 'console':
        return ConsoleOutput()
    if backend == 'vigem':
        try:
            return ViGEmOutput()
        except Exception as exc:
            logger.error(f"ViGEm backend unavailable ({exc}); falling back to console.")
            return ConsoleOutput()
    if backend == 'vjoy':
        try:
            return VJoyOutput()
        except Exception as exc:
            logger.error(f"vJoy backend unavailable ({exc}); falling back to console.")
            return ConsoleOutput()
    raise ValueError(f"Unknown output backend: {backend}")


class CommandSmoother:
    def __init__(self, majority_window: int = 5, min_confidence: float = 0.6, cooldown_s: float = 0.5):
        self.majority_window = int(majority_window)
        self.min_confidence = float(min_confidence)
        self.cooldown_s = float(cooldown_s)
        self.history = deque(maxlen=self.majority_window)
        self.current_command = "IDLE"
        self.last_change_time = -1e9

    def update(self, raw_command: str, confidence: float, now_s: float) -> str:
        command = raw_command if confidence >= self.min_confidence else "IDLE"
        self.history.append(command)
        counts = Counter(self.history)
        majority_command = counts.most_common(1)[0][0]
        if majority_command != self.current_command:
            if (now_s - self.last_change_time) >= self.cooldown_s:
                self.current_command = majority_command
                self.last_change_time = now_s
        return self.current_command


def generate_symbolic_commands(y_pred: np.ndarray, y_conf: np.ndarray,
                                threshold: float, emit: bool = True,
                                device: Optional[OutputDevice] = None) -> List[Dict]:
    """
    Generates window-by-window symbolic commands from LOOCV held-out
    predictions. y_conf is P(WALK), so command confidence is converted to
    the predicted class confidence before thresholding.
    """
    if emit and device is None:
        device = ConsoleOutput()
    rows = []
    for i, (pred, p_walk) in enumerate(zip(y_pred, y_conf)):
        conf = prediction_confidence(int(pred), float(p_walk))
        command = prediction_to_command(int(pred), conf, threshold)
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
# 18b) PLOTS
# ==============================================================================
def _roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple dependency-free ROC curve: sweep the observed confidence scores
    as thresholds and compute (FPR, TPR) at each. Same y_true/y_score already
    used for the AUC in run_loocv - this just also keeps the curve, not only
    the summary number."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.unique(y_score)[::-1]
    thresholds = np.concatenate([[thresholds[0] + 1e-6], thresholds, [thresholds[-1] - 1e-6]])
    tpr, fpr = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tpr.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr.append(fp / n_neg if n_neg > 0 else 0.0)
    return np.array(fpr), np.array(tpr), thresholds


def plot_roc_curve(results: Dict, output_dir: str, condition: str = 'raw_eeg'):
    r = results[condition]
    fpr, tpr, _ = _roc_curve_points(r['y_true'], r['y_conf'])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, marker='o', markersize=3, label=f"AUC = {r['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"ROC Curve - {condition} (LOOCV held-out predictions)")
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/roc_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results: Dict, output_dir: str):
    conditions = [('raw_eeg', 'A) EEG only'), ('eog_only', 'B) EOG only'),
                  ('eeg_plus_eog', 'C) EEG+EOG'), ('eog_cleaned', 'D) EOG-cleaned EEG')]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (key, title) in zip(axes, conditions):
        tn, fp, fn, tp = results[key]['confusion']
        mat = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(mat, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                        color='white' if mat[i, j] > mat.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred STOP', 'Pred WALK'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['True STOP', 'True WALK'])
        ax.set_title(title, fontsize=10)
    fig.suptitle('Confusion Matrices (LOOCV)')
    fig.tight_layout()
    fig.savefig(f"{output_dir}/confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_accuracy_comparison(results: Dict, output_dir: str):
    labels = ['A) EEG only', 'B) EOG only', 'C) EEG+EOG', 'D) EOG-cleaned EEG']
    keys = ['raw_eeg', 'eog_only', 'eeg_plus_eog', 'eog_cleaned']
    accs = [results[k]['accuracy'] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, accs, color=['#4C72B0', '#C44E52', '#55A868', '#8172B2'])
    ax.axhline(0.5, linestyle='--', color='gray', label='Chance (2-class)')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('LOOCV Accuracy')
    ax.set_title('Accuracy by Condition')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.1%}", ha='center', fontsize=9)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(f"{output_dir}/accuracy_comparison.png", dpi=150)
    plt.close(fig)


# ======================================================================
# 19) CLI / MAIN
# ======================================================================
LABEL_MAP = {'x5': 0, 'x8': 1}  # x5=STOP=0, x8=WALK=1. x99 not in map -> dropped.
REQUIRED_EEG = ['C3', 'C4', 'Cz']
REQUIRED_EOG = ['EOG_HL', 'EOG_HR', 'EOG_VA', 'EOG_VB']
BEST_SHRINKAGE = 0.0
BEST_K = 5

SCIENCE_NOTE = (
    "This prediction mode estimates Walk/Stop command periods from EEG using a "
    "previously trained model. It does not prove pure brain-only walking intention "
    "decoding, and performance depends on training data, artifacts, and "
    "subject/session similarity."
)
GENERALIZATION_RISK = (
    "Important technical risk: if the model was trained on sub-01 or one specific "
    "session, it may generalize poorly to another subject, another day, another cap "
    "placement, or a noisier recording."
)


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
        description="Walk/Stop BCI: labeled validation/training or unlabeled EDF prediction"
    )
    parser.add_argument('--mode', choices=['train_validate', 'predict'], default='train_validate',
                        help="train_validate = labeled LOOCV + final model; predict = unlabeled EDF inference")
    parser.add_argument('--edf', required=True, help="Path to EDF file")
    parser.add_argument('--events', required=False,
                        help="Path to labeled rexcommand events file; required only in train_validate mode")
    parser.add_argument('--model', required=False, help="Path to trained_model.npz; required only in predict mode")
    parser.add_argument('--model-info', required=False,
                        help="Optional path to model_info.json; defaults to model directory/model_info.json")
    parser.add_argument('--output-dir', default='results', help="Output directory")
    parser.add_argument('--window-len', type=float, default=5.0, help="Window length (seconds)")
    parser.add_argument('--step-len', type=float, default=1.0, help="Prediction sliding-window step length (seconds)")
    parser.add_argument('--skip-start', type=float, default=1.0, help="Time trimmed from event start (s), train_validate only")
    parser.add_argument('--skip-end', type=float, default=1.0, help="Time trimmed from event end (s), train_validate only")
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                        help="Minimum predicted-class confidence for WALK/STOP; below this -> IDLE")
    parser.add_argument('--output-backend', choices=['console', 'vigem', 'vjoy'], default='console',
                        help="Joystick output backend. Default console is safest.")
    parser.add_argument('--replay', action='store_true',
                        help="In predict mode, emit window commands sequentially as a replay over the EDF")
    parser.add_argument('--replay-speed', type=float, default=1.0,
                        help="Replay speed: 1.0 real-time, 5.0 five times faster, 0 no waiting")
    parser.add_argument('--majority-window', type=int, default=5, help="Command smoother majority window")
    parser.add_argument('--cooldown-s', type=float, default=0.5, help="Minimum seconds between smoothed command changes")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser


def run_train_validate(args) -> None:
    if not args.events:
        raise ValueError("--events is required in --mode train_validate")

    print("=" * 70)
    print("BCI PIPELINE v2.0 - train_validate mode")
    print("=" * 70)

    signals, info = load_data(args.edf)
    fs = int(round(info['sampling_rate'][REQUIRED_EEG[0]]))
    validate_channels(info, REQUIRED_EEG, REQUIRED_EOG)

    usable_events, dropped_events = load_events(args.events, LABEL_MAP)
    windows = build_windows(usable_events, fs, args.skip_start, args.skip_end, args.window_len)
    n_stop = sum(1 for _, _, lbl in windows if lbl == 0)
    n_walk = sum(1 for _, _, lbl in windows if lbl == 1)
    logger.info(f"{len(windows)} windows extracted (STOP={n_stop}, WALK={n_walk})")

    config_factory = lambda channels: make_config(channels, fs)

    artifact_results = run_artifact_checks(config_factory, REQUIRED_EEG, REQUIRED_EOG, signals, windows)
    preproc_eeg = artifact_results.pop('_preproc_eeg')
    preproc_eog = artifact_results.pop('_preproc_eog')

    corr_rows = run_eog_correlation_analysis(preproc_eeg, preproc_eog, windows)

    config_eeg = config_factory(REQUIRED_EEG)
    config_eeg.confidence_threshold = args.confidence_threshold
    result_clean = run_eog_cleaning_analysis(config_eeg, REQUIRED_EEG, preproc_eeg,
                                             preproc_eog, REQUIRED_EOG, windows)
    r2_per_channel = result_clean.pop('r2_per_channel')

    results = {
        'raw_eeg': artifact_results['raw_eeg'],
        'eog_only': artifact_results['eog_only'],
        'eeg_plus_eog': artifact_results['eeg_plus_eog'],
        'eog_cleaned': result_clean,
    }

    X_eeg, y_eeg = extract_epochs(preproc_eeg, windows, REQUIRED_EEG)
    final_pipe = train_final_model(config_eeg, X_eeg, y_eeg)

    commands = generate_symbolic_commands(
        results['raw_eeg']['y_pred'], results['raw_eeg']['y_conf'],
        threshold=args.confidence_threshold, emit=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_text = print_summary(results)
    summary_text += "\n\n" + GENERALIZATION_RISK + "\n"
    save_results(args.output_dir, results, corr_rows, r2_per_channel, commands)
    with open(f"{args.output_dir}/summary.txt", 'w') as f:
        f.write(summary_text)

    model_save_info = save_trained_model(args.output_dir, final_pipe, REQUIRED_EEG)

    model_info = {
        'mode': 'train_validate',
        'sampling_rate_hz': fs,
        'channels': REQUIRED_EEG,
        'config': config_to_dict(config_eeg),
        'n_features_total': model_save_info['n_features_total'],
        'n_features_selected': model_save_info['n_features_selected'],
        'csp_n_filters': model_save_info['csp_n_filters'],
        'lda_shrinkage': config_eeg.lda_shrinkage,
        'confidence_threshold': args.confidence_threshold,
        'loocv_accuracy_raw_eeg': results['raw_eeg']['accuracy'],
        'loocv_roc_auc_raw_eeg': results['raw_eeg']['roc_auc'],
        'loocv_f1_raw_eeg': results['raw_eeg']['f1'],
        'dataset': {
            'edf_path': args.edf,
            'events_path': args.events,
            'patient_id': info.get('patient_id', ''),
            'recording_id': info.get('recording_id', ''),
            'n_windows': len(windows),
            'n_stop_windows': n_stop,
            'n_walk_windows': n_walk,
        },
        'scope_note': (
            "train_validate mode uses labeled EEG and event annotations to validate "
            "the model and train a final deployable model. commands.csv reflects "
            "LOOCV held-out predictions, not the final model's predictions on its own training data."
        ),
        'generalization_risk': GENERALIZATION_RISK,
    }
    with open(f"{args.output_dir}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    plot_roc_curve(results, args.output_dir, condition='raw_eeg')
    plot_confusion_matrices(results, args.output_dir)
    plot_accuracy_comparison(results, args.output_dir)

    print("\n" + "=" * 70)
    print("train_validate complete: validation outputs + final trained_model.npz written.")
    print("Use --mode predict with a raw unlabeled EDF to generate predicted_timeline.csv.")
    print("=" * 70)


def predict_unlabeled_timeline(pipe: ModernBCIPipeline,
                               preprocessed_eeg: Dict[str, np.ndarray],
                               windows: List[Tuple[int, int, float, float]],
                               threshold: float,
                               smoother: CommandSmoother,
                               output: Optional[OutputDevice] = None,
                               replay: bool = False,
                               replay_speed: float = 1.0) -> Tuple[List[Dict], List[Dict]]:
    epochs = extract_prediction_epochs(preprocessed_eeg, windows, pipe.config.channels)
    if not epochs:
        raise ValueError("No prediction windows could be built. Check EDF length/window_len/step_len.")

    y_pred, p_walk = pipe.predict(epochs)
    rows = []
    command_log = []

    previous_wall_time = time.time()
    for i, ((start_idx, end_idx, start_time_s, end_time_s), pred, p1) in enumerate(zip(windows, y_pred, p_walk)):
        conf = prediction_confidence(int(pred), float(p1))
        raw_command = prediction_to_command(int(pred), conf, threshold)
        smoothed_command = smoother.update(raw_command, conf, start_time_s)

        row = {
            'window_index': i,
            'start_time': float(start_time_s),
            'end_time': float(end_time_s),
            'predicted_label': int(pred),
            'confidence': float(conf),
            'p_walk': float(p1),
            'raw_command': raw_command,
            'smoothed_command': smoothed_command,
        }
        rows.append(row)

        if output is not None and replay:
            output.send(smoothed_command)
            command_log.append({
                'window_index': i,
                'sent_time_wall_clock': time.time(),
                'edf_start_time': float(start_time_s),
                'command': smoothed_command,
            })
            if replay_speed > 0 and i < len(windows) - 1:
                next_start = windows[i + 1][2]
                wait_s = max(0.0, (next_start - start_time_s) / replay_speed)
                time.sleep(wait_s)
                previous_wall_time = time.time()

    return rows, command_log


def save_prediction_outputs(output_dir: str, rows: List[Dict], command_log: List[Dict],
                            args, pipe: ModernBCIPipeline, info: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)

    timeline_path = f"{output_dir}/predicted_timeline.csv"
    with open(timeline_path, 'w', newline='') as f:
        fieldnames = ['window_index', 'start_time', 'end_time', 'predicted_label',
                      'confidence', 'p_walk', 'raw_command', 'smoothed_command']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    commands_path = f"{output_dir}/predicted_commands.csv"
    with open(commands_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['window_index', 'start_time', 'end_time', 'raw_command', 'smoothed_command'])
        for row in rows:
            writer.writerow([row['window_index'], row['start_time'], row['end_time'],
                             row['raw_command'], row['smoothed_command']])

    if command_log:
        with open(f"{output_dir}/command_log.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['window_index', 'sent_time_wall_clock', 'edf_start_time', 'command'])
            writer.writeheader()
            writer.writerows(command_log)

    counts_raw = Counter(row['raw_command'] for row in rows)
    counts_smooth = Counter(row['smoothed_command'] for row in rows)
    mean_conf = float(np.mean([row['confidence'] for row in rows])) if rows else float('nan')
    summary_lines = [
        "BCI prediction summary",
        "======================",
        "No ground-truth events were provided. This is inference only; accuracy cannot be computed.",
        SCIENCE_NOTE,
        GENERALIZATION_RISK,
        "",
        f"EDF: {args.edf}",
        f"Model: {args.model}",
        f"Channels: {pipe.config.channels}",
        f"Sampling rate: {pipe.config.sampling_rate} Hz",
        f"Window length: {args.window_len} s",
        f"Step length: {args.step_len} s",
        f"Confidence threshold: {args.confidence_threshold}",
        f"Replay: {args.replay}",
        f"Output backend: {args.output_backend}",
        f"Number of windows: {len(rows)}",
        f"Mean predicted-class confidence: {mean_conf:.3f}",
        f"Raw command counts: {dict(counts_raw)}",
        f"Smoothed command counts: {dict(counts_smooth)}",
        "",
        "Outputs:",
        "- predicted_timeline.csv: raw + smoothed window-by-window inference",
        "- predicted_commands.csv: compact command timeline",
        "- command_log.csv: only written when replay emitted commands",
    ]
    with open(f"{output_dir}/prediction_summary.txt", 'w') as f:
        f.write("\n".join(summary_lines) + "\n")

    logger.info(f"Prediction outputs written: {output_dir}/")


def run_predict(args) -> None:
    if not args.model:
        raise ValueError("--model is required in --mode predict")

    print("=" * 70)
    print("BCI PIPELINE v2.0 - predict mode")
    print("=" * 70)
    print("No ground-truth events were provided. This is inference only; accuracy cannot be computed.")
    print(SCIENCE_NOTE)

    pipe = load_trained_model(args.model, args.model_info)
    pipe.config.confidence_threshold = args.confidence_threshold

    signals, info = load_data(args.edf)
    missing = [ch for ch in pipe.config.channels if ch not in signals]
    if missing:
        raise ValueError(f"Predict EDF is missing model-required channels: {missing}")

    fs_file = int(round(info['sampling_rate'][pipe.config.channels[0]]))
    if fs_file != int(pipe.config.sampling_rate):
        logger.warning(
            f"EDF sampling rate ({fs_file}Hz) differs from model sampling rate "
            f"({pipe.config.sampling_rate}Hz). This script does not resample; results may be invalid."
        )

    raw_eeg = {ch: signals[ch] for ch in pipe.config.channels}
    preproc_eeg = pipe.preprocess(raw_eeg)
    signal_len = min(len(preproc_eeg[ch]) for ch in pipe.config.channels)
    windows = build_prediction_windows(signal_len, pipe.config.sampling_rate, args.window_len, args.step_len)
    logger.info(f"{len(windows)} unlabeled prediction windows built")

    smoother = CommandSmoother(
        majority_window=args.majority_window,
        min_confidence=args.confidence_threshold,
        cooldown_s=args.cooldown_s,
    )
    output = make_output_device(args.output_backend)

    try:
        rows, command_log = predict_unlabeled_timeline(
            pipe, preproc_eeg, windows, args.confidence_threshold,
            smoother=smoother,
            output=output,
            replay=args.replay,
            replay_speed=args.replay_speed,
        )
    except KeyboardInterrupt:
        logger.warning("Ctrl+C received. Sending neutral command before exit.")
        raise
    finally:
        # Safety: real backends must always release the stick to neutral.
        try:
            output.send("STOP")
        finally:
            output.close()

    save_prediction_outputs(args.output_dir, rows, command_log, args, pipe, info)
    print("\n" + "=" * 70)
    print("predict complete: predicted_timeline.csv and predicted_commands.csv written.")
    print(GENERALIZATION_RISK)
    print("=" * 70)


def main():
    args = build_arg_parser().parse_args()
    np.random.seed(args.seed)
    if args.mode == 'train_validate':
        run_train_validate(args)
    elif args.mode == 'predict':
        run_predict(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------
# SCOPE DISCLAIMER:
#
# train_validate mode uses labeled EEG + event annotations for LOOCV validation
# and final deployable model training. predict mode uses only raw EDF + a saved
# trained model; no events are read and no accuracy is computed.
# -------------------------------------------------------------------------------
