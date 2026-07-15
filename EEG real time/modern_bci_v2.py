"""
modern_bci_v2.py
================================================================================
BCI sinyal isleme + CSP + feature selection + shrinkage LDA.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import signal, stats
from scipy.linalg import eig

import warnings
warnings.filterwarnings('ignore')


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        return f"[{record.levelname}] {record.getMessage()}"


def setup_logging(name="BCI", level=logging.INFO):
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    lg.addHandler(handler)
    return lg


logger = setup_logging()


@dataclass
class BCIConfig:
    sampling_rate: int = 256
    n_channels: int = 3
    channels: List[str] = None
    notch_freq: int = 50
    use_notch: bool = True
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


class CSPFilter:
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.filters is None:
            return X
        if X.ndim == 2:
            return (self.filters.T @ X).flatten()
        return np.array([(self.filters.T @ trial).flatten() for trial in X])

    def transform_logvar(self, X: np.ndarray) -> np.ndarray:
        if self.filters is None:
            return np.array([])
        if X.ndim == 2:
            Z = self.filters.T @ X
            var = np.var(Z, axis=1)
            var_norm = var / (np.sum(var) + 1e-10)
            return np.log(var_norm + 1e-10)
        return np.array([self.transform_logvar(trial) for trial in X])


class FeatureSelector:
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


class SimpleLDA:
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
        decision = np.clip(decision, -500, 500)
        proba = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - proba, proba])


class FIRBandpassFilter:
    def __init__(self, low_freq: float, high_freq: float, sampling_rate: int, order: int = 256):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = sampling_rate
        self.order = order if order % 2 == 1 else order + 1
        nyquist = sampling_rate / 2
        # pass_zero=False is required: with two cutoff frequencies, firwin's
        # default (pass_zero=True) builds a BANDSTOP filter (passes DC and
        # near-Nyquist, blocks the band in between) instead of a bandpass
        # filter. Without this, [low_freq, high_freq] was being blocked
        # rather than kept - i.e. the entire useful EEG band was discarded
        # and near-DC drift / near-Nyquist noise was kept instead.
        self.taps = signal.firwin(
            self.order, [low_freq / nyquist, high_freq / nyquist], window='hamming',
            pass_zero=False,
        )

    def apply_offline(self, signal_data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(self.taps, 1, signal_data)


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
                    f"notch filtre atlaniyor (sampling_rate cok dusuk)."
                )
        self.csp = CSPFilter(n_filters=self.config.csp_n_filters) if self.config.use_csp else None
        self.classifier = SimpleLDA(shrinkage=self.config.lda_shrinkage)
        self.feature_selector = (
            FeatureSelector(k=self.config.n_features_select)
            if self.config.use_feature_selection else None
        )
        self.metrics = BCIMetrics()
        self.command_feature_mean = None
        self.command_feature_std = None

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
        names = []
        sorted_channels = sorted(present_channels)
        for ch in sorted_channels:
            for feat_name in TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES:
                names.append(f"{ch}_{feat_name}")
        if len(sorted_channels) >= 2:
            names.append(f"asymmetry_{sorted_channels[0]}_{sorted_channels[1]}")
        if self.config.use_csp and self.csp is not None and self.csp.filters is not None:
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

        self.command_feature_mean = X.mean(axis=0)
        self.command_feature_std = X.std(axis=0)
        self.command_feature_std[self.command_feature_std < 1e-8] = 1e-8

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

    def predict_detailed(self, raw_trials: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        features_full = np.array([self.extract_full_features(t) for t in raw_trials])
        X = features_full
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        predictions = self.classifier.predict(X)
        proba = self.classifier.predict_proba(X)
        proba_walk = proba[:, 1]
        decision_confidence = np.where(predictions == 1, proba[:, 1], proba[:, 0])

        z_distance = None
        if self.command_feature_mean is not None:
            z = np.abs((features_full - self.command_feature_mean) / self.command_feature_std)
            z_distance = z.mean(axis=1)

        return {
            'prediction': predictions, 'proba_walk': proba_walk,
            'decision_confidence': decision_confidence,
            'features_full': features_full, 'z_distance': z_distance,
        }


def k_fold_cross_validate(config: BCIConfig, raw_trials: List[Dict[str, np.ndarray]],
                           labels: np.ndarray, k_folds: int = 5) -> np.ndarray:
    y = np.asarray(labels)
    idx = np.arange(len(raw_trials))
    np.random.shuffle(idx)
    folds = np.array_split(idx, k_folds)
    accs = []
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        for i in range(k_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            train_trials = [raw_trials[t] for t in train_idx]
            test_trials = [raw_trials[t] for t in test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipe = ModernBCIPipeline(config)
            pipe.train(train_trials, y_train)
            y_pred, _ = pipe.predict(test_trials)
            accs.append(np.mean(y_pred == y_test))
    finally:
        logger.setLevel(prev_level)
    return np.array(accs)
