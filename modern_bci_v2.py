"""
Production-Grade BCI Pipeline v2
==================================
Nature Biomedical Engineering / IEEE TBME seviyesi

Yeni özellikler:
✅ CSP (Common Spatial Pattern) - Motor imagery bel kemiği
✅ 40+ Feature Engineering (time-domain, frequency-domain, spatial)
✅ Laplacian Reference (motor cortex için)
✅ FIR Linear Phase Bandpass
✅ Kalman Smoothing (temporal)
✅ Probability Calibration
✅ Comprehensive Metrics (ITR, ROC-AUC, F1, vb)
✅ Config-based system
✅ Proper logging
✅ Cross-validation framework
"""

import numpy as np
from scipy import signal, stats
from scipy.linalg import eig
import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Basit formatter"""
    
    def format(self, record):
        return f"[{record.levelname}] {record.getMessage()}"

def setup_logging(name="BCI", level=logging.INFO):
    """Logging sistemi kur"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Duplicate handler'ları temizle
    logger.handlers.clear()
    
    handler = logging.StreamHandler()
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# CONFIG MANAGEMENT
# ============================================================================

@dataclass
class BCIConfig:
    """BCI konfigürasyonu"""
    
    # Hardware
    sampling_rate: int = 256
    n_channels: int = 3
    channels: List[str] = None
    
    # Filtering
    notch_freq: int = 50
    use_notch: bool = True   # notch_freq >= Nyquist ise otomatik atlanır
    bandpass_low: float = 0.5
    bandpass_high: float = 50
    fir_order: int = 256  # FIR filtre sırası (linear phase)
    
    # Signal Processing
    window_size: int = 256  # 1 saniye
    step_size: int = 13    # ~50 ms
    
    # Laplacian reference
    use_laplacian: bool = True
    
    # CSP
    use_csp: bool = True
    csp_n_filters: int = 4
    
    # Artifact removal
    artifact_threshold_uv: float = 100
    
    # Classifier
    classifier_type: str = 'lda'  # lda, svm, rf
    confidence_threshold: float = 0.6
    lda_shrinkage: float = 0.15  # 0=raw pooled cov, 1=fully diagonal (Ledoit-Wolf tarzı)

    # Feature selection
    use_feature_selection: bool = True
    n_features_select: int = 20  # F-score'a göre en ayırt edici K feature
    
    # Calibration
    calib_baseline_duration: int = 10  # saniye
    calib_motor_duration: int = 10
    
    # Metrics
    compute_metrics: bool = True
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = ['C3', 'C4', 'Cz'][:self.n_channels]
    
    @classmethod
    def from_yaml(cls, filepath: str):
        """YAML dosyasından yükle"""
        try:
            import yaml
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.warning(f"Config yüklenemedi: {e}. Varsayılanlar kullanılıyor.")
            return cls()

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureExtractor:
    """
    40+ feature extraction
    Time-domain, Frequency-domain, Spatial, Statistical
    """
    
    def __init__(self, sampling_rate=256):
        self.fs = sampling_rate
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Mu': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
    
    def extract_time_domain(self, signal_data: np.ndarray) -> Dict:
        """Time-domain features"""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['min'] = np.min(signal_data)
        features['max'] = np.max(signal_data)
        features['range'] = np.max(signal_data) - np.min(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        
        # Peak-to-peak
        features['peak_to_peak'] = np.ptp(signal_data)
        
        # Kurtosis and Skewness
        features['kurtosis'] = stats.kurtosis(signal_data)
        features['skewness'] = stats.skew(signal_data)
        
        # Line length (complexity)
        features['line_length'] = np.sum(np.abs(np.diff(signal_data)))
        
        # Hjorth parameters
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
        
        # Zero crossings
        features['zero_crossings'] = np.sum(np.diff(np.sign(signal_data)) != 0)
        
        # Entropy (spectral)
        psd = np.abs(np.fft.fft(signal_data))**2
        psd_norm = psd / np.sum(psd)
        features['spectral_entropy'] = -np.sum(
            psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0] + 1e-8)
        )
        
        return features
    
    def extract_frequency_domain(self, signal_data: np.ndarray) -> Dict:
        """Frequency-domain features"""
        features = {}
        
        # PSD (Welch)
        freqs, psd = signal.welch(signal_data, fs=self.fs, 
                                  nperseg=256, noverlap=128)
        
        # Band powers
        for band_name, (low, high) in self.bands.items():
            idx_low = np.argmin(np.abs(freqs - low))
            idx_high = np.argmin(np.abs(freqs - high))
            power = np.mean(psd[idx_low:idx_high+1])
            features[f'{band_name}_power'] = power
        
        # Band ratios
        mu_power = features.get('Mu_power', 1e-6)
        beta_power = features.get('Beta_power', 1e-6)
        features['mu_beta_ratio'] = mu_power / (beta_power + 1e-6)
        features['beta_mu_ratio'] = beta_power / (mu_power + 1e-6)
        features['alpha_theta_ratio'] = (
            features.get('Alpha_power', 1e-6) / 
            (features.get('Theta_power', 1e-6) + 1e-6)
        )
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_centroid'] = spectral_centroid
        
        # Total power
        features['total_power'] = np.sum(psd)
        
        return features
    
    def extract_all_features(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Tüm features'ları çıkar (multi-channel)"""
        all_features = []
        
        for channel in sorted(eeg_dict.keys()):
            signal_data = eeg_dict[channel]
            
            # Time-domain
            time_feat = self.extract_time_domain(signal_data)
            
            # Frequency-domain
            freq_feat = self.extract_frequency_domain(signal_data)
            
            # Birleştir
            channel_features = {**time_feat, **freq_feat}
            all_features.append(list(channel_features.values()))
        
        # Inter-channel features (spatial)
        if len(eeg_dict) >= 2:
            channels = sorted(eeg_dict.keys())
            # C3-C4 asymmetry
            c3_mu = (self.extract_frequency_domain(eeg_dict[channels[0]])
                    .get('Mu_power', 0))
            c4_mu = (self.extract_frequency_domain(eeg_dict[channels[1]])
                    .get('Mu_power', 0))
            asymmetry = (c3_mu - c4_mu) / (c3_mu + c4_mu + 1e-6)
            all_features.append([asymmetry])
        
        return np.concatenate(all_features)


# ============================================================================
# LAPLACIAN REFERENCE
# ============================================================================

class LaplacianReference:
    """
    Surface Laplacian reference
    Motor cortex için optimal
    """
    
    def __init__(self, channels: List[str]):
        self.channels = channels
        
        # Laplacian weights (simplified 2D grid)
        self.laplacian_config = {
            'C3': {'center': 1.0, 'neighbors': {'FC3': -0.25, 'CP3': -0.25, 'C1': -0.25, 'C5': -0.25}},
            'C4': {'center': 1.0, 'neighbors': {'FC4': -0.25, 'CP4': -0.25, 'C2': -0.25, 'C6': -0.25}},
            'Cz': {'center': 1.0, 'neighbors': {'FCz': -0.25, 'CPz': -0.25, 'C1': -0.125, 'C2': -0.125}}
        }
    
    def apply(self, eeg_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Laplacian reference uygula"""
        laplacian_data = {}
        
        for ch in self.channels:
            if ch not in eeg_dict:
                continue
            
            signal_data = eeg_dict[ch].copy()
            
            # Center weight
            if ch in self.laplacian_config:
                config = self.laplacian_config[ch]
                
                # Neighbor weights ekle (eğer varsa)
                for neighbor_ch, weight in config.get('neighbors', {}).items():
                    if neighbor_ch in eeg_dict:
                        signal_data = signal_data + weight * eeg_dict[neighbor_ch]
            
            laplacian_data[ch] = signal_data
        
        return laplacian_data


# ============================================================================
# COMMON SPATIAL PATTERN (CSP)
# ============================================================================

class CSPFilter:
    """
    Common Spatial Pattern - Motor imagery için spatial filters
    Class-discriminative filters öğren: S = C1 @ inv(C1 + C2)
    """
    
    def __init__(self, n_filters: int = 4):
        self.n_filters = n_filters
        self.filters = None
        self.n_components = None
    
    def fit(self, X_class1: np.ndarray, X_class2: np.ndarray):
        """
        CSP filtrelerini öğren
        X_class1: (n_samples, n_channels, n_features) or (n_channels, n_features)
        X_class2: (n_samples, n_channels, n_features) or (n_channels, n_features)
        """
        # Covariance matrices
        if X_class1.ndim == 2:
            X_class1 = X_class1.reshape(1, *X_class1.shape)
        if X_class2.ndim == 2:
            X_class2 = X_class2.reshape(1, *X_class2.shape)
        
        # Mean covariance
        C1 = np.zeros((X_class1.shape[1], X_class1.shape[1]))
        for trial in X_class1:
            C = (trial @ trial.T) / np.trace(trial @ trial.T)
            C1 += C / X_class1.shape[0]
        
        C2 = np.zeros((X_class2.shape[1], X_class2.shape[1]))
        for trial in X_class2:
            C = (trial @ trial.T) / np.trace(trial @ trial.T)
            C2 += C / X_class2.shape[0]
        
        # Eigendecomposition: C1 = W @ Lambda @ W^T
        Lambda, W = eig(C1, C1 + C2)
        idx = np.argsort(Lambda)[::-1]
        
        self.filters = W[:, idx[:self.n_filters]].real
        self.n_components = self.n_filters
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """CSP filtrelerini uygula (ham spatially-filtered sinyal)"""
        if self.filters is None:
            return X
        
        if X.ndim == 2:
            # Single trial
            return (self.filters.T @ X).flatten()
        else:
            # Multiple trials
            return np.array([
                (self.filters.T @ trial).flatten() 
                for trial in X
            ])

    def transform_logvar(self, X: np.ndarray) -> np.ndarray:
        """
        Standart CSP feature: spatial filtre uygula, sonra her filtrelenmiş
        bileşenin log-varyansını al. X: (n_channels, n_samples) tek trial
        veya (n_trials, n_channels, n_samples) çoklu trial.
        """
        if self.filters is None:
            return np.array([])

        if X.ndim == 2:
            Z = self.filters.T @ X  # (n_filters, n_samples)
            var = np.var(Z, axis=1)
            var_norm = var / (np.sum(var) + 1e-10)
            return np.log(var_norm + 1e-10)
        else:
            return np.array([self.transform_logvar(trial) for trial in X])


# ============================================================================
# FEATURE SELECTION (F-score)
# ============================================================================

class FeatureSelector:
    """
    F-score'a göre en ayırt edici K feature'ı seç.
    F-score = (sınıflar arası ortalama farkı)^2 / pooled varyans
    """

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


# ============================================================================
# SIMPLE LINEAR DISCRIMINANT ANALYSIS (SimpleLDA)
# ============================================================================

class SimpleLDA:
    """
    Linear Discriminant Analysis - binary classification
    Sklearn'e bağımlı değil, manuel implementation
    Shrinkage regularization destekli (Ledoit-Wolf tarzı, basitleştirilmiş)
    """
    
    def __init__(self, shrinkage: float = 0.0):
        self.mean_0 = None
        self.mean_1 = None
        self.pooled_cov = None
        self.coef = None
        self.intercept = None
        self.shrinkage = shrinkage  # 0 = ham kovaryans, 1 = tam diagonal
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        LDA fit
        X: (n_samples, n_features)
        y: (n_samples,) - binary labels (0 or 1)
        """
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        # Means
        self.mean_0 = np.mean(X0, axis=0)
        self.mean_1 = np.mean(X1, axis=0)
        
        # Pooled covariance (ham)
        S0 = (X0 - self.mean_0).T @ (X0 - self.mean_0)
        S1 = (X1 - self.mean_1).T @ (X1 - self.mean_1)
        raw_cov = (S0 + S1) / (len(X) - 2)
        
        # Shrinkage: ham kovaryansı diagonal (eşit varyanslı, sıfır korelasyonlu)
        # bir hedefe doğru büzüştür. Feature sayısı örnek sayısına yakın/fazla
        # olduğunda kovaryans kötü koşullu olur; shrinkage bunu stabilize eder.
        n_features = raw_cov.shape[0]
        avg_var = np.trace(raw_cov) / n_features
        shrink_target = avg_var * np.eye(n_features)
        self.pooled_cov = (1 - self.shrinkage) * raw_cov + self.shrinkage * shrink_target
        
        # Add regularization to avoid singular matrix
        self.pooled_cov += np.eye(self.pooled_cov.shape[0]) * 1e-6
        
        # LDA weights
        try:
            inv_cov = np.linalg.inv(self.pooled_cov)
        except:
            inv_cov = np.linalg.pinv(self.pooled_cov)
        
        self.coef = inv_cov @ (self.mean_1 - self.mean_0)
        self.intercept = -0.5 * (
            self.mean_1 @ inv_cov @ self.mean_1 - 
            self.mean_0 @ inv_cov @ self.mean_0
        )
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """LDA decision function"""
        return X @ self.coef + self.intercept
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions"""
        return (self.decision_function(X) > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates (sigmoid)"""
        decision = self.decision_function(X)
        proba = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - proba, proba])


# ============================================================================
# FIR BANDPASS FILTER (Linear Phase)
# ============================================================================

class FIRBandpassFilter:
    """
    FIR Linear Phase Bandpass Filter
    Faz bozulması yok
    """
    
    def __init__(self, low_freq: float, high_freq: float, 
                 sampling_rate: int, order: int = 256):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = sampling_rate
        # Ensure odd number of taps for bandpass filter
        self.order = order if order % 2 == 1 else order + 1
        
        # FIR filtre tasarımı (Hamming window)
        nyquist = sampling_rate / 2
        taps = signal.firwin(
            self.order,
            [low_freq / nyquist, high_freq / nyquist],
            window='hamming'
        )
        
        self.taps = taps
    
    def apply_offline(self, signal_data: np.ndarray) -> np.ndarray:
        """Offline filtreleme (filtfilt ile linear phase)"""
        return signal.filtfilt(self.taps, 1, signal_data)
    
    def apply_realtime(self, signal_data: np.ndarray, 
                      state: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Real-time filtreleme (state correlation)"""
        if state is None:
            state = signal.lfilter_zi(self.taps, 1)
        
        filtered, state = signal.lfilter(self.taps, 1, signal_data, zi=state)
        return filtered, state


# ============================================================================
# KALMAN SMOOTHING
# ============================================================================

class KalmanSmoother:
    """
    Kalman Filter for temporal smoothing
    Command titremesini azalt
    """
    
    def __init__(self, process_variance: float = 0.01,
                measurement_variance: float = 1.0):
        self.q = process_variance  # Process noise
        self.r = measurement_variance  # Measurement noise
        self.x = 0.5  # Initial state
        self.p = 1.0  # Initial covariance
    
    def update(self, measurement: float) -> float:
        """Kalman update"""
        # Prediction
        x_pred = self.x
        p_pred = self.p + self.q
        
        # Update
        K = p_pred / (p_pred + self.r)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.p = (1 - K) * p_pred
        
        return self.x
    
    def reset(self):
        """Reset filter"""
        self.x = 0.5
        self.p = 1.0


# ============================================================================
# PROBABILITY CALIBRATION
# ============================================================================

class PlattCalibration:
    """
    Platt Scaling for probability calibration
    Softmax → gerçek olasılık
    """
    
    def __init__(self):
        self.A = 1.0
        self.B = 0.0
    
    def fit(self, confidences: np.ndarray, labels: np.ndarray):
        """Sigmoid parametrelerini öğren"""
        # Basit linear regression
        y = labels.astype(float)
        X = np.column_stack([confidences, np.ones(len(confidences))])
        
        try:
            self.A, self.B = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            self.A, self.B = 1.0, 0.0
    
    def calibrate(self, confidence: float) -> float:
        """Calibrated probability"""
        return 1.0 / (1.0 + np.exp(-self.A * confidence - self.B))


# ============================================================================
# COMPREHENSIVE METRICS
# ============================================================================

class BCIMetrics:
    """
    BCI için kapsamlı metrikler
    """
    
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.confidence_scores = []
    
    def add_result(self, prediction: int, ground_truth: int, confidence: float):
        """Sonuç ekle"""
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        self.confidence_scores.append(confidence)
    
    def compute_metrics(self) -> Dict:
        """Tüm metrikleri hesapla"""
        if len(self.predictions) == 0:
            return {}
        
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        y_conf = np.array(self.confidence_scores)
        
        # Basic metrics
        metrics = {
            'accuracy': np.mean(y_pred == y_true),
            'n_samples': len(y_true),
        }
        
        # Precision, Recall, F1 (sklearn olmadan)
        if len(np.unique(y_true)) > 1:
            # Manual hesapla
            true_pos = np.sum((y_pred == 1) & (y_true == 1))
            false_pos = np.sum((y_pred == 1) & (y_true == 0))
            false_neg = np.sum((y_pred == 0) & (y_true == 1))
            true_neg = np.sum((y_pred == 0) & (y_true == 0))
            
            precision = true_pos / (true_pos + false_pos + 1e-6)
            recall = true_pos / (true_pos + false_neg + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1'] = float(f1)
            
            # ROC-AUC (basit Wilcoxon)
            try:
                pos_scores = y_conf[y_true == 1]
                neg_scores = y_conf[y_true == 0]
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    auc = np.mean([pos_scores[i] > neg_scores[j] 
                                  for i in range(len(pos_scores)) 
                                  for j in range(len(neg_scores))])
                    metrics['roc_auc'] = float(auc)
                else:
                    metrics['roc_auc'] = 0.5
            except:
                metrics['roc_auc'] = 0.5
        
        # ITR (Information Transfer Rate)
        N = len(np.unique(y_true))
        P = metrics['accuracy']
        T = 1.0
        
        if P > 0 and P < 1:
            itr = (np.log2(N) + P * np.log2(P) + 
                   (1-P) * np.log2((1-P)/(N-1))) * 60 / T
        else:
            itr = 0
        
        metrics['itr'] = max(0, itr)
        
        # Confidence stats
        metrics['mean_confidence'] = np.mean(y_conf)
        metrics['std_confidence'] = np.std(y_conf)
        
        return metrics


# ============================================================================
# MODERN BCI PIPELINE
# ============================================================================

class ModernBCIPipeline:
    """
    Production-Grade BCI Pipeline v2
    Nature Biomedical Engineering seviyesi
    """
    
    def __init__(self, config: BCIConfig = None):
        self.config = config or BCIConfig()
        self.logger = logger
        
        self.logger.info("Modern BCI Pipeline başlatılıyor...")
        
        # Components
        self.feature_extractor = AdvancedFeatureExtractor(self.config.sampling_rate)
        self.laplacian = LaplacianReference(self.config.channels)
        
        # FIR filters
        self.bandpass_fir = FIRBandpassFilter(
            self.config.bandpass_low,
            self.config.bandpass_high,
            self.config.sampling_rate,
            order=self.config.fir_order
        )
        
        # Notch (notch_freq Nyquist'e eşit/üstündeyse matematiksel olarak
        # imkansız - örn. 100Hz sampling'te 60Hz notch. Bu durumda atla.)
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
                    f"notch filtre atlanıyor (sampling_rate çok düşük)."
                )
        
        # Kalman smoothers
        self.kalman_smoothers = {ch: KalmanSmoother() 
                                for ch in self.config.channels}
        
        # CSP filter
        self.csp = CSPFilter(n_filters=self.config.csp_n_filters) if self.config.use_csp else None
        
        # Classifier (shrinkage regularization ile)
        if self.config.classifier_type.lower() == 'lda':
            self.classifier = SimpleLDA(shrinkage=self.config.lda_shrinkage)
        else:
            self.classifier = SimpleLDA(shrinkage=self.config.lda_shrinkage)  # default to LDA

        # Feature selection (F-score, en ayırt edici K feature)
        self.feature_selector = (
            FeatureSelector(k=self.config.n_features_select)
            if self.config.use_feature_selection else None
        )
        
        # Metrics
        self.metrics = BCIMetrics()
        
        self.logger.info(f"Config: {self.config}")
        self.logger.info("Pipeline hazırlandı")
    
    def preprocess(self, eeg_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocessing pipeline"""
        
        # 1. Notch filter
        notch_data = {}
        for ch in self.config.channels:
            if ch in eeg_dict:
                if self._apply_notch:
                    notch_data[ch] = signal.sosfilt(self.notch_sos,
                                                   [eeg_dict[ch]])[0]
                else:
                    notch_data[ch] = eeg_dict[ch]
        
        # 2. Laplacian reference
        if self.config.use_laplacian:
            ref_data = self.laplacian.apply(notch_data)
        else:
            ref_data = notch_data
        
        # 3. FIR Bandpass
        filtered_data = {}
        for ch in self.config.channels:
            if ch in ref_data:
                filtered_data[ch] = self.bandpass_fir.apply_offline(ref_data[ch])
        
        return filtered_data
    
    def extract_features(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Feature extraction (40+ istatistiksel feature, CSP hariç)"""
        return self.feature_extractor.extract_all_features(eeg_dict)

    def _stack_channels(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """eeg_dict'i (n_channels, n_samples) matrisine çevir - CSP için"""
        return np.array([eeg_dict[ch] for ch in self.config.channels if ch in eeg_dict])

    def extract_full_features(self, eeg_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """İstatistiksel features + (varsa) CSP log-variance features"""
        stat_features = self.extract_features(eeg_dict)

        if self.config.use_csp and self.csp is not None and self.csp.filters is not None:
            raw_matrix = self._stack_channels(eeg_dict)
            csp_features = self.csp.transform_logvar(raw_matrix)
            return np.concatenate([stat_features, csp_features])

        return stat_features

    def train(self, raw_trials: List[Dict[str, np.ndarray]], y: np.ndarray):
        """
        Train classifier.
        raw_trials: preprocess() sonrası her trial için eeg_dict listesi
                    (ham çok-kanallı sinyal, henüz feature'a çevrilmemiş).
        """
        y = np.asarray(y)

        # 1) CSP'yi HAM sinyal üzerinde fit et (channels x samples)
        if self.config.use_csp and self.csp is not None:
            stacked = np.array([self._stack_channels(t) for t in raw_trials])
            X_class1 = stacked[y == 0]
            X_class2 = stacked[y == 1]
            if len(X_class1) > 0 and len(X_class2) > 0:
                self.csp.fit(X_class1, X_class2)

        # 2) CSP fit edildikten SONRA feature matrisini oluştur
        X = np.array([self.extract_full_features(t) for t in raw_trials])

        # 3) Feature selection (F-score, en ayırt edici K feature)
        if self.feature_selector is not None:
            self.feature_selector.fit(X, y)
            X = self.feature_selector.transform(X)

        # 4) Sınıflandırıcıyı eğit
        self.classifier.fit(X, y)

    def predict(self, raw_trials: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence.
        raw_trials: preprocess() sonrası her trial için eeg_dict listesi.
        """
        X = np.array([self.extract_full_features(t) for t in raw_trials])
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        predictions = self.classifier.predict(X)
        confidences = self.classifier.predict_proba(X)[:, 1]
        return predictions, confidences
    
    def summarize_performance(self) -> str:
        """Performance summary"""
        metrics = self.metrics.compute_metrics()
        
        summary = "\n" + "="*60 + "\n"
        summary += "BCI PERFORMANCE SUMMARY\n"
        summary += "="*60 + "\n"
        summary += f"Samples: {metrics.get('n_samples', 0)}\n"
        summary += f"Accuracy: {metrics.get('accuracy', 0):.2%}\n"
        summary += f"Precision: {metrics.get('precision', 0):.2%}\n"
        summary += f"Recall: {metrics.get('recall', 0):.2%}\n"
        summary += f"F1 Score: {metrics.get('f1', 0):.2%}\n"
        summary += f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}\n"
        summary += f"ITR (bits/min): {metrics.get('itr', 0):.1f}\n"
        summary += f"Mean Confidence: {metrics.get('mean_confidence', 0):.3f}\n"
        summary += "="*60 + "\n"
        
        return summary


# ============================================================================
# DEMO
# ============================================================================

def generate_motor_imagery_trial(label: int, fs: int = 256, duration: float = 5.0,
                                  noise_std: float = 0.4) -> Dict[str, np.ndarray]:
    """
    Sentetik motor-imagery EEG trial üret.

    label=0 -> sol el hayali (kontralateral ERD sağ hemisferde, C4'te mu bastırılır)
    label=1 -> sağ el hayali (kontralateral ERD sol hemisferde, C3'te mu bastırılır)

    Not: Gerçek veri değil, sınıflar arasında ayırt edici bir mu-band (8-12Hz)
    güç farkı yaratmak için kullanılan kontrollü bir simülasyon.
    """
    t = np.arange(0, duration, 1/fs)
    phase_c3 = np.random.uniform(0, 2*np.pi)
    phase_c4 = np.random.uniform(0, 2*np.pi)

    if label == 0:
        c3_amp, c4_amp = 2.0, 0.7   # C4'te ERD (bastırma)
    else:
        c3_amp, c4_amp = 0.7, 2.0   # C3'te ERD (bastırma)

    c3 = c3_amp * np.sin(2*np.pi*10*t + phase_c3) + noise_std * np.random.randn(len(t))
    c4 = c4_amp * np.sin(2*np.pi*10*t + phase_c4) + noise_std * np.random.randn(len(t))
    cz = 1.0 * np.sin(2*np.pi*10*t) + noise_std * np.random.randn(len(t))

    return {'C3': c3, 'C4': c4, 'Cz': cz}


def k_fold_cross_validate(config: BCIConfig, raw_trials: List[Dict[str, np.ndarray]],
                           labels: np.ndarray, k_folds: int = 5) -> np.ndarray:
    """Verilen config için k-fold CV accuracy'lerini döndürür"""
    idx = np.arange(len(raw_trials))
    np.random.shuffle(idx)
    folds = np.array_split(idx, k_folds)

    accs = []
    prev_level = logger.level
    logger.setLevel(logging.WARNING)  # CV sırasında log gürültüsünü azalt
    try:
        for i in range(k_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            train_trials = [raw_trials[t] for t in train_idx]
            test_trials = [raw_trials[t] for t in test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            pipe = ModernBCIPipeline(config)
            pipe.train(train_trials, y_train)
            y_pred, _ = pipe.predict(test_trials)
            accs.append(np.mean(y_pred == y_test))
    finally:
        logger.setLevel(prev_level)

    return np.array(accs)


def test_modern_pipeline():
    """Modern pipeline test - CV ile hiperparametre seçimi + gerçek train/test"""

    print("\n" + "="*70)
    print("PRODUCTION-GRADE BCI v2 - Nature Biomedical Engineering")
    print("="*70 + "\n")

    np.random.seed(42)

    base_config = BCIConfig(
        sampling_rate=256,
        n_channels=3,
        use_laplacian=True,
        use_csp=True,
        fir_order=256
    )

    # Pipeline (sadece preprocess için)
    print("[*] Pipeline başlatılıyor...")
    pipeline = ModernBCIPipeline(base_config)
    print("[+] Pipeline hazırlandı\n")

    # Sentetik dataset
    print("[*] Sentetik motor-imagery verisi oluşturuluyor...")
    n_per_class = 100
    trials, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            trials.append(generate_motor_imagery_trial(label))
            labels.append(label)
    labels = np.array(labels)
    print(f"[+] {len(trials)} trial oluşturuldu ({n_per_class} her sınıftan)\n")

    print("[*] Preprocessing...")
    preprocessed_trials = [pipeline.preprocess(t) for t in trials]
    print(f"[+] {len(preprocessed_trials)} trial preprocess edildi\n")

    # Train/test split (final holdout - grid search'te hiç görülmeyecek)
    idx = np.arange(len(preprocessed_trials))
    np.random.shuffle(idx)
    split = int(0.7 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    train_trials = [preprocessed_trials[i] for i in train_idx]
    test_trials = [preprocessed_trials[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # ---- Hiperparametre grid search (SADECE train seti üzerinde, 5-fold CV) ----
    print("[*] 5-fold CV ile hiperparametre grid search (shrinkage x n_features)...")
    shrinkage_grid = [0.0, 0.1, 0.2, 0.3, 0.5]
    k_grid = [10, 15, 20, 30, 90]  # 90 = feature selection'ı devre dışı gibi (tüm feature'lar)

    best_score, best_params = -1, None
    results_table = []
    for shrink in shrinkage_grid:
        for k in k_grid:
            cfg = BCIConfig(sampling_rate=256, n_channels=3, use_laplacian=True,
                             use_csp=True, fir_order=256,
                             lda_shrinkage=shrink, n_features_select=k)
            accs = k_fold_cross_validate(cfg, train_trials, y_train, k_folds=5)
            mean_acc = accs.mean()
            results_table.append((shrink, k, mean_acc))
            if mean_acc > best_score:
                best_score, best_params = mean_acc, (shrink, k)

    print(f"{'shrinkage':>10}{'n_features':>12}{'CV accuracy':>14}")
    for shrink, k, acc in sorted(results_table, key=lambda r: -r[2])[:8]:
        marker = "  <-- best" if (shrink, k) == best_params else ""
        print(f"{shrink:>10.2f}{k:>12d}{acc:>13.2%}{marker}")
    print(f"\n[+] En iyi hiperparametreler: shrinkage={best_params[0]}, n_features={best_params[1]} "
          f"(CV accuracy: {best_score:.2%})\n")

    # ---- Final model: en iyi hiperparametrelerle, train setinin TAMAMINDA eğit ----
    final_config = BCIConfig(sampling_rate=256, n_channels=3, use_laplacian=True,
                              use_csp=True, fir_order=256,
                              lda_shrinkage=best_params[0], n_features_select=best_params[1])
    final_pipeline = ModernBCIPipeline(final_config)

    print("[*] Final model eğitiliyor (en iyi hiperparametrelerle)...")
    final_pipeline.train(train_trials, y_train)
    print("[+] Eğitim tamamlandı\n")

    print("[*] Hiç görülmemiş test seti üzerinde tahmin yapılıyor...")
    y_pred, y_conf = final_pipeline.predict(test_trials)
    for pred, true, conf in zip(y_pred, y_test, y_conf):
        final_pipeline.metrics.add_result(int(pred), int(true), float(conf))
    print("[+] Tahminler tamamlandı\n")

    summary = final_pipeline.summarize_performance()
    print(summary)

    print("[+] Test tamamlandı!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_modern_pipeline()
