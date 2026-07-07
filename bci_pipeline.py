"""
bci_pipeline_v3.py
==================
EEG Walk/Stop BCI - v3

Brief'teki kritik tasarım hatası düzeltmesi:
  Training verisi (etiketli event pencereleri) != Prediction verisi (tüm
  sürekli kayıt üzerinde sliding window) -> dağılım uyuşmazlığı -> collapse.

Bu dosya üç modu ayırır:
  train             : etiketli event pencerelerinden model eğit, kaydet
  validate_timeline : eğitilmiş modeli TAM etiketli oturumda sliding window
                      ile test et, ground-truth timeline ile karşılaştır
  predict           : etiketsiz ham EEG üzerinde inference (event gerekmez)

Sınıflandırıcı DEĞİŞMEDİ: CSP + feature selection + shrinkage LDA
(modern_bci_v2.py'den aynen reuse ediliyor). Yeni bir ML sınıflandırıcısı
eklenmedi - sadece normalizasyon + IDLE gate + model persistence + doğru
confidence tanımı eklendi.

Bilimsel dürüstlük notu: Bu sistem, önceden eğitilmiş bir modelle EEG'den
Walk/Stop komut periyotlarını TAHMİN eder. Saf beyin-içi yürüme niyetinin
kanıtlanmış decode'u DEĞİLDİR. Performans; eğitim verisine, artefaktlara,
denek/oturum benzerliğine ve deployment koşullarına bağlıdır. Tek bir
denek/oturumda eğitilen model başka bir denek/oturuma iyi genellemeyebilir.
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
COMMAND_LABEL_MAP = {'x5': 0, 'x8': 1}  # x5=STOP=0, x8=WALK=1. x99/diğerleri -> ignore


# ============================================================================
# FEATURE NAMES (deterministic sıra - kaydetmek için)
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


# ============================================================================
# SLIDING WINDOWS (deployment-style: tam kayıt üzerinde)
# ============================================================================

def build_sliding_windows(signal_len, fs, window_len=5.0, step_len=1.0):
    """
    Tüm kayıt üzerinde overlapping sliding window'lar üretir (deployment-style).
    Event-anchored, etiketli eğitim pencereleriyle KARIŞTIRILMAMALI.
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


def label_windows_from_events(windows, events, overlap_threshold=0.5):
    """
    Ground-truth timeline etiketleme (SADECE validate_timeline kullanır).
    events: (onset, duration, trial_type) listesi.
    Dönen label: 0=STOP, 1=WALK, 2=IDLE (belirsiz/örtüşme yok)
    """
    x5_intervals = [(o, o + d) for o, d, t in events if t == 'x5']
    x8_intervals = [(o, o + d) for o, d, t in events if t == 'x8']

    def overlap_duration(win_start, win_end, intervals):
        total = 0.0
        for s, e in intervals:
            total += max(0.0, min(win_end, e) - max(win_start, s))
        return total

    labels = []
    for w in windows:
        dur = w['end_time'] - w['start_time']
        ov5 = overlap_duration(w['start_time'], w['end_time'], x5_intervals)
        ov8 = overlap_duration(w['start_time'], w['end_time'], x8_intervals)
        frac5, frac8 = ov5 / dur, ov8 / dur
        if frac8 >= overlap_threshold and frac8 >= frac5:
            labels.append(1)
        elif frac5 >= overlap_threshold and frac5 > frac8:
            labels.append(0)
        else:
            labels.append(2)  # IDLE (belirsiz/event dışı)
    return np.array(labels)


def extract_command_training_windows(events, fs, skip_start=1.0, skip_end=1.0, window_len=5.0):
    """
    Eğitim için event-ANKORLU, non-overlapping pencereler (x5/x8 SADECE).
    Bu, deployment sliding window'undan FARKLI ve BİLE İSTEYEREK öyle -
    eğitim verisi hep etiketli command aralığından gelir.
    """
    windows = []
    for onset, duration, trial_type in events:
        if trial_type not in COMMAND_LABEL_MAP:
            continue
        label = COMMAND_LABEL_MAP[trial_type]
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
# DEPLOYABLE MODEL: ModernBCIPipeline sarmalayıcısı + normalizasyon + IDLE gate
# ============================================================================

class DeployableBCIModel:
    """
    CSP + feature selection + shrinkage LDA (modern_bci_v2.ModernBCIPipeline)
    + feature normalizasyonu + IDLE gate + model persistence.

    Yeni bir ML sınıflandırıcı YOK - sadece deployment için gerekli sarmalayıcı.
    """

    def __init__(self, channels, sampling_rate, bandpass_low=0.5, bandpass_high=45,
                 fir_order=200, use_notch=False, notch_freq=60, use_laplacian=True,
                 use_csp=True, lda_shrinkage=0.0, n_features_select=5,
                 window_len=5.0, step_len=1.0,
                 confidence_threshold=0.6, idle_distance_threshold=3.5):
        self.channels = list(channels)
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

        # Fit sonrası doldurulacak:
        self.feature_mean = None
        self.feature_std = None
        self.command_feature_mean = None
        self.command_feature_std = None
        self.feature_names = None
        self.n_csp_filters = self.config.csp_n_filters if use_csp else 0

    def preprocess_continuous(self, raw_signals):
        """Tam sürekli kaydı BİR KEZ preprocess et (per-window değil)."""
        return self.pipeline.preprocess({ch: raw_signals[ch] for ch in self.channels})

    def _epoch_dict(self, preprocessed, start_idx, end_idx):
        return {ch: preprocessed[ch][start_idx:end_idx] for ch in self.channels}

    def train(self, preprocessed_continuous, train_windows):
        """
        preprocessed_continuous: preprocess_continuous() çıktısı
        train_windows: extract_command_training_windows() çıktısı (label ile)
        """
        raw_trials = [self._epoch_dict(preprocessed_continuous, w['start_idx'], w['end_idx'])
                      for w in train_windows]
        y = np.array([w['label'] for w in train_windows])

        # 1) CSP fit + feature selection fit + (throwaway) ilk classifier fit
        #    - modern_bci_v2.ModernBCIPipeline.train() bunları tek çağrıda yapar
        self.pipeline.train(raw_trials, y)

        # 2) Şimdi CSP + feature_selector fit edildi -> seçili feature'ları
        #    (ham ölçekte) tüm eğitim pencereleri için çıkar
        X_raw = np.array([self.pipeline.extract_full_features(t) for t in raw_trials])
        X_sel_raw = self.pipeline.feature_selector.transform(X_raw)

        # 3) Normalizasyon istatistikleri (seçili feature uzayında)
        self.feature_mean = X_sel_raw.mean(axis=0)
        self.feature_std = X_sel_raw.std(axis=0) + EPS
        X_sel_norm = (X_sel_raw - self.feature_mean) / self.feature_std

        # 4) Classifier'ı NORMALİZE edilmiş feature'larla yeniden eğit
        #    (CSP/feature selection aynı; sadece LDA'nın girdisi normalize)
        self.pipeline.classifier.fit(X_sel_norm, y)

        # 5) IDLE gate referans dağılımı (command-window feature dağılımı)
        self.command_feature_mean = X_sel_norm.mean(axis=0)
        self.command_feature_std = X_sel_norm.std(axis=0) + EPS

        self.feature_names = get_feature_names(self.channels, self.n_csp_filters)

        return {
            'n_train_windows': len(train_windows),
            'n_stop': int(np.sum(y == 0)), 'n_walk': int(np.sum(y == 1)),
            'selected_feature_idx': self.pipeline.feature_selector.selected_idx.tolist(),
        }

    def predict_window(self, preprocessed_continuous, start_idx, end_idx):
        """
        Tek pencere için: IDLE gate -> (geçerse) LDA -> düzeltilmiş confidence.
        Dönen: (label:int[0/1/2], confidence:float veya None, z_distance:float, raw_pred_or_None)
        """
        epoch = self._epoch_dict(preprocessed_continuous, start_idx, end_idx)
        x_raw = self.pipeline.extract_full_features(epoch).reshape(1, -1)
        x_sel_raw = self.pipeline.feature_selector.transform(x_raw)[0]
        x_sel_norm = (x_sel_raw - self.feature_mean) / self.feature_std

        z_distance = float(np.mean(np.abs(
            (x_sel_norm - self.command_feature_mean) / self.command_feature_std
        )))

        if z_distance > self.idle_distance_threshold:
            return 2, None, z_distance, None  # IDLE gate tetiklendi

        proba = self.pipeline.classifier.predict_proba(x_sel_norm.reshape(1, -1))[0]
        pred = int(self.pipeline.classifier.predict(x_sel_norm.reshape(1, -1))[0])
        # DÜZELTME: confidence = P(tahmin edilen sınıf), her zaman P(WALK) değil
        confidence = float(proba[pred])

        if confidence < self.confidence_threshold:
            return 2, confidence, z_distance, pred  # düşük güven -> IDLE

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
            'command_label_map_source': COMMAND_LABEL_MAP,
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
        )

        # Fit edilmiş durumu geri yükle (CSP, feature_selector, classifier)
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
# MODE: train
# ============================================================================

def run_train(edf_path, events_path, output_dir, channels=('C3', 'C4', 'Cz'),
              window_len=5.0, step_len=1.0, idle_distance_threshold=3.5,
              confidence_threshold=0.6, n_features_select=5, lda_shrinkage=0.0):
    print("=" * 70)
    print("MODE: train")
    print("=" * 70)

    signals, info = read_edf(edf_path)
    events_all = parse_events(events_path)
    events = [(o, d, t) for o, d, t in events_all if t in COMMAND_LABEL_MAP]
    dropped = [(o, d, t) for o, d, t in events_all if t not in COMMAND_LABEL_MAP]
    if dropped:
        print(f"[*] {len(dropped)} event x5/x8 dışında, atlandı (örn: {dropped[:3]})")

    fs = info['sampling_rate'][channels[0]]
    print(f"[*] EDF: {len(signals[channels[0]])} örnek @ {fs:.0f}Hz")
    print(f"[*] Events: {len(events)} kullanılabilir (x5/x8)")

    model = DeployableBCIModel(
        channels=channels, sampling_rate=fs, window_len=window_len, step_len=step_len,
        idle_distance_threshold=idle_distance_threshold,
        confidence_threshold=confidence_threshold,
        n_features_select=n_features_select, lda_shrinkage=lda_shrinkage,
    )

    preprocessed = model.preprocess_continuous(signals)
    train_windows = extract_command_training_windows(events, fs, window_len=window_len)
    print(f"[*] {len(train_windows)} eğitim penceresi çıkarıldı "
          f"(event-ankorlu, non-overlapping, {window_len}s)")

    stats = model.train(preprocessed, train_windows)
    print(f"[+] Eğitim tamamlandı: STOP={stats['n_stop']}, WALK={stats['n_walk']}")
    print(f"[+] Seçili feature indeksleri: {stats['selected_feature_idx']}")

    info_saved = model.save(output_dir)
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("BCI Walk/Stop Model - Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"EDF: {edf_path}\n")
        f.write(f"Events: {events_path}\n")
        f.write(f"Channels: {channels}\n")
        f.write(f"Sampling rate: {fs}\n")
        f.write(f"Train windows: {len(train_windows)} (STOP={stats['n_stop']}, WALK={stats['n_walk']})\n")
        f.write(f"CSP: {info_saved['use_csp']}, n_features_select: {info_saved['n_features_select']}, "
                f"lda_shrinkage: {info_saved['lda_shrinkage']}\n")
        f.write(f"idle_distance_threshold: {idle_distance_threshold}, "
                f"confidence_threshold: {confidence_threshold}\n\n")
        f.write("BILIMSEL DURUSTLUK: Bu model onceden egitilmis bir modelle EEG'den\n")
        f.write("Walk/Stop komut periyotlarini tahmin eder. Saf beyin-ici yurume\n")
        f.write("niyetinin kanitlanmis decode'u degildir. Performans egitim verisine,\n")
        f.write("artefaktlara, denek/oturum benzerligine baglidir.\n")

    print(f"[+] Model kaydedildi: {output_dir}/")
    return model, signals, info, events


# ============================================================================
# MODE: validate_timeline
# ============================================================================

def run_validate_timeline(edf_path, events_path, model_dir, output_dir,
                           overlap_threshold=0.5):
    print("=" * 70)
    print("MODE: validate_timeline")
    print("=" * 70)

    model = DeployableBCIModel.load(model_dir)
    signals, info = read_edf(edf_path)
    events_all = parse_events(events_path)

    fs = model.sampling_rate
    preprocessed = model.preprocess_continuous(signals)
    signal_len = len(preprocessed[model.channels[0]])

    windows = build_sliding_windows(signal_len, fs, model.window_len, model.step_len)
    print(f"[*] {len(windows)} sliding window oluşturuldu "
          f"(window={model.window_len}s, step={model.step_len}s, tam kayıt üzerinde)")

    gt_labels = label_windows_from_events(windows, events_all, overlap_threshold)

    rows = []
    for w, gt in zip(windows, gt_labels):
        label, conf, z_dist, raw_pred = model.predict_window(preprocessed, w['start_idx'], w['end_idx'])
        rows.append({
            'start_time': w['start_time'], 'end_time': w['end_time'],
            'ground_truth': int(gt), 'predicted': label,
            'confidence': conf, 'z_distance': z_dist,
        })

    os.makedirs(output_dir, exist_ok=True)

    # --- timeline CSV ---
    import csv
    with open(os.path.join(output_dir, 'validated_timeline.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_time', 'end_time', 'ground_truth_label', 'predicted_label',
                          'confidence', 'z_distance'])
        for r in rows:
            writer.writerow([f"{r['start_time']:.2f}", f"{r['end_time']:.2f}",
                              LABEL_NAMES[r['ground_truth']], LABEL_NAMES[r['predicted']],
                              '' if r['confidence'] is None else f"{r['confidence']:.4f}",
                              f"{r['z_distance']:.3f}"])

    y_true = np.array([r['ground_truth'] for r in rows])
    y_pred = np.array([r['predicted'] for r in rows])

    classes = [0, 1, 2]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    with open(os.path.join(output_dir, 'timeline_confusion_matrix.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + [f"pred_{LABEL_NAMES[c]}" for c in classes])
        for i, c in enumerate(classes):
            writer.writerow([f"true_{LABEL_NAMES[c]}"] + cm[i].tolist())

    # per-class precision/recall/f1
    per_class = {}
    for c in classes:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall) else float('nan')
        )
        per_class[LABEL_NAMES[c]] = {'precision': precision, 'recall': recall, 'f1': f1,
                                      'support': int(cm[c, :].sum())}

    pred_counts = {LABEL_NAMES[c]: int(np.sum(y_pred == c)) for c in classes}
    total = len(y_pred)
    walk_recall = per_class['WALK']['recall']
    stop_recall = per_class['STOP']['recall']
    idle_fp_rate = (cm[:, 2].sum() - cm[2, 2]) / max(total - cm[2, :].sum(), 1)

    dominant_frac = max(pred_counts.values()) / total
    collapse = dominant_frac > 0.90 or pred_counts['WALK'] == 0

    metrics = {
        'total_windows': total,
        'prediction_counts': pred_counts,
        'per_class': per_class,
        'walk_recall': walk_recall,
        'stop_recall': stop_recall,
        'idle_false_positive_rate': idle_fp_rate,
        'collapse_warning': bool(collapse),
        'dominant_prediction_fraction': dominant_frac,
    }
    with open(os.path.join(output_dir, 'timeline_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)

    with open(os.path.join(output_dir, 'collapse_report.txt'), 'w') as f:
        f.write("COLLAPSE CHECK\n" + "=" * 40 + "\n")
        f.write(f"Total windows: {total}\n")
        f.write(f"Prediction counts: {pred_counts}\n")
        f.write(f"Dominant prediction fraction: {dominant_frac:.2%}\n")
        f.write(f"WALK count: {pred_counts['WALK']}\n")
        f.write(f"COLLAPSE WARNING: {'YES' if collapse else 'no'}\n")

    # --- konsola özet ---
    print(f"\n[+] Prediction dağılımı: {pred_counts}")
    print(f"[+] WALK recall: {walk_recall:.2%}" if not np.isnan(walk_recall) else "[+] WALK recall: N/A")
    print(f"[+] STOP recall: {stop_recall:.2%}" if not np.isnan(stop_recall) else "[+] STOP recall: N/A")
    print(f"[+] IDLE false positive rate: {idle_fp_rate:.2%}")
    print(f"\nConfusion matrix (satır=gerçek, sütun=tahmin):")
    print(f"{'':>12}{'pred_STOP':>12}{'pred_WALK':>12}{'pred_IDLE':>12}")
    for i, c in enumerate(classes):
        print(f"{'true_'+LABEL_NAMES[c]:>12}{cm[i,0]:>12}{cm[i,1]:>12}{cm[i,2]:>12}")

    if collapse:
        print(f"\n[!!!] COLLAPSE WARNING: dominant prediction = {dominant_frac:.1%} "
              f"of all windows, or WALK count = 0. MODEL NOT DEPLOYABLE YET.")
    else:
        print(f"\n[+] Collapse yok - dominant prediction {dominant_frac:.1%} (<90%), WALK count>0.")

    print(f"\n[+] Çıktılar: {output_dir}/validated_timeline.csv, "
          f"timeline_confusion_matrix.csv, timeline_metrics.json, collapse_report.txt")

    return metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BCI Walk/Stop pipeline v3")
    parser.add_argument('--mode', required=True, choices=['train', 'validate_timeline', 'predict'])
    parser.add_argument('--edf', required=True)
    parser.add_argument('--events', default=None)
    parser.add_argument('--model', default=None, help='trained_model dizini (validate_timeline/predict için)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--event-overlap-threshold', type=float, default=0.5)
    parser.add_argument('--idle-distance-threshold', type=float, default=3.5)
    parser.add_argument('--confidence-threshold', type=float, default=0.6)
    parser.add_argument('--n-features-select', type=int, default=5)
    parser.add_argument('--lda-shrinkage', type=float, default=0.0)
    args = parser.parse_args()

    logger.setLevel(logging.WARNING)

    if args.mode == 'train':
        if not args.events:
            raise SystemExit("--events gerekli (train modu)")
        run_train(args.edf, args.events, args.output_dir,
                   idle_distance_threshold=args.idle_distance_threshold,
                   confidence_threshold=args.confidence_threshold,
                   n_features_select=args.n_features_select,
                   lda_shrinkage=args.lda_shrinkage)

    elif args.mode == 'validate_timeline':
        if not args.events or not args.model:
            raise SystemExit("--events ve --model gerekli (validate_timeline modu)")
        run_validate_timeline(args.edf, args.events, args.model, args.output_dir,
                               overlap_threshold=args.event_overlap_threshold)

    elif args.mode == 'predict':
        raise SystemExit(
            "predict modu henüz eklenmedi - geliştirme planına göre önce "
            "train + validate_timeline sonuçları değerlendirilmeli."
        )


if __name__ == '__main__':
    main()
