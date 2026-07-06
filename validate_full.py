"""
DOĞRULAMA PIPELINE'I - Gerçek EEG verisi (sub-01, task-training)
==================================================================
Yeni algoritma YOK. Sadece:
1) Veri okuma ve etiket doğrulama
2) Epoch çıkarma
3) Baseline model (LOOCV)
4) EOG artefakt testi (EEG / EOG / EEG+EOG)
5) EOG korelasyon analizi
6) EOG temizleme testi (basit lineer regresyon)

Mevcut modern_bci_v2.py pipeline'ı (CSP + feature selection + shrinkage LDA)
aynen kullanılıyor - yeni bir sınıflandırma yöntemi eklenmedi.
"""
import numpy as np
from edf_reader import read_edf
from parse_events import parse_events
from modern_bci_v2 import BCIConfig, ModernBCIPipeline, logger
import logging

np.random.seed(42)

EDF_PATH = '/mnt/user-data/uploads/sub-01_ses-01_task-training_eeg.edf'
CMD_EVENTS_PATH = '/mnt/user-data/uploads/sub-01_ses-01_task-training_acq-rexcommand_events.tsv'

FS = 100
WINDOW_LEN_S = 5.0
SKIP_START_S = 1.0
SKIP_END_S = 1.0
LABEL_MAP = {'x5': 0, 'x8': 1}  # x5=STOP=0, x8=WALK=1. x99 (varsa) haritada yok -> atlanır.

BEST_SHRINKAGE = 0.0
BEST_K = 5  # önceki CV grid search'ten (bkz. önceki analiz)


def make_config(channels, n_features_select=BEST_K, shrinkage=BEST_SHRINKAGE, use_csp=True):
    return BCIConfig(
        sampling_rate=FS, n_channels=len(channels), channels=channels,
        notch_freq=60, use_notch=False,   # 60Hz notch 100Hz sampling'te imkansız (Nyquist=50Hz)
        bandpass_low=0.5, bandpass_high=45,
        fir_order=200, use_laplacian=True, use_csp=use_csp,
        lda_shrinkage=shrinkage, n_features_select=n_features_select,
    )


def roc_auc(y_true, y_score):
    """Basit ROC-AUC (Mann-Whitney U tabanlı), ek bağımlılık yok"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    n_pos, n_neg = len(pos), len(neg)
    ranks = stats_rankdata(np.concatenate([pos, neg]))
    sum_ranks_pos = ranks[:n_pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def stats_rankdata(a):
    from scipy.stats import rankdata
    return rankdata(a)


def run_loocv(config, X, y):
    """
    Leave-one-out CV. Her fold'da: kalan (n-1) trial ile CSP+feature
    selection+shrinkage LDA eğitilir, dışarıda bırakılan tek trial
    tahmin edilir. Tüm fold'ların gerçek/tahmin/confidence'ı toplanır.
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
        auc = roc_auc(y_true_all, y_conf_all)
    except Exception:
        auc = float('nan')

    return {
        'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'roc_auc': auc, 'confusion': (tn, fp, fn, tp),
        'y_true': y_true_all, 'y_pred': y_pred_all, 'y_conf': y_conf_all,
    }


def print_confusion(cm, labels=('STOP', 'WALK')):
    tn, fp, fn, tp = cm
    print(f"{'':>15}{'Pred: ' + labels[0]:>13}{'Pred: ' + labels[1]:>13}")
    print(f"{'Gerçek: ' + labels[0]:>15}{tn:>13}{fp:>13}")
    print(f"{'Gerçek: ' + labels[1]:>15}{fn:>13}{tp:>13}")


# ============================================================================
# BÖLÜM 1: VERİ OKUMA VE ETİKET DOĞRULAMA
# ============================================================================
print("=" * 70)
print("BÖLÜM 1: VERİ OKUMA VE ETİKET DOĞRULAMA")
print("=" * 70)

signals, info = read_edf(EDF_PATH)

print(f"\n[1] EDF kanal listesi ({info['n_signals']} kanal):")
print(info['labels'])

required_eeg = ['C3', 'C4', 'Cz']
required_eog = ['EOG_HL', 'EOG_HR', 'EOG_VA', 'EOG_VB']
print(f"\n[2] Kanal doğrulama:")
for ch in required_eeg + required_eog:
    present = ch in info['labels']
    print(f"    {ch:10s}: {'VAR' if present else 'YOK!!'}")

print(f"\n[3] rexcommand events tablosu:")
cmd_events = parse_events(CMD_EVENTS_PATH)
print(f"{'onset':>8}{'duration':>10}{'trial_type':>12}")
for onset, duration, trial_type in cmd_events:
    print(f"{onset:>8.2f}{duration:>10.2f}{trial_type:>12}")

print(f"\n[4-5] Label mapping: x5=STOP=0, x8=WALK=1 (x99 haritada yok -> düşer)")
print(f"\n[6] Her event için onset/duration/trial_type/label:")
print(f"{'onset':>8}{'duration':>10}{'trial_type':>12}{'label':>8}")
usable_events = []
dropped_events = []
for onset, duration, trial_type in cmd_events:
    if trial_type in LABEL_MAP:
        label = LABEL_MAP[trial_type]
        print(f"{onset:>8.2f}{duration:>10.2f}{trial_type:>12}{label:>8}")
        usable_events.append((onset, duration, trial_type, label))
    else:
        print(f"{onset:>8.2f}{duration:>10.2f}{trial_type:>12}{'DROP':>8}")
        dropped_events.append((onset, duration, trial_type))

n_stop_events = sum(1 for e in usable_events if e[3] == 0)
n_walk_events = sum(1 for e in usable_events if e[3] == 1)
print(f"\n[7] Event-seviyesi özet: STOP events={n_stop_events}, WALK events={n_walk_events}, "
      f"dropped={len(dropped_events)}")
print("    (Bölüm 2'de bu event'ler 5s pencerelere bölünecek; beklenen trial-seviyesi "
      "sayılar STOP=26, WALK=15, Total=41 - önceki analizle eşleşiyor)")


# ============================================================================
# BÖLÜM 2: EPOCH ÇIKARMA
# ============================================================================
print("\n" + "=" * 70)
print("BÖLÜM 2: EPOCH ÇIKARMA")
print("=" * 70)


def build_windows(events, fs, skip_start, skip_end, window_len):
    """
    events: (onset, duration, trial_type, label) listesi
    Her event: baştan skip_start, sondan skip_end atılır, kalan aralıktan
    non-overlapping window_len'lik pencereler çıkarılır. Pencere sığmayan
    (çok kısa) event'ler atlanır.
    """
    windows = []  # (start_idx, end_idx, label)
    for onset, duration, trial_type, label in events:
        usable_start = onset + skip_start
        usable_end = onset + duration - skip_end
        usable_duration = usable_end - usable_start
        n_windows = int(usable_duration // window_len)
        if n_windows <= 0:
            print(f"    [ATLANDI - çok kısa] onset={onset:.2f} duration={duration:.2f} "
                  f"trial_type={trial_type} (kullanılabilir süre={max(usable_duration,0):.2f}s < {window_len}s)")
            continue
        for w in range(n_windows):
            start_t = usable_start + w * window_len
            start_idx = int(start_t * fs)
            end_idx = start_idx + int(window_len * fs)
            windows.append((start_idx, end_idx, label))
    return windows


print(f"\n[*] Pencereleme parametreleri: skip_start={SKIP_START_S}s, skip_end={SKIP_END_S}s, "
      f"window_len={WINDOW_LEN_S}s")
windows = build_windows(usable_events, FS, SKIP_START_S, SKIP_END_S, WINDOW_LEN_S)

y = np.array([w[2] for w in windows])
n_stop = int(np.sum(y == 0))
n_walk = int(np.sum(y == 1))
print(f"\n[+] X = [trial1, trial2, ..., trial{len(windows)}]")
print(f"[+] y = {y.tolist()}")
print(f"\nSTOP: {n_stop}")
print(f"WALK: {n_walk}")
print(f"Total usable trials: {len(windows)}")


def epochs_from_windows(preprocessed_dict, windows, channels):
    X = []
    y_out = []
    max_len = len(next(iter(preprocessed_dict.values())))
    for start_idx, end_idx, label in windows:
        if end_idx > max_len:
            continue
        epoch = {ch: preprocessed_dict[ch][start_idx:end_idx] for ch in channels}
        X.append(epoch)
        y_out.append(label)
    return X, np.array(y_out)


# ============================================================================
# BÖLÜM 3: BASELINE MODEL (C3, C4, Cz - LOOCV)
# ============================================================================
print("\n" + "=" * 70)
print("BÖLÜM 3: BASELINE MODEL (C3, C4, Cz)")
print("=" * 70)
print("Pipeline: EEG -> bandpass -> feature extraction -> CSP -> feature selection -> shrinkage LDA -> LOOCV")
print("(NOT: notch devre dışı [Nyquist=50Hz < 60Hz line noise]; Laplacian bu kanal setinde")
print(" no-op çünkü komşu elektrotlar (FC3/CP3/C1/C5 vb.) EDF'den yüklenmedi - sadece C3/C4/Cz var.)")

eeg_channels = ['C3', 'C4', 'Cz']
config_eeg = make_config(eeg_channels)
pipeline_eeg = ModernBCIPipeline(config_eeg)

raw_eeg = {ch: signals[ch] for ch in eeg_channels}
preproc_eeg = pipeline_eeg.preprocess(raw_eeg)
X_eeg, y_eeg = epochs_from_windows(preproc_eeg, windows, eeg_channels)

print(f"\n[*] LOOCV çalıştırılıyor ({len(X_eeg)} fold)...")
result_eeg = run_loocv(config_eeg, X_eeg, y_eeg)

print(f"\nAccuracy : {result_eeg['accuracy']:.2%}")
print(f"Precision: {result_eeg['precision']:.2%}")
print(f"Recall   : {result_eeg['recall']:.2%}")
print(f"F1 Score : {result_eeg['f1']:.2%}")
print(f"ROC-AUC  : {result_eeg['roc_auc']:.3f}")
print("Confusion Matrix:")
print_confusion(result_eeg['confusion'])


# ============================================================================
# BÖLÜM 4: EOG ARTEFAKT TESTİ (A: EEG, B: EOG, C: EEG+EOG)
# ============================================================================
print("\n" + "=" * 70)
print("BÖLÜM 4: EOG ARTEFAKT TESTİ")
print("=" * 70)

eog_channels = ['EOG_HL', 'EOG_HR', 'EOG_VA', 'EOG_VB']
combined_channels = eeg_channels + eog_channels

# A) Sadece EEG - Bölüm 3'te zaten hesaplandı, tekrar kullan
print("\n[A] Sadece EEG (C3, C4, Cz) - Bölüm 3 sonucu tekrar kullanılıyor")
result_A = result_eeg

# B) Sadece EOG
print("\n[B] Sadece EOG (EOG_HL, EOG_HR, EOG_VA, EOG_VB)")
config_eog = make_config(eog_channels)
pipeline_eog = ModernBCIPipeline(config_eog)
raw_eog = {ch: signals[ch] for ch in eog_channels}
preproc_eog = pipeline_eog.preprocess(raw_eog)
X_eog, y_eog = epochs_from_windows(preproc_eog, windows, eog_channels)
print(f"[*] LOOCV çalıştırılıyor ({len(X_eog)} fold)...")
result_B = run_loocv(config_eog, X_eog, y_eog)

# C) EEG + EOG
print("\n[C] EEG + EOG (C3, C4, Cz, EOG_HL, EOG_HR, EOG_VA, EOG_VB)")
config_combined = make_config(combined_channels)
pipeline_combined = ModernBCIPipeline(config_combined)
raw_combined = {ch: signals[ch] for ch in combined_channels}
preproc_combined = pipeline_combined.preprocess(raw_combined)
X_combined, y_combined = epochs_from_windows(preproc_combined, windows, combined_channels)
print(f"[*] LOOCV çalıştırılıyor ({len(X_combined)} fold)...")
result_C = run_loocv(config_combined, X_combined, y_combined)

print("\n" + "-" * 60)
print(f"{'Koşul':>20}{'Accuracy':>12}{'F1':>10}{'ROC-AUC':>10}")
print("-" * 60)
print(f"{'A) Sadece EEG':>20}{result_A['accuracy']:>12.2%}{result_A['f1']:>10.2%}{result_A['roc_auc']:>10.3f}")
print(f"{'B) Sadece EOG':>20}{result_B['accuracy']:>12.2%}{result_B['f1']:>10.2%}{result_B['roc_auc']:>10.3f}")
print(f"{'C) EEG+EOG':>20}{result_C['accuracy']:>12.2%}{result_C['f1']:>10.2%}{result_C['roc_auc']:>10.3f}")
print("-" * 60)

print("\nYorum:")
if result_B['accuracy'] < 0.65:
    print(f"  - Sadece EOG accuracy düşük ({result_B['accuracy']:.2%}) -> iyi işaret, "
          f"göz hareketi tek başına Walk/Stop'u ayırt etmiyor.")
else:
    print(f"  - Sadece EOG accuracy YÜKSEK ({result_B['accuracy']:.2%}) -> DİKKAT: "
          f"model muhtemelen artefakt öğreniyor, EEG sonucu şüpheli.")

diff = result_C['accuracy'] - result_A['accuracy']
if diff > 0.05:
    print(f"  - EEG+EOG, sadece EEG'ye göre {diff:.2%} daha yüksek -> EOG katkısı güçlü, "
          f"EEG-only sonucu artefaktla şişmiş olabilir.")
else:
    print(f"  - EEG+EOG ({result_C['accuracy']:.2%}) ile sadece EEG ({result_A['accuracy']:.2%}) "
          f"arasında büyük fark yok -> EOG eklenmesi sonucu belirgin şekilde artırmıyor.")


# ============================================================================
# BÖLÜM 5: EOG KORELASYON ANALİZİ
# ============================================================================
print("\n" + "=" * 70)
print("BÖLÜM 5: EOG KORELASYON ANALİZİ")
print("=" * 70)

pairs = [('C3', 'EOG_VA'), ('C4', 'EOG_VA'), ('Cz', 'EOG_VA'),
         ('C3', 'EOG_HL'), ('C4', 'EOG_HL')]

# Aynı preprocess edilmiş sinyaller (Bölüm 3/4'ten) - EEG ve EOG ayrı ayrı preprocess edilmişti
all_signals_preproc = {**preproc_eeg, **preproc_eog}

corr_rows = []
for start_idx, end_idx, label in windows:
    row = {'label': label}
    for ch1, ch2 in pairs:
        s1 = all_signals_preproc[ch1][start_idx:end_idx]
        s2 = all_signals_preproc[ch2][start_idx:end_idx]
        if np.std(s1) > 0 and np.std(s2) > 0:
            row[f"{ch1}-{ch2}"] = np.corrcoef(s1, s2)[0, 1]
        else:
            row[f"{ch1}-{ch2}"] = 0.0
    corr_rows.append(row)

pair_names = [f"{a}-{b}" for a, b in pairs]
stop_corrs = {p: np.mean([r[p] for r in corr_rows if r['label'] == 0]) for p in pair_names}
walk_corrs = {p: np.mean([r[p] for r in corr_rows if r['label'] == 1]) for p in pair_names}

print(f"\n{'Kanal çifti':>15}{'STOP ort. corr':>18}{'WALK ort. corr':>18}{'Fark':>10}")
for p in pair_names:
    diff = walk_corrs[p] - stop_corrs[p]
    print(f"{p:>15}{stop_corrs[p]:>18.3f}{walk_corrs[p]:>18.3f}{diff:>10.3f}")

max_abs_diff = max(abs(walk_corrs[p] - stop_corrs[p]) for p in pair_names)
print(f"\nYorum: en büyük |WALK-STOP| korelasyon farkı = {max_abs_diff:.3f}")
if max_abs_diff > 0.15:
    print("  -> WALK sırasında EOG korelasyonu belirgin şekilde değişiyor - göz/hareket "
          "etkisi olası, EEG sonucu temkinli yorumlanmalı.")
else:
    print("  -> STOP/WALK arasında EOG korelasyonu belirgin farklılaşmıyor - EEG kanallarının "
          "EOG'dan bağımsız bilgi taşıdığına dair zayıf bir kanıt.")


# ============================================================================
# BÖLÜM 6: EOG TEMİZLEME TESTİ (basit lineer regresyon)
# ============================================================================
print("\n" + "=" * 70)
print("BÖLÜM 6: EOG TEMİZLEME TESTİ")
print("=" * 70)
print("EEG_clean = EEG - (EOG kanallarından lineer regresyonla tahmin edilen kısım)")
print("Regresyon katsayıları TÜM sürekli kayıt üzerinde (etiketsiz, sızıntı yok) hesaplanıyor.\n")

# EOG regressor matrisi (sürekli kayıt, preprocess edilmiş EOG sinyalleri)
EOG_mat = np.column_stack([preproc_eog[ch] for ch in eog_channels])  # (n_samples, 4)
EOG_design = np.column_stack([EOG_mat, np.ones(len(EOG_mat))])  # + intercept

preproc_eeg_clean = {}
betas = {}
for ch in eeg_channels:
    target = preproc_eeg[ch]
    beta, *_ = np.linalg.lstsq(EOG_design, target, rcond=None)
    betas[ch] = beta
    predicted = EOG_design @ beta
    preproc_eeg_clean[ch] = target - predicted
    r2 = 1 - np.sum((target - predicted) ** 2) / np.sum((target - target.mean()) ** 2)
    print(f"  {ch}: EOG regresyonu R^2 = {r2:.4f} (EOG'un açıkladığı varyans oranı)")

X_clean, y_clean = epochs_from_windows(preproc_eeg_clean, windows, eeg_channels)
print(f"\n[*] Temizlenmiş EEG ile LOOCV çalıştırılıyor ({len(X_clean)} fold)...")
result_clean = run_loocv(config_eeg, X_clean, y_clean)

print("\n" + "-" * 60)
print(f"{'Koşul':>25}{'Accuracy':>12}{'F1':>10}{'ROC-AUC':>10}")
print("-" * 60)
print(f"{'Raw EEG':>25}{result_A['accuracy']:>12.2%}{result_A['f1']:>10.2%}{result_A['roc_auc']:>10.3f}")
print(f"{'EOG-only':>25}{result_B['accuracy']:>12.2%}{result_B['f1']:>10.2%}{result_B['roc_auc']:>10.3f}")
print(f"{'EOG-cleaned EEG':>25}{result_clean['accuracy']:>12.2%}{result_clean['f1']:>10.2%}{result_clean['roc_auc']:>10.3f}")
print("-" * 60)

drop = result_A['accuracy'] - result_clean['accuracy']
print(f"\nYorum: EOG temizleme sonrası accuracy değişimi: {-drop:+.2%}")
if result_clean['accuracy'] >= result_A['accuracy'] - 0.05:
    print("  -> Skor temizleme sonrası da yüksek kalıyor - EEG sinyalinin EOG'a bağımlı "
          "olmadığına dair olumlu kanıt.")
else:
    print("  -> Skor temizleme sonrası belirgin düştü - önceki EEG sonucunun bir kısmı "
          "EOG artefaktından geliyor olabilir.")

print("\n" + "=" * 70)
print("DOĞRULAMA TAMAMLANDI (Adım 1-6). Adım 7 (rexstate ikinci analiz) bekliyor.")
print("=" * 70)
