# EEG Walk/Stop BCI Pipeline

C3/C4/Cz EEG kayıtlarından, EOG/hareket artefaktı etkisi kontrol edilerek
Walk/Stop komut periyotlarını sınıflandıran offline BCI pipeline'ı.
Pipeline; CSP tabanlı öznitelik çıkarımı, F-score feature selection,
shrinkage-regularized LDA sınıflandırıcı ve LOOCV / cross-session validasyon
kullanır; sonuçta held-out tahminlerden sembolik joystick komutu (WALK/STOP/IDLE)
üretir.

**Kapsam notu:** Bu bir offline validasyon ve sembolik komut üretim sistemidir.
Gerçek zamanlı EEG streaming veya gerçek HID/vJoy kontrolü sağlamaz —
üretilen her "joystick komutu", sabit bir kayıt üzerindeki held-out (LOOCV)
tahminlerinden türetilmiş, terminale/CSV'ye yazılan bir etikettir.

## Ana dosya
- `bci_pipeline.py` — güncel sürüm (v3.6): tek dosya, CLI destekli, tüm pipeline'ı içerir.

## Geçmiş / bağımlılık dosyaları (erken sürümlerde kullanıldı)
- `modern_bci_v2.py` — çekirdek sinyal işleme motoru (ilk versiyonlar `bci_pipeline.py`'nin
  bunu import ettiği modüler yapıdaydı; v1.1'den itibaren tek dosyaya taşındı).
- `edf_reader.py`, `parse_events.py`, `validate_full.py` — ilk gerçek veri validasyon script'i.

## Erken denemeler (terk edildi)
- `production_grade_bci.py`, `realtime_eeg_motor_control.py`, `joystick_output.py` —
  gerçek zamanlı, multi-threaded, vJoy/LSL tabanlı ilk yaklaşım. Karmaşıklığı nedeniyle
  offline validasyon + sembolik komut yaklaşımına pivot edildi (bkz. CHANGELOG).

## Kullanım (güncel sürüm)
```bash
python bci_pipeline.py \
    --edf sub-01_ses-01_task-training_eeg.edf \
    --events sub-01_ses-01_task-training_acq-rexcommand_events.tsv \
    --output-dir results
```

Sürüm geçmişi için `CHANGELOG.md` dosyasına bakın; her sürüm ayrı bir git commit'i olarak
mevcuttur (`git log --oneline`).

## Veri
Ham EEG kayıtları (`.edf`) ve eğitilmiş model dosyaları (`.npz`, `.npy`) `.gitignore` ile
repo dışında tutulur (dosya boyutu + denek verisi gizliliği).
