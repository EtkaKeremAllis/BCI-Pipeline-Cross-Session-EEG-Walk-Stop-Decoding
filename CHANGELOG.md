# Changelog

Tüm sürümler, EEG'den (C3/C4/Cz) Walk/Stop niyetini offline olarak sınıflandırıp
sembolik joystick komutu (WALK/STOP/IDLE) üretme hedefine doğru ilerler.

## Faz 0 — Gerçek zamanlı prototip (terk edildi, offline'a pivot)
- **Symbolic joystick output layer** — WALK/STOP/IDLE için console tabanlı çıktı katmanı.
- **Real-time EEG motor control loop** — LSL streaming ile canlı okuma denemesi.
- **production_grade_bci.py** — Multi-threaded gerçek zamanlı sistem: EMA baseline locking,
  ICA artefakt temizleme, CAR, CSP+LDA, vJoy + LSL entegrasyonu. Karmaşıklığı yönetilemez
  hale geldiği için offline yaklaşıma geçildi.

## Faz 1 — Çekirdek sinyal işleme motoru (`modern_bci_v2.py`)
- **v0.1** — CSP, 40+ zaman/frekans/uzamsal öznitelik, Laplacian referans, FIR bandpass, config sistemi.
- **v0.2** — CSPFilter / FeatureSelector / SimpleLDA sınıfları, sentetik motor-imagery trial üretici, k-fold CV.
- **v0.3** — Refactor: gereksiz karmaşıklık atıldı, sadece CSP + F-score feature selection + shrinkage-LDA kaldı (487 satır).

## Faz 2 — Gerçek veri doğrulaması
- **validate_full.py + edf_reader.py + parse_events.py** — sub-01 training session üzerinde
  ilk gerçek LOOCV baseline + EOG artefakt/korelasyon/temizleme testleri.

## Faz 3 — `bci_pipeline.py` tek dosya CLI
- **v1.1** — Modüler fonksiyonlara reorganize edildi, CLI eklendi, structured output (metrics.json, commands.csv, summary.txt).
- **v1.1.1** — Model persistence, ROC curve + confusion matrix plotları, TSV-first event parsing, `OutputDevice` soyutlaması.
- **v2.0** — `train_validate` / `predict` mod ayrımı, vJoy/ViGEm çıktı backend'leri, command smoothing.

## Faz 4 — v3 hattı (kritik hata düzeltmesi + iterasyon)
- **v3.0** — **Kritik düzeltme**: training verisi (etiketli event pencereleri) ile prediction
  verisinin (sürekli kayıt üzerinde sliding window) dağılımı uyuşmuyordu → model collapse
  oluyordu. `validate_timeline` modu, IDLE confidence gate, model persistence yeniden tasarlandı.
- **v3.1** — Multi-dataset / cross-session training desteği (`run_train_multi`).
- **v3.2** — WALK/STOP class balancing.
- **v3.3** — Sliding window boyutu/adımı ince ayar: 5.0s/1.0s → 3.0s/0.25s.
- **v3.4** — Subject-level balancing + tam kayıt üzerinde etiketsiz prediction pipeline (`run_unlabeled_prediction`).
- **v3.5** — Predicted label'lar için temporal smoothing, timeline metrikleri, confusion matrix CSV export.
- **v3.6 (mevcut en güncel)** — Sabit C3/C4/Cz yerine esnek kanal seti çözümleme (`resolve_channels`).

---

### Not: Atlanan/birleştirilen kopyalar
Yüklenen dosyaların önemli bir kısmı birebir kopyaydı (aynı içerik, farklı dosya adı) ve
bu geçmişe dahil edilmedi: `bci_pipeline_v3_1_.py` (= v3.0), `bci_pipeline_2_.py` (= v2.0),
`bci_pipeline_v3_5__1_/_2_/v3_6_.py` (= v3.4), `bci_pipeline_v2_6_1_.py` (= v3.5),
`validate_full`'un 3 ek kopyası, `parse_events_1_.py`, `edf_reader_1_.py`,
`modern_bci_v2_3_.py` (= v0.3), `modern_bci_vclaude3.py` (v0.2'ye çok yakın, ~37 satır fark).
