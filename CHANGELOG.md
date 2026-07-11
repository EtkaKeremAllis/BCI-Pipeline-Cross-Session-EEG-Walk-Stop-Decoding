# Changelog

All versions move toward the goal of offline-classifying Walk/Stop intent from
EEG and producing a symbolic joystick command (WALK/STOP/IDLE).

## Phase 0 — Real-time prototype (abandoned, pivoted to offline)
- **Symbolic joystick output layer** — a console-based output layer for WALK/STOP/IDLE.
- **Real-time EEG motor control loop** — a live-reading attempt with LSL streaming.
- **production_grade_bci.py** — Multi-threaded real-time system: EMA baseline locking,
  ICA artifact removal, CAR, CSP+LDA, vJoy + LSL integration. Switched to the offline
  approach once the complexity became unmanageable.

## Phase 1 — Core signal processing engine (`modern_bci_v2.py`)
- **v0.1** — CSP, 40+ time/frequency/spatial features, Laplacian reference, FIR bandpass, config system.
- **v0.2** — CSPFilter / FeatureSelector / SimpleLDA classes, synthetic motor-imagery trial generator, k-fold CV.
- **v0.3** — Refactor: unnecessary complexity dropped, leaving only CSP + F-score feature selection + shrinkage-LDA (487 lines).

## Phase 2 — Real data validation
- **validate_full.py + edf_reader.py + parse_events.py** — the first real LOOCV baseline
  + EOG artifact/correlation/cleaning tests on the sub-01 training session.

## Phase 3 — `bci_pipeline.py` single-file CLI
- **v1.1** — Reorganized into modular functions, CLI added, structured output (metrics.json, commands.csv, summary.txt).
- **v1.1.1** — Model persistence, ROC curve + confusion matrix plots, TSV-first event parsing, `OutputDevice` abstraction.
- **v2.0** — `train_validate` / `predict` mode split, vJoy/ViGEm output backends, command smoothing.

## Phase 4 — v3.demo line 'experimental' (critical bug fix + iteration)
- **v3.0** — **Critical fix**: the distribution of the training data (labeled event
  windows) didn't match the prediction data (sliding window over the continuous
  recording), causing model collapse. Redesigned `validate_timeline` mode, IDLE
  confidence gate, and model persistence.
- **v3.1** — Multi-dataset / cross-session training support (`run_train_multi`).
- **v3.2** — WALK/STOP class balancing.
- **v3.3** — Fine-tuned sliding window size/step: 5.0s/1.0s -> 3.0s/0.25s.
- **v3.4** — Subject-level balancing + unlabeled prediction pipeline over the full recording (`run_unlabeled_prediction`).
- **v3.5** — Temporal smoothing for predicted labels, timeline metrics, confusion matrix CSV export.

## Phase 5 - v2 final series 
- 
**v2.8**
• Flexible channel-set selection (motor3 / motor5 / motor9 / motor13 / all_eeg)
• Temporal smoothing
• Timeline metrics and confusion-matrix export
• Optional per-channel z-score normalization
• Model metadata improvements and backward compatibility

---

