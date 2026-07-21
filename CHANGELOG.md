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

## v2.9 — Corrected filtering, Web UI, multi-session training, and regression testing

### Correctness

- Fixed a longstanding FIR filter configuration error in
  `modern_bci_v2.py`.
- Added `pass_zero=False` to the two-cutoff `scipy.signal.firwin` call.
- The previous implementation constructed a band-stop filter instead of the
  intended band-pass filter, suppressing the configured EEG band while passing
  near-DC and near-Nyquist components.
- Models trained before this correction must be retrained.
- Previously published real-data metrics are considered historical pre-fix
  results until the benchmark suite is regenerated with v2.9.

### Pipeline

- Renamed the main pipeline entry point from `bci_pipeline_v2.8.py` to
  `bci_pipeline_v2.9.py`.
- Retained the original single-session `train`, `validate_timeline`, and
  `predict` workflows.
- Added `train_multi` for pooling event-anchored trials from multiple
  independently preprocessed recordings.
- Added class downsampling through `--balance-classes downsample`.
- Added subject downsampling for multi-session training through
  `--balance-subjects downsample`.
- Removed the unimplemented `class_weight` option.
- Added configurable event-label mappings through `--label-map`.
- Event mappings can be supplied as a JSON object or JSON file and are
  persisted with the trained model.
- Added case-insensitive channel-name matching while preserving the original
  EDF channel names.
- Added per-recording, per-channel z-score normalization and persisted the
  normalization mode in model metadata.
- Added resolved and used training manifests for multi-session provenance.

### Validation and outputs

- Added separate raw and temporally smoothed timeline metrics.
- Added separate raw and smoothed confusion-matrix files.
- Preserved backward-compatible metric aliases for the deployment-facing
  smoothed prediction stream.
- Expanded collapse diagnostics and prediction-distribution reporting.
- Improved model metadata, selected-feature reporting, and result provenance.

## Web UI

- Added `bci_web_ui.py`, a dependency-free local browser interface for the
  v2.9 pipeline.
- Added support for single-session training, multi-session training, timeline
  validation, and unlabeled prediction.
- Added local result summaries, percentage visualizations, generated-file
  listings, and command reporting.
- EEG data remains on the local computer; the interface does not upload
  recordings.
- The Web UI is an interface around the command-line pipeline and does not
  change the underlying model or signal-processing logic.

### Testing

- Added a pytest-based unit-test suite covering:
  - CSP;
  - feature selection;
  - shrinkage LDA;
  - sliding-window construction;
  - temporal smoothing;
  - IDLE gating.
- Added `tests/synthetic_data.py` for controlled synthetic EEG generation.
- Added a minimal plain-EDF writer for end-to-end file-I/O testing.
- Added `tests/test_smoke_synthetic_pipeline.py`.
- The smoke test exercises the complete synthetic
  `EDF + TSV -> train -> save -> load -> validate_timeline -> metrics` path.
- Added regression assertions against single-class model collapse.
- The synthetic smoke test exposed the FIR band-pass configuration error:
  performance was near chance before the correction and strongly above chance
  after it.
- Synthetic-test accuracy is used only as a regression check and must not be
  interpreted as real EEG decoding performance.

### Documentation

- Added installation and Web UI setup instructions.
- Added instructions for running the complete test suite:

  ```bash
  pytest tests/ -v
  ```

## Real-time subsystem (`EEG real time/`)

A separate, experimental real-time-capable classifier and pipeline,
independent from the offline `bci_pipeline_v2.9.1.py` workflow above. See
`EEG real time/ARCHITECTURE.md` for the full design and scope note
(**replay-simulation only, not tested against real EEG hardware**).

- `fast_causal_bci.py` — a from-scratch causal classifier: stateful IIR
  filter bank (`scipy.signal.lfilter`), multi-scale energy/correlation
  features, the same shrinkage-LDA core reused from `modern_bci_v2.py`.
  Documented benchmark in `WEB_UI.md`: ~73.77% cross-session balanced
  accuracy.
- `fast_causal_web_ui.py` — a local, real-time-paced replay demo UI for
  the classifier above.
- `realtime/` package (built around `fast_causal_bci.py` without modifying
  its filter/feature/model core):
  - `eeg_source.py` — `EEGSource`, a source-agnostic streaming Protocol
    (`sampling_rate`, `channels`, `is_live`, `chunks()`, `close()`).
  - `file_replay_source.py` — `FileReplaySource`, replays a recorded EDF
    as real-time-paced device-like chunks (drift-free absolute scheduling
    deadlines).
  - `online_smoothing.py` — `OnlineSmoother`, a causal, delayed
    counterpart to the offline pipeline's centered
    `apply_temporal_smoothing()`; verified byte-for-byte identical given
    the same input.
  - `decision_engine.py` — `DecisionEngine`, the online inference loop:
    predict -> confidence gate (IDLE) -> `OnlineSmoother`, yielding
    correctly-paired raw/smoothed `Decision` objects with per-decision
    latency.
  - `output_sink.py` — `OutputSink` and four implementations
    (`ConsoleOutput`, `CSVOutput`, `WebSocketOutput`, `HardwareOutput`
    placeholder) plus `MultiOutput` fan-out.
  - `web_ui_live.py` — a local HTTP replay UI backed by `DecisionEngine` +
    `WebSocketOutput`, reusing the same UI already verified in
    `fast_causal_web_ui.py`.
- End-to-end verification (`EEG real time/REALTIME_E2E.md`,
  `EEG real time/tests/test_realtime_e2e.py`): the online chain agrees
  exactly with the offline batch reference (`validate()`) on every window
  they both compute; an 8.0s real-time-paced replay took 8.01s wall-clock;
  processing a full 247.98s recording unpaced took only 3.96% of that in
  CPU time.
- 81 tests across this subsystem (`EEG real time/tests/`), independent of
  the root repository's own 24-test suite.

---

