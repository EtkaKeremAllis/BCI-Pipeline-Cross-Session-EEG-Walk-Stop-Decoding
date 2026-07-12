#!/usr/bin/env python3
"""
BCI Pipeline Web UI
===================

A small, dependency-free local web interface for bci_pipeline_v2.9.py.
It does not change the pipeline architecture or model logic. It only builds
and runs the same CLI commands from a browser form.

Run:
    python bci_web_ui.py

Then open:
    http://127.0.0.1:8765

Recommended: keep this file in the same folder as bci_pipeline_v2.9.py and the
helper modules imported by that pipeline.
"""

from __future__ import annotations

import html
import json
import math
import os
import shlex
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 8765

VALID_MODES = {"train", "train_multi", "validate_timeline", "predict"}
VALID_CHANNEL_SETS = {"motor3", "motor5", "motor9", "motor13", "all_eeg"}
VALID_CHANNEL_NORMALIZATION = {"none", "zscore"}
VALID_BALANCE_CLASSES = {"none", "downsample"}
VALID_BALANCE_SUBJECTS = {"none", "downsample"}
VALID_SMOOTHING_WINDOWS = {"1", "3", "5"}

DEFAULTS = {
    "mode": "validate_timeline",
    "pipeline_path": "",
    "edf": "",
    "events": "",
    "model": "results/model",
    "output_dir": "results/ui_run",
    "dataset_list": "",
    "dataset_dir": ".",
    "event_overlap_threshold": "0.5",
    "idle_distance_threshold": "999.0",
    "confidence_threshold": "0.45",
    "n_features_select": "45",
    "lda_shrinkage": "0.0",
    "balance_classes": "none",
    "balance_subjects": "none",
    "seed": "42",
    "smoothing_window": "3",
    "channel_normalization": "zscore",
    "channel_set": "motor3",
}


def guess_pipeline_path() -> str:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "bci_pipeline_v2.9.py",
        here / "bci_pipeline_v2.8.py",
        here / "bci_pipeline_v2.8 (1).py",
        here / "bci_pipeline_v2.8(1).py",
        here / "bci_pipeline_v3.py",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(here / "bci_pipeline_v2.9.py")


def json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def require(payload: Dict[str, Any], key: str, label: str, errors: List[str]) -> str:
    value = clean_string(payload.get(key))
    if not value:
        errors.append(f"{label} is required.")
    return value


def choice(payload: Dict[str, Any], key: str, allowed: set[str], default: str, label: str, errors: List[str]) -> str:
    value = clean_string(payload.get(key)) or default
    if value not in allowed:
        errors.append(f"{label} must be one of: {', '.join(sorted(allowed))}.")
        return default
    return value


def numeric_string(payload: Dict[str, Any], key: str, default: str, label: str, errors: List[str], as_int: bool = False) -> str:
    value = clean_string(payload.get(key)) or default
    try:
        if as_int:
            int(value)
        else:
            float(value)
    except ValueError:
        errors.append(f"{label} must be numeric.")
        return default
    return value


def build_command(payload: Dict[str, Any]) -> tuple[Optional[List[str]], List[str], Optional[Path]]:
    errors: List[str] = []

    mode = choice(payload, "mode", VALID_MODES, DEFAULTS["mode"], "Mode", errors)
    pipeline_path = clean_string(payload.get("pipeline_path")) or guess_pipeline_path()
    pipeline = Path(pipeline_path).expanduser().resolve()

    if not pipeline.exists():
        errors.append(f"Pipeline script was not found: {pipeline}")
    if pipeline.suffix.lower() != ".py":
        errors.append("Pipeline script must be a Python .py file.")

    output_dir = require(payload, "output_dir", "Output directory", errors)
    cmd = [sys.executable, str(pipeline), "--mode", mode, "--output-dir", output_dir]

    if mode == "train":
        cmd += ["--edf", require(payload, "edf", "EDF file", errors)]
        cmd += ["--events", require(payload, "events", "Events file", errors)]
        cmd += ["--channel-set", choice(payload, "channel_set", VALID_CHANNEL_SETS, "motor3", "Channel set", errors)]
        cmd += ["--channel-normalization", choice(payload, "channel_normalization", VALID_CHANNEL_NORMALIZATION, "zscore", "Channel normalization", errors)]
        cmd += ["--idle-distance-threshold", numeric_string(payload, "idle_distance_threshold", "999.0", "Idle distance threshold", errors)]
        cmd += ["--confidence-threshold", numeric_string(payload, "confidence_threshold", "0.45", "Confidence threshold", errors)]
        cmd += ["--n-features-select", numeric_string(payload, "n_features_select", "45", "Selected feature count", errors, as_int=True)]
        cmd += ["--lda-shrinkage", numeric_string(payload, "lda_shrinkage", "0.0", "LDA shrinkage", errors)]
        cmd += ["--balance-classes", choice(payload, "balance_classes", VALID_BALANCE_CLASSES, "none", "Class balancing", errors)]
        cmd += ["--seed", numeric_string(payload, "seed", "42", "Seed", errors, as_int=True)]

    elif mode == "train_multi":
        cmd += ["--dataset-list", require(payload, "dataset_list", "Dataset list CSV", errors)]
        dataset_dir = clean_string(payload.get("dataset_dir")) or "."
        cmd += ["--dataset-dir", dataset_dir]
        cmd += ["--channel-set", choice(payload, "channel_set", VALID_CHANNEL_SETS, "motor3", "Channel set", errors)]
        cmd += ["--channel-normalization", choice(payload, "channel_normalization", VALID_CHANNEL_NORMALIZATION, "zscore", "Channel normalization", errors)]
        cmd += ["--idle-distance-threshold", numeric_string(payload, "idle_distance_threshold", "999.0", "Idle distance threshold", errors)]
        cmd += ["--confidence-threshold", numeric_string(payload, "confidence_threshold", "0.45", "Confidence threshold", errors)]
        cmd += ["--n-features-select", numeric_string(payload, "n_features_select", "45", "Selected feature count", errors, as_int=True)]
        cmd += ["--lda-shrinkage", numeric_string(payload, "lda_shrinkage", "0.0", "LDA shrinkage", errors)]
        cmd += ["--balance-classes", choice(payload, "balance_classes", VALID_BALANCE_CLASSES, "none", "Class balancing", errors)]
        cmd += ["--balance-subjects", choice(payload, "balance_subjects", VALID_BALANCE_SUBJECTS, "none", "Subject balancing", errors)]
        cmd += ["--seed", numeric_string(payload, "seed", "42", "Seed", errors, as_int=True)]

    elif mode == "validate_timeline":
        cmd += ["--edf", require(payload, "edf", "EDF file", errors)]
        cmd += ["--model", require(payload, "model", "Model directory", errors)]
        events = clean_string(payload.get("events"))
        if events:
            cmd += ["--events", events]
        cmd += ["--event-overlap-threshold", numeric_string(payload, "event_overlap_threshold", "0.5", "Event overlap threshold", errors)]
        smoothing = choice(payload, "smoothing_window", VALID_SMOOTHING_WINDOWS, "3", "Smoothing window", errors)
        cmd += ["--smoothing-window", smoothing]

    elif mode == "predict":
        cmd += ["--edf", require(payload, "edf", "EDF file", errors)]
        cmd += ["--model", require(payload, "model", "Model directory", errors)]
        smoothing = choice(payload, "smoothing_window", VALID_SMOOTHING_WINDOWS, "3", "Smoothing window", errors)
        cmd += ["--smoothing-window", smoothing]

    if errors:
        return None, errors, pipeline.parent if pipeline.exists() else None
    return cmd, [], pipeline.parent


def resolve_output_dir(output_dir: str, cwd: Optional[Path]) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        base = cwd if cwd is not None else Path.cwd()
        path = base / path
    return path.resolve()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def fraction_to_percent(value: Any) -> Optional[float]:
    v = finite_float(value)
    if v is None:
        return None
    if -1.0 <= v <= 1.0:
        return round(v * 100.0, 2)
    return round(v, 2)


def add_bar(bars: List[Dict[str, Any]], label: str, value: Any, group: str) -> None:
    pct = fraction_to_percent(value)
    if pct is None:
        return
    bars.append({"label": label, "value": max(0.0, min(100.0, pct)), "group": group})


def counts_to_shares(counts: Dict[str, Any], group: str) -> List[Dict[str, Any]]:
    parsed = {str(k): finite_float(v) or 0.0 for k, v in counts.items()}
    total = sum(parsed.values())
    if total <= 0:
        return []
    return [
        {"label": f"{group} {name} share", "value": round((count / total) * 100.0, 2), "group": group}
        for name, count in parsed.items()
    ]


def collect_metrics(output_dir: str, cwd: Optional[Path] = None, expect_timeline_metrics: bool = False) -> Dict[str, Any]:
    out = resolve_output_dir(output_dir, cwd)
    bars: List[Dict[str, Any]] = []
    shares: List[Dict[str, Any]] = []
    files: List[Dict[str, str]] = []
    notes: List[str] = []

    timeline = read_json(out / "timeline_metrics.json")
    prediction = read_json(out / "prediction_summary.json")
    model_info = read_json(out / "model_info.json")

    for filename in [
        "timeline_metrics.json",
        "prediction_summary.json",
        "validated_timeline.csv",
        "predicted_timeline.csv",
        "timeline_confusion_matrix.csv",
        "timeline_confusion_matrix_raw.csv",
        "timeline_confusion_matrix_smoothed.csv",
        "collapse_report.txt",
        "training_summary.txt",
        "model_info.json",
        "selected_features.json",
        "trained_model.npz",
    ]:
        path = out / filename
        if path.exists():
            files.append({"name": filename, "path": str(path)})

    if expect_timeline_metrics and timeline is None:
        notes.append(
            "An events file was supplied, but timeline_metrics.json was not found in the resolved output directory. "
            "Accuracy and balanced accuracy could not be displayed."
        )

    if timeline:
        for stream_name, stream_label in [("smoothed", "Smoothed"), ("raw", "Raw")]:
            stream = timeline.get(stream_name, {}) or {}
            add_bar(bars, f"{stream_label} window-level accuracy", stream.get("deployment_accuracy_non_idle"), "Validation")
            add_bar(bars, f"{stream_label} balanced accuracy", stream.get("balanced_accuracy"), "Validation")
            add_bar(bars, f"{stream_label} WALK recall", stream.get("walk_recall"), "Validation")
            add_bar(bars, f"{stream_label} STOP recall", stream.get("stop_recall"), "Validation")
            add_bar(bars, f"{stream_label} idle false positive rate", stream.get("idle_false_positive_rate"), "Validation")
            add_bar(bars, f"{stream_label} dominant prediction", stream.get("dominant_prediction_fraction"), "Prediction balance")
            counts = stream.get("prediction_counts")
            if isinstance(counts, dict):
                shares.extend(counts_to_shares(counts, stream_label))

        if timeline.get("collapse_warning"):
            notes.append("Collapse warning is active for the deployment-facing stream.")

    if prediction:
        for stream_name, stream_label in [("smoothed", "Smoothed"), ("raw", "Raw")]:
            stream = prediction.get(stream_name, {}) or {}
            add_bar(bars, f"{stream_label} dominant prediction", stream.get("dominant_prediction_fraction"), "Prediction balance")
            counts = stream.get("prediction_counts")
            if isinstance(counts, dict):
                shares.extend(counts_to_shares(counts, stream_label))
        if prediction.get("collapse_warning"):
            notes.append("Collapse warning is active for the deployment-facing stream.")

    if model_info:
        add_bar(bars, "Confidence threshold", model_info.get("confidence_threshold"), "Model settings")
        # idle_distance_threshold is not a percentage, so it is intentionally not visualized as a percent.

    # Remove duplicate bars while preserving order.
    seen = set()
    unique_bars: List[Dict[str, Any]] = []
    for item in bars:
        key = (item["label"], item["value"], item["group"])
        if key not in seen:
            seen.add(key)
            unique_bars.append(item)

    seen = set()
    unique_shares: List[Dict[str, Any]] = []
    for item in shares:
        key = (item["label"], item["value"], item["group"])
        if key not in seen:
            seen.add(key)
            unique_shares.append(item)

    return {
        "output_dir": str(out),
        "percentage_bars": unique_bars,
        "prediction_shares": unique_shares,
        "files": files,
        "notes": notes,
        "has_metrics": bool(unique_bars or unique_shares),
    }


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BCI Pipeline Web UI</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #151b23;
      --panel-2: #1f2630;
      --text: #eef4ff;
      --muted: #9fb0c7;
      --line: #2d3746;
      --accent: #6aa9ff;
      --accent-2: #9bd67a;
      --danger: #ff7b72;
      --warning: #f2cc60;
      --shadow: 0 18px 50px rgba(0,0,0,.35);
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #172236 0, var(--bg) 44%), var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    header {
      padding: 32px 24px 12px;
      max-width: 1220px;
      margin: 0 auto;
    }
    .eyebrow { color: var(--accent); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: 12px; }
    h1 { margin: 8px 0 8px; font-size: clamp(30px, 5vw, 58px); line-height: .95; }
    .subtitle { max-width: 850px; color: var(--muted); font-size: 16px; line-height: 1.6; }
    main { max-width: 1220px; margin: 0 auto; padding: 20px 24px 60px; display: grid; grid-template-columns: minmax(320px, 480px) 1fr; gap: 20px; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
    .card { background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.015)), var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); }
    .card h2 { margin: 0 0 12px; font-size: 20px; }
    .form-card { padding: 22px; position: sticky; top: 16px; align-self: start; }
    @media (max-width: 980px) { .form-card { position: static; } }
    label { display: block; font-size: 13px; color: var(--muted); margin: 14px 0 7px; }
    input, select {
      width: 100%; background: #0e1420; color: var(--text); border: 1px solid var(--line); border-radius: 12px;
      padding: 12px 13px; outline: none; font-size: 14px;
    }
    input:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(106,169,255,.14); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 520px) { .grid2 { grid-template-columns: 1fr; } }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.45; margin-top: 7px; }
    .mode-note { margin: 12px 0 0; padding: 12px; border: 1px solid var(--line); border-radius: 14px; background: rgba(106,169,255,.07); color: #cfe2ff; font-size: 13px; line-height: 1.45; }
    .section-title { margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--line); color: #dce9ff; font-weight: 800; }
    button {
      margin-top: 20px; width: 100%; border: 0; border-radius: 14px; padding: 14px 16px; font-weight: 800; cursor: pointer;
      color: #07111f; background: linear-gradient(135deg, var(--accent), var(--accent-2)); font-size: 15px;
    }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .result-card { padding: 22px; min-height: 520px; }
    .status { display: flex; align-items: center; gap: 10px; padding: 14px; border-radius: 14px; background: var(--panel-2); border: 1px solid var(--line); color: var(--muted); }
    .dot { width: 12px; height: 12px; border-radius: 999px; background: var(--muted); box-shadow: 0 0 0 6px rgba(159,176,199,.12); }
    .dot.running { background: var(--warning); box-shadow: 0 0 0 6px rgba(242,204,96,.14); }
    .dot.ok { background: var(--accent-2); box-shadow: 0 0 0 6px rgba(155,214,122,.14); }
    .dot.fail { background: var(--danger); box-shadow: 0 0 0 6px rgba(255,123,114,.14); }
    .command { margin-top: 14px; padding: 12px; border-radius: 12px; background: #090d14; border: 1px solid var(--line); color: #cbd8ea; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
    .metrics { margin-top: 18px; display: grid; gap: 12px; }
    .metric-row { background: rgba(255,255,255,.035); border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
    .metric-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 9px; }
    .metric-label { font-weight: 700; font-size: 13px; }
    .metric-value { color: #dce9ff; font-weight: 900; }
    .bar-bg { height: 12px; background: #0b111b; border-radius: 999px; overflow: hidden; border: 1px solid #263143; }
    .bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); border-radius: 999px; transition: width .6s ease; }
    .chart-wrap { margin-top: 18px; padding: 16px; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.03); }
    .chart-title { font-weight: 900; margin-bottom: 12px; }
    canvas { width: 100%; max-width: 100%; background: #0b111b; border: 1px solid var(--line); border-radius: 14px; }
    .files { margin-top: 18px; display: grid; gap: 8px; }
    .file-pill { padding: 10px 12px; border: 1px solid var(--line); border-radius: 12px; background: rgba(255,255,255,.03); color: var(--muted); font-size: 13px; word-break: break-all; }
    .note { margin-top: 12px; padding: 12px; border-left: 4px solid var(--warning); background: rgba(242,204,96,.08); color: #ffe9a6; border-radius: 10px; font-size: 13px; }
    .hidden { display: none !important; }
    .footer-note { margin-top: 14px; color: var(--muted); font-size: 12px; line-height: 1.5; }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">Local BCI Control Panel</div>
    <h1>BCI Pipeline Web UI</h1>
    <p class="subtitle">Select a mode, enter file paths, and start the existing pipeline from your browser. The model logic stays unchanged: this page only builds the same command-line call for you.</p>
  </header>

  <main>
    <section class="card form-card">
      <h2>Run settings</h2>
      <form id="runForm">
        <label for="mode">Mode</label>
        <select id="mode" name="mode">
          <option value="train">Train</option>
          <option value="train_multi">Train multi-session</option>
          <option value="validate_timeline" selected>Validate timeline</option>
          <option value="predict">Predict unlabeled timeline</option>
        </select>
        <div id="modeNote" class="mode-note"></div>

        <label for="pipeline_path">Pipeline script path</label>
        <input id="pipeline_path" name="pipeline_path" value="__PIPELINE_PATH__" />
        <div class="hint">Use the existing pipeline file. Absolute paths are recommended.</div>

        <div class="section-title">Input files</div>
        <div class="field field-edf">
          <label for="edf">EDF file</label>
          <input id="edf" name="edf" placeholder="C:\\path\\to\\recording.edf" />
        </div>
        <div class="field field-events">
          <label for="events">Events file</label>
          <input id="events" name="events" placeholder="C:\\path\\to\\events.tsv or .csv" />
          <div class="hint validate-only">Optional for validation. If empty, prediction runs without accuracy metrics.</div>
        </div>
        <div class="field field-model">
          <label for="model">Model directory</label>
          <input id="model" name="model" value="results/model" />
        </div>
        <div class="field field-dataset-list">
          <label for="dataset_list">Dataset list CSV</label>
          <input id="dataset_list" name="dataset_list" placeholder="C:\\path\\to\\dataset_list.csv" />
        </div>
        <div class="field field-dataset-dir">
          <label for="dataset_dir">Dataset base directory</label>
          <input id="dataset_dir" name="dataset_dir" value="." />
        </div>

        <label for="output_dir">Output directory</label>
        <input id="output_dir" name="output_dir" value="results/ui_run" />

        <div class="section-title train-field train-multi-field">Training options</div>
        <div class="grid2 train-field train-multi-field">
          <div>
            <label for="channel_set">Channel set</label>
            <select id="channel_set" name="channel_set">
              <option value="motor3" selected>motor3</option>
              <option value="motor5">motor5</option>
              <option value="motor9">motor9</option>
              <option value="motor13">motor13</option>
              <option value="all_eeg">all_eeg</option>
            </select>
          </div>
          <div>
            <label for="channel_normalization">Channel normalization</label>
            <select id="channel_normalization" name="channel_normalization">
              <option value="zscore" selected>zscore</option>
              <option value="none">none</option>
            </select>
          </div>
        </div>
        <div class="grid2 train-field train-multi-field">
          <div>
            <label for="n_features_select">Selected feature count</label>
            <input id="n_features_select" name="n_features_select" value="45" />
          </div>
          <div>
            <label for="confidence_threshold">Confidence threshold</label>
            <input id="confidence_threshold" name="confidence_threshold" value="0.45" />
          </div>
        </div>
        <div class="grid2 train-field train-multi-field">
          <div>
            <label for="idle_distance_threshold">Idle distance threshold</label>
            <input id="idle_distance_threshold" name="idle_distance_threshold" value="999.0" />
          </div>
          <div>
            <label for="lda_shrinkage">LDA shrinkage</label>
            <input id="lda_shrinkage" name="lda_shrinkage" value="0.0" />
          </div>
        </div>
        <div class="grid2 train-field train-multi-field">
          <div>
            <label for="balance_classes">Class balancing</label>
            <select id="balance_classes" name="balance_classes">
              <option value="none" selected>none</option>
              <option value="downsample">downsample</option>
            </select>
          </div>
          <div>
            <label for="seed">Seed</label>
            <input id="seed" name="seed" value="42" />
          </div>
        </div>
        <div class="train-multi-field">
          <label for="balance_subjects">Subject balancing</label>
          <select id="balance_subjects" name="balance_subjects">
            <option value="none" selected>none</option>
            <option value="downsample">downsample</option>
          </select>
        </div>

        <div class="section-title validate-field predict-field">Timeline options</div>
        <div class="grid2 validate-field predict-field">
          <div class="validate-field">
            <label for="event_overlap_threshold">Event overlap threshold</label>
            <input id="event_overlap_threshold" name="event_overlap_threshold" value="0.5" />
          </div>
          <div>
            <label for="smoothing_window">Smoothing window</label>
            <select id="smoothing_window" name="smoothing_window">
              <option value="1">1 - no smoothing</option>
              <option value="3" selected>3</option>
              <option value="5">5</option>
            </select>
          </div>
        </div>

        <button id="runButton" type="submit">Start pipeline</button>
        <div class="footer-note">This interface starts a local process on your machine. It does not upload EEG data anywhere.</div>
      </form>
    </section>

    <section class="card result-card">
      <h2>Run result</h2>
      <div id="statusBox" class="status"><span id="statusDot" class="dot"></span><span id="statusText">Ready.</span></div>
      <div id="commandBox" class="command hidden"></div>
      <div id="notes"></div>
      <div id="chartWrap" class="chart-wrap hidden">
        <div class="chart-title">Percentage overview</div>
        <canvas id="percentChart" width="920" height="420"></canvas>
      </div>
      <div id="metrics" class="metrics"></div>
      <div id="files" class="files"></div>
    </section>
  </main>

<script>
const mode = document.getElementById('mode');
const form = document.getElementById('runForm');
const button = document.getElementById('runButton');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const commandBox = document.getElementById('commandBox');
const metricsBox = document.getElementById('metrics');
const filesBox = document.getElementById('files');
const notesBox = document.getElementById('notes');
const chartWrap = document.getElementById('chartWrap');
const chart = document.getElementById('percentChart');

const notes = {
  train: 'Train one model from one labeled EDF recording and its event file.',
  train_multi: 'Train one model from a dataset-list CSV. Each recording is processed independently; only epoched command windows are pooled.',
  validate_timeline: 'Run the saved model on a full EDF timeline and compare it with events when an events file is provided.',
  predict: 'Run the saved model on an unlabeled EDF timeline. No accuracy is computed without ground truth.'
};

function setStatus(kind, text) {
  statusDot.className = 'dot ' + (kind || '');
  statusText.textContent = text;
}

function showByMode() {
  const m = mode.value;
  document.getElementById('modeNote').textContent = notes[m];
  const all = document.querySelectorAll('.train-field, .train-multi-field, .validate-field, .predict-field, .field-edf, .field-events, .field-model, .field-dataset-list, .field-dataset-dir');
  all.forEach(el => el.classList.add('hidden'));

  if (m === 'train') {
    document.querySelectorAll('.train-field, .field-edf, .field-events').forEach(el => el.classList.remove('hidden'));
  } else if (m === 'train_multi') {
    document.querySelectorAll('.train-multi-field, .field-dataset-list, .field-dataset-dir').forEach(el => el.classList.remove('hidden'));
  } else if (m === 'validate_timeline') {
    document.querySelectorAll('.validate-field, .field-edf, .field-events, .field-model').forEach(el => el.classList.remove('hidden'));
  } else if (m === 'predict') {
    document.querySelectorAll('.predict-field, .field-edf, .field-model').forEach(el => el.classList.remove('hidden'));
  }
}

function formPayload() {
  const data = new FormData(form);
  const obj = {};
  for (const [k, v] of data.entries()) obj[k] = v;
  return obj;
}

function renderBars(items) {
  metricsBox.innerHTML = '';
  if (!items || !items.length) {
    metricsBox.innerHTML = '<div class="footer-note">No percentage metrics were found yet. Training runs usually produce model files first; validation or prediction runs produce percentage summaries.</div>';
    return;
  }
  for (const item of items) {
    const row = document.createElement('div');
    row.className = 'metric-row';
    const value = Number(item.value || 0);
    row.innerHTML = `
      <div class="metric-head"><div><div class="metric-label">${escapeHtml(item.label)}</div><div class="hint">${escapeHtml(item.group || 'Metric')}</div></div><div class="metric-value">${value.toFixed(2)}%</div></div>
      <div class="bar-bg"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, value))}%"></div></div>`;
    metricsBox.appendChild(row);
  }
}

function renderFiles(files) {
  filesBox.innerHTML = '';
  if (!files || !files.length) return;
  const title = document.createElement('div');
  title.className = 'section-title';
  title.textContent = 'Generated files';
  filesBox.appendChild(title);
  for (const f of files) {
    const pill = document.createElement('div');
    pill.className = 'file-pill';
    pill.textContent = `${f.name}: ${f.path}`;
    filesBox.appendChild(pill);
  }
}

function renderNotes(notes) {
  notesBox.innerHTML = '';
  if (!notes || !notes.length) return;
  for (const n of notes) {
    const div = document.createElement('div');
    div.className = 'note';
    div.textContent = n;
    notesBox.appendChild(div);
  }
}

function drawChart(items) {
  const rows = (items || []).slice(0, 10);
  if (!rows.length) {
    chartWrap.classList.add('hidden');
    return;
  }
  chartWrap.classList.remove('hidden');
  const ctx = chart.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = chart.clientWidth || 920;
  const cssHeight = Math.max(280, rows.length * 46 + 70);
  chart.width = cssWidth * dpr;
  chart.height = cssHeight * dpr;
  chart.style.height = cssHeight + 'px';
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const padLeft = 230;
  const padRight = 38;
  const padTop = 36;
  const barH = 18;
  const gap = 26;
  const width = cssWidth - padLeft - padRight;

  ctx.fillStyle = '#cfe2ff';
  ctx.font = '700 15px system-ui, sans-serif';
  ctx.fillText('Percentage metrics', 18, 22);

  ctx.strokeStyle = '#263143';
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 100; tick += 25) {
    const x = padLeft + width * tick / 100;
    ctx.beginPath();
    ctx.moveTo(x, padTop - 8);
    ctx.lineTo(x, cssHeight - 22);
    ctx.stroke();
    ctx.fillStyle = '#9fb0c7';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(tick + '%', x - 12, cssHeight - 6);
  }

  rows.forEach((item, i) => {
    const y = padTop + i * (barH + gap);
    const val = Math.max(0, Math.min(100, Number(item.value || 0)));
    ctx.fillStyle = '#9fb0c7';
    ctx.font = '12px system-ui, sans-serif';
    const label = String(item.label || '').slice(0, 32);
    ctx.fillText(label, 18, y + 14);
    ctx.fillStyle = '#0e1420';
    roundRect(ctx, padLeft, y, width, barH, 9, true);
    const grad = ctx.createLinearGradient(padLeft, y, padLeft + width, y);
    grad.addColorStop(0, '#6aa9ff');
    grad.addColorStop(1, '#9bd67a');
    ctx.fillStyle = grad;
    roundRect(ctx, padLeft, y, width * val / 100, barH, 9, true);
    ctx.fillStyle = '#eef4ff';
    ctx.font = '700 12px system-ui, sans-serif';
    ctx.fillText(val.toFixed(1) + '%', padLeft + width * val / 100 + 8, y + 14);
  });
}

function roundRect(ctx, x, y, w, h, r, fill) {
  const rr = Math.min(r, Math.abs(w) / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
  if (fill) ctx.fill();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  button.disabled = true;
  commandBox.classList.add('hidden');
  metricsBox.innerHTML = '';
  filesBox.innerHTML = '';
  notesBox.innerHTML = '';
  chartWrap.classList.add('hidden');
  setStatus('running', 'Running pipeline. Keep this browser tab open.');
  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formPayload())
    });
    const data = await response.json();
    if (data.command) {
      commandBox.textContent = data.command;
      commandBox.classList.remove('hidden');
    }
    if (!response.ok || !data.ok) {
      setStatus('fail', data.message || 'Run failed.');
      renderNotes(data.errors || []);
      return;
    }
    setStatus('ok', `Completed in ${data.elapsed_seconds.toFixed(1)} seconds.`);
    const allBars = [];
    if (data.metrics) {
      allBars.push(...(data.metrics.percentage_bars || []));
      allBars.push(...(data.metrics.prediction_shares || []));
      renderNotes(data.metrics.notes || []);
      renderFiles(data.metrics.files || []);
    }
    renderBars(allBars);
    drawChart(allBars);
  } catch (err) {
    setStatus('fail', 'The web UI could not complete the request.');
    renderNotes([String(err)]);
  } finally {
    button.disabled = false;
  }
});

mode.addEventListener('change', showByMode);
showByMode();
</script>
</body>
</html>
"""


class BCIWebHandler(BaseHTTPRequestHandler):
    server_version = "BCIWebUI/1.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            page = INDEX_HTML.replace("__PIPELINE_PATH__", html.escape(guess_pipeline_path(), quote=True))
            html_response(self, page)
            return
        if path == "/api/defaults":
            payload = dict(DEFAULTS)
            payload["pipeline_path"] = guess_pipeline_path()
            json_response(self, payload)
            return
        json_response(self, {"ok": False, "message": "Not found."}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/run":
            json_response(self, {"ok": False, "message": "Not found."}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw) if raw else {}
        except Exception:
            json_response(self, {"ok": False, "message": "Invalid JSON request."}, status=400)
            return

        cmd, errors, cwd = build_command(payload)
        if errors or cmd is None:
            json_response(self, {"ok": False, "message": "Please fix the highlighted settings.", "errors": errors}, status=400)
            return

        output_dir = clean_string(payload.get("output_dir"))
        printable_command = " ".join(shlex.quote(part) for part in cmd)
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                errors="replace",
            )
        except Exception as exc:
            json_response(
                self,
                {
                    "ok": False,
                    "message": "The pipeline process could not be started.",
                    "errors": [str(exc)],
                    "command": printable_command,
                },
                status=500,
            )
            return

        elapsed = time.time() - started

        # Keep the browser UI English-only. Raw output from the original pipeline is printed
        # to this server terminal for debugging, because that script may contain non-English messages.
        if proc.stdout:
            print("\n--- Pipeline stdout ---\n" + proc.stdout)
        if proc.stderr:
            print("\n--- Pipeline stderr ---\n" + proc.stderr, file=sys.stderr)

        expect_timeline_metrics = (
            clean_string(payload.get("mode")) == "validate_timeline"
            and bool(clean_string(payload.get("events")))
        )
        metrics = collect_metrics(
            output_dir,
            cwd=cwd,
            expect_timeline_metrics=expect_timeline_metrics,
        )
        if proc.returncode != 0:
            json_response(
                self,
                {
                    "ok": False,
                    "message": f"Pipeline exited with code {proc.returncode}. Check the terminal running this web UI for the original pipeline output.",
                    "errors": ["No pipeline logic was changed by the web UI."],
                    "command": printable_command,
                    "elapsed_seconds": elapsed,
                    "metrics": metrics,
                },
                status=500,
            )
            return

        json_response(
            self,
            {
                "ok": True,
                "message": "Pipeline completed successfully.",
                "command": printable_command,
                "elapsed_seconds": elapsed,
                "metrics": metrics,
            },
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    host = os.environ.get("BCI_WEB_UI_HOST", HOST)
    port = int(os.environ.get("BCI_WEB_UI_PORT", str(PORT)))
    server = ThreadingHTTPServer((host, port), BCIWebHandler)
    print("BCI Pipeline Web UI")
    print(f"Open http://{host}:{port} in your browser.")
    print(f"Default pipeline script: {guess_pipeline_path()}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
