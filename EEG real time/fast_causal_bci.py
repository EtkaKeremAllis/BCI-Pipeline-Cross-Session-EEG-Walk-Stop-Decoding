"""Low-latency causal WALK/STOP classifier using stateful scipy.signal.lfilter.

No future samples, filtfilt, centered smoothing, or recording-wide z-score are
used. The exact same causal filters are used for offline training and live
chunks. This is an experimental research classifier, not a safety controller.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Protocol, Sequence

import numpy as np
from scipy import signal

# edf_reader/parse_events/modern_bci_v2 live at the repo root, shared with
# the offline pipeline - not duplicated here, to avoid the two copies drifting.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edf_reader import read_edf
from parse_events import parse_events
from modern_bci_v2 import SimpleLDA
from realtime.file_replay_source import FileReplaySource


LABELS = {0: "STOP", 1: "WALK", 2: "IDLE"}


class ChunkSource(Protocol):
    """Source boundary shared by recorded replay and a future EEG device."""

    sampling_rate: float
    channels: Sequence[str]

    def chunks(self, chunk_samples: int) -> Iterator[Mapping[str, np.ndarray]]:
        ...


# Moved to realtime/file_replay_source.py (Phase B naming) and re-exported
# here under the original name so this module's own CLI (main(),
# run_decision_source()) keeps working unchanged. The chunk-emission logic
# itself was not touched by the move.
RecordedReplaySource = FileReplaySource


class LiveDeviceSource:
    """Adapter for a blocking device SDK call returning channel->samples.

    ``driver.read(chunk_samples)`` must return equally-sized arrays for every
    requested model channel. Returning None ends the stream cleanly.
    """

    def __init__(self, driver, channels, fs):
        self.driver = driver
        self.channels = list(channels)
        self.sampling_rate = float(fs)

    def chunks(self, chunk_samples):
        while True:
            chunk = self.driver.read(chunk_samples)
            if chunk is None:
                return
            yield chunk


class CausalFilterBank:
    """Stateful low-order IIR filters executed through lfilter."""

    BANDS = {"broad": (1.0, 40.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0)}

    def __init__(self, channels: Sequence[str], fs: float, order: int = 2):
        self.channels = list(channels)
        self.fs = float(fs)
        self.coefficients = {}
        self.state = {}
        for band, limits in self.BANDS.items():
            b, a = signal.butter(order, limits, btype="bandpass", fs=fs)
            self.coefficients[band] = (b, a)
            for ch in self.channels:
                self.state[(band, ch)] = np.zeros(max(len(a), len(b)) - 1)

    def process(self, chunk: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        output = {band: {} for band in self.BANDS}
        for band, (b, a) in self.coefficients.items():
            for ch in self.channels:
                values = np.asarray(chunk[ch], dtype=np.float64).reshape(-1)
                filtered, final_state = signal.lfilter(
                    b, a, values, zi=self.state[(band, ch)]
                )
                self.state[(band, ch)] = final_state
                output[band][ch] = filtered
        return output


class FeatureStream:
    def __init__(self, channels: Sequence[str], fs: float, window_seconds: float,
                 context_seconds: float | None = None):
        self.channels = list(channels)
        self.fs = float(fs)
        self.window_samples = max(5, int(round(window_seconds * fs)))
        self.context_seconds = max(window_seconds, context_seconds or window_seconds)
        self.history_samples = max(self.window_samples, int(round(self.context_seconds * fs)))
        self.bank = CausalFilterBank(channels, fs)
        self.buffers = {
            (band, ch): deque(maxlen=self.history_samples)
            for band in CausalFilterBank.BANDS for ch in channels
        }
        self.samples_seen = 0

    def push(self, chunk: Mapping[str, np.ndarray]) -> None:
        lengths = {len(np.asarray(chunk[ch]).reshape(-1)) for ch in self.channels}
        if len(lengths) != 1:
            raise ValueError("All channels must have equal chunk lengths")
        filtered = self.bank.process(chunk)
        for band in filtered:
            for ch in self.channels:
                self.buffers[(band, ch)].extend(filtered[band][ch])
        self.samples_seen += next(iter(lengths))

    @property
    def ready(self) -> bool:
        return all(len(buf) == self.history_samples for buf in self.buffers.values())

    def features(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Feature window is not full")
        result = []
        eps = 1e-10
        # One immutable snapshot per buffer for this feature call. Converting a
        # deque to ndarray copies its contents, so doing it inside every scale
        # loop multiplied the live decision cost without adding information.
        arrays = {
            key: np.asarray(values, dtype=np.float64)
            for key, values in self.buffers.items()
        }
        scale_samples = sorted(set([
            self.window_samples,
            min(self.history_samples, max(self.window_samples, int(round(0.5 * self.fs)))),
            self.history_samples,
        ]))
        energy_by_scale = {}
        for scale in scale_samples:
            band_energy = {}
            for ch in self.channels:
                broad = arrays[("broad", ch)][-scale:]
                mu = arrays[("mu", ch)][-scale:]
                beta = arrays[("beta", ch)][-scale:]
                eb = np.mean(broad * broad) + eps
                em = np.mean(mu * mu) + eps
                et = np.mean(beta * beta) + eps
                band_energy[ch] = (em, et)
                result.extend([
                    np.log(eb), np.log(em), np.log(et), np.log(em / et),
                    np.mean(np.abs(np.diff(broad))) + eps,
                    np.std(np.diff(broad)) + eps, np.ptp(broad),
                ])
            energy_by_scale[scale] = band_energy
            for left, right in zip(self.channels[:-1], self.channels[1:]):
                lm, lb = band_energy[left]; rm, rb = band_energy[right]
                result.extend([np.log(lm / rm), np.log(lb / rb)])

        # Explicit transition features: recent short window versus immediately
        # preceding short window. These can react within window_seconds even
        # while the longer context still contains the previous state.
        if self.history_samples >= 2 * self.window_samples:
            for band in ("broad", "mu", "beta"):
                for ch in self.channels:
                    values = arrays[(band, ch)]
                    recent = np.mean(values[-self.window_samples:] ** 2) + eps
                    previous = np.mean(values[-2*self.window_samples:-self.window_samples] ** 2) + eps
                    result.extend([np.log(recent / previous), (recent - previous) / previous])
        # Explicit multi-scale spatial structure. Each scale is a separate
        # feature group: the fast decision-window covariance can react quickly,
        # while longer covariance describes the stable spatial trend. This is
        # not label leakage: training already requires the complete context to
        # belong to one state; during live transitions old context is expected.
        correlation_by_scale = {}
        for scale in scale_samples:
            broad_matrix = np.vstack([
                arrays[("broad", ch)][-scale:]
                for ch in self.channels
            ])
            channel_std = broad_matrix.std(axis=1) + eps
            total_energy = np.mean(broad_matrix * broad_matrix) + eps
            scale_correlations = []
            for i in range(len(self.channels)):
                for j in range(i + 1, len(self.channels)):
                    centered_i = broad_matrix[i] - broad_matrix[i].mean()
                    centered_j = broad_matrix[j] - broad_matrix[j].mean()
                    covariance = np.mean(centered_i * centered_j)
                    correlation = covariance / (channel_std[i] * channel_std[j])
                    scale_correlations.append(correlation)
                    result.extend([
                        correlation,
                        covariance / total_energy,
                        np.log((channel_std[i] ** 2 + eps) /
                               (channel_std[j] ** 2 + eps)),
                    ])
            correlation_by_scale[scale] = np.asarray(scale_correlations)

        if len(scale_samples) > 1:
            short_corr = correlation_by_scale[scale_samples[0]]
            long_corr = correlation_by_scale[scale_samples[-1]]
            result.extend((short_corr - long_corr).tolist())
        return np.asarray(result, dtype=np.float64)


def event_label_at(t: float, events, label_map=None):
    label_map = label_map or {"x5": 0, "x8": 1}
    for onset, duration, kind in events:
        if onset <= t < onset + duration and kind in label_map:
            return label_map[kind]
    return None


def extract_causal_features_from_source(source: ChunkSource, events, window_seconds,
                                        step_seconds, training=False,
                                        context_seconds=None):
    channels = list(source.channels)
    fs = float(source.sampling_rate)
    stream = FeatureStream(channels, fs, window_seconds, context_seconds)
    chunk_samples = max(1, int(round(step_seconds * fs)))
    X, y, times = [], [], []
    for chunk in source.chunks(chunk_samples):
        stream.push(chunk)
        if not stream.ready:
            continue
        t = stream.samples_seen / fs
        label = event_label_at(t, events)
        # Training requires the COMPLETE feature context to belong to one
        # state, preventing a STOP target from silently containing WALK context
        # (and vice versa). During validation/live transition measurement only
        # the fast decision window must be in the new state; retaining older
        # context there is intentional and measures real transition behavior.
        label_span = stream.context_seconds if training else window_seconds
        start_label = event_label_at(t - label_span + 1 / fs, events)
        if label is None or start_label != label:
            continue
        X.append(stream.features()); y.append(label); times.append(t)
    return np.asarray(X), np.asarray(y, dtype=int), np.asarray(times)


def extract_causal_features(signals, channels, fs, events, window_seconds, step_seconds,
                            training=False, context_seconds=None):
    """Backward-compatible recorded-data wrapper around ChunkSource."""
    source = RecordedReplaySource(signals, channels, fs, realtime_pace=False)
    return extract_causal_features_from_source(
        source, events, window_seconds, step_seconds, training, context_seconds
    )


@dataclass
class FastCausalModel:
    channels: list[str]
    fs: float
    window_seconds: float
    step_seconds: float
    context_seconds: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    selected_idx: np.ndarray
    lda: SimpleLDA

    @classmethod
    def train(cls, signals, channels, fs, events, window_seconds=0.2,
              step_seconds=0.05, n_features=24, shrinkage=0.2, context_seconds=None):
        context_seconds = max(window_seconds, context_seconds or window_seconds)
        # Non-overlapping training windows reduce duplicate-sample weighting.
        X, y, _ = extract_causal_features(
            signals, channels, fs, events, window_seconds, window_seconds, training=True,
            context_seconds=context_seconds
        )
        scores = (X[y == 1].mean(0) - X[y == 0].mean(0)) ** 2 / (X.var(0) + 1e-9)
        selected = np.argsort(scores)[::-1][:min(n_features, X.shape[1])]
        X = X[:, selected]
        mean, std = X.mean(0), X.std(0) + 1e-8
        lda = SimpleLDA(shrinkage=shrinkage); lda.fit((X - mean) / std, y)
        return cls(list(channels), fs, window_seconds, step_seconds, context_seconds,
                   mean, std, selected, lda)

    def predict_features(self, X):
        normalized = (X[:, self.selected_idx] - self.feature_mean) / self.feature_std
        return self.lda.predict(normalized), self.lda.predict_proba(normalized)

    def save(self, directory):
        path = Path(directory); path.mkdir(parents=True, exist_ok=True)
        (path / "model_info.json").write_text(json.dumps({
            "channels": self.channels, "sampling_rate": self.fs,
            "window_seconds": self.window_seconds, "step_seconds": self.step_seconds,
            "context_seconds": self.context_seconds,
            "causal": True, "filter": "butterworth-order2-scipy-lfilter",
        }, indent=2))
        np.savez(path / "fast_causal_model.npz", feature_mean=self.feature_mean,
                 feature_std=self.feature_std, selected_idx=self.selected_idx,
                 lda_coef=self.lda.coef, lda_intercept=self.lda.intercept,
                 lda_mean_0=self.lda.mean_0, lda_mean_1=self.lda.mean_1)

    @classmethod
    def load(cls, directory):
        path = Path(directory); info = json.loads((path / "model_info.json").read_text())
        data = np.load(path / "fast_causal_model.npz")
        lda = SimpleLDA(shrinkage=0.2)
        lda.coef, lda.intercept = data["lda_coef"], data["lda_intercept"]
        lda.mean_0, lda.mean_1 = data["lda_mean_0"], data["lda_mean_1"]
        return cls(info["channels"], info["sampling_rate"], info["window_seconds"],
                   info["step_seconds"], info.get("context_seconds", info["window_seconds"]),
                   data["feature_mean"], data["feature_std"],
                   data["selected_idx"], lda)


def load_recording(edf_path, channels=None):
    signals, info = read_edf(edf_path)
    channels = list(channels or [ch for ch in ("C3", "Cz", "C4") if ch in signals])
    rates = [info["sampling_rate"][ch] for ch in channels]
    if not np.allclose(rates, rates[0]):
        raise ValueError("Channel sampling rates differ")
    return signals, channels, float(rates[0])


def validate(model, edf_path, events_path):
    signals, channels, fs = load_recording(edf_path, model.channels)
    events = parse_events(events_path)
    started = time.perf_counter()
    X, y, times = extract_causal_features(
        signals, channels, fs, events, model.window_seconds, model.step_seconds,
        context_seconds=model.context_seconds
    )
    pred, proba = model.predict_features(X)
    elapsed = time.perf_counter() - started
    recalls = [np.mean(pred[y == label] == label) for label in (0, 1)]
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.mean(recalls)),
        "stop_recall": float(recalls[0]), "walk_recall": float(recalls[1]),
        "n_windows": len(y), "processing_ms_per_decision": elapsed * 1000 / len(y),
        "predictions": pred, "truth": y, "times": times,
    }


def run_decision_source(model, source: ChunkSource, output_csv=None, events=None):
    """Run one causal decision loop for either replay or a live EEG device."""
    if list(source.channels) != list(model.channels):
        raise ValueError(f"Channel/order mismatch: source={source.channels}, model={model.channels}")
    if not np.isclose(source.sampling_rate, model.fs):
        raise ValueError(f"Sampling-rate mismatch: source={source.sampling_rate}, model={model.fs}")
    stream = FeatureStream(model.channels, model.fs, model.window_seconds,
                           model.context_seconds)
    chunk_samples = max(1, int(round(model.step_seconds * model.fs)))
    records = []
    for chunk in source.chunks(chunk_samples):
        arrived = time.perf_counter()
        stream.push(chunk)
        if not stream.ready:
            continue
        features = stream.features()
        features_ready = time.perf_counter()
        pred, proba = model.predict_features(features.reshape(1, -1))
        decided = time.perf_counter()
        stream_time = stream.samples_seen / model.fs
        truth = event_label_at(stream_time, events) if events else None
        records.append({
            "stream_time_s": stream_time,
            "prediction": LABELS[int(pred[0])],
            "confidence": float(proba[0, int(pred[0])]),
            "truth": "" if truth is None else LABELS[int(truth)],
            "feature_ms": (features_ready - arrived) * 1000,
            "decision_ms": (decided - features_ready) * 1000,
            "end_to_end_ms": (decided - arrived) * 1000,
            "source_lateness_ms": getattr(source, "last_lateness_ms", np.nan),
        })
    if output_csv:
        path = Path(output_csv); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys() if records else [
                "stream_time_s", "prediction", "confidence", "truth", "feature_ms",
                "decision_ms", "end_to_end_ms", "source_lateness_ms"
            ])
            writer.writeheader(); writer.writerows(records)
    if not records:
        return {"n_decisions": 0}
    summary = {"n_decisions": len(records)}
    for field in ("feature_ms", "decision_ms", "end_to_end_ms", "source_lateness_ms"):
        values = np.asarray([row[field] for row in records], dtype=float)
        values = values[np.isfinite(values)]
        summary[field] = {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
    if events:
        valid = [row for row in records if row["truth"]]
        summary["accuracy"] = float(np.mean([
            row["prediction"] == row["truth"] for row in valid
        ])) if valid else None
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "validate", "replay"], required=True)
    parser.add_argument("--edf", required=True); parser.add_argument("--events", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--window", type=float, default=0.2)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--context", type=float, default=None)
    parser.add_argument("--output", default="replay_timing.csv")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    if args.mode == "train":
        if not args.events:
            raise SystemExit("--events is required for training")
        signals, channels, fs = load_recording(args.edf)
        model = FastCausalModel.train(signals, channels, fs, parse_events(args.events),
                                      args.window, args.step, context_seconds=args.context)
        model.save(args.model); print(args.model)
    elif args.mode == "validate":
        if not args.events:
            raise SystemExit("--events is required for validation")
        result = validate(FastCausalModel.load(args.model), args.edf, args.events)
        printable = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        print(json.dumps(printable, indent=2))
    else:
        model = FastCausalModel.load(args.model)
        signals, channels, fs = load_recording(args.edf, model.channels)
        source = RecordedReplaySource(signals, channels, fs, realtime_pace=True,
                                      max_seconds=args.max_seconds)
        events = parse_events(args.events) if args.events else None
        summary = run_decision_source(model, source, args.output, events)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
