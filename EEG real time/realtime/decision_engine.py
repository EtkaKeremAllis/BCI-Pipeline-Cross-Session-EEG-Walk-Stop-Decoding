"""
DecisionEngine: the online inference loop (Phase B slice 4).

Consumes chunks from an EEGSource, pushes them through fast_causal_bci's
existing, untouched FeatureStream + FastCausalModel, and runs each decision
through the same three-step sequence the offline pipeline's
predict_window() uses - predict -> confidence gate -> temporal smoothing -
adapted to fast_causal_bci's simpler binary STOP/WALK model (it has no
z-distance IDLE gate the way the offline DeployableBCIModel does; only a
prediction confidence is available, so that's what the gate here uses).

Per-decision timing (feature extraction, classification, end-to-end) is
recorded the same way fast_causal_bci.run_decision_source() already does,
so latency behavior is directly comparable to that existing, documented
baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

from fast_causal_bci import FastCausalModel, FeatureStream
from realtime.eeg_source import EEGSource
from realtime.online_smoothing import OnlineSmoother

IDLE_LABEL = 2


@dataclass
class Decision:
    stream_time_s: float
    raw_label: int
    confidence: float
    # None until OnlineSmoother has seen enough future context to emit a
    # smoothed label for this position (see online_smoothing.py); collect
    # DecisionEngine.flush() after the source is exhausted for the tail.
    smoothed_label: Optional[int]
    feature_ms: float
    decision_ms: float
    end_to_end_ms: float


class DecisionEngine:
    def __init__(self, model: FastCausalModel, smoothing_window: int = 3,
                 confidence_threshold: float = 0.0, clock=time.perf_counter):
        """
        confidence_threshold: predictions with confidence below this are
        gated to IDLE before smoothing, mirroring the offline pipeline's
        confidence gate. Defaults to 0.0 (gate effectively off - every
        binary-classifier prediction has confidence >= 0.5), since
        fast_causal_bci.py has no prior tuning for this value; pick one
        empirically for your use case rather than trusting a made-up
        default.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.smoother = OnlineSmoother(smoothing_window)
        self.clock = clock
        self.stream = FeatureStream(model.channels, model.fs, model.window_seconds,
                                     model.context_seconds)
        self.chunk_samples = max(1, int(round(model.step_seconds * model.fs)))

    def run(self, source: EEGSource) -> Iterator[Decision]:
        """Consume the full source, yielding one Decision per ready feature window."""
        if list(source.channels) != list(self.model.channels):
            raise ValueError(
                f"Channel/order mismatch: source={source.channels}, model={self.model.channels}"
            )
        if not (source.sampling_rate == self.model.fs):
            raise ValueError(
                f"Sampling-rate mismatch: source={source.sampling_rate}, model={self.model.fs}"
            )

        for chunk in source.chunks(self.chunk_samples):
            arrived = self.clock()
            self.stream.push(chunk)
            if not self.stream.ready:
                continue

            features = self.stream.features()
            features_ready = self.clock()

            pred, proba = self.model.predict_features(features.reshape(1, -1))
            raw_label = int(pred[0])
            confidence = float(proba[0, raw_label])
            if confidence < self.confidence_threshold:
                raw_label = IDLE_LABEL
            decided = self.clock()

            ready = self.smoother.push(raw_label)
            smoothed_label = ready[-1] if ready else None

            yield Decision(
                stream_time_s=self.stream.samples_seen / self.model.fs,
                raw_label=raw_label,
                confidence=confidence,
                smoothed_label=smoothed_label,
                feature_ms=(features_ready - arrived) * 1000,
                decision_ms=(decided - features_ready) * 1000,
                end_to_end_ms=(decided - arrived) * 1000,
            )

    def flush(self) -> list[int]:
        """Drain any pending smoothed labels once the source is exhausted."""
        return self.smoother.flush()
