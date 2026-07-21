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

Pending-record re-pairing and stop_check were adopted from Kerem's
independent extension of run_decision_source() itself (branch
codex/callback-smoothing) - ported here, onto this module, rather than into
fast_causal_bci.py, so the "don't touch the core" boundary holds even as
both sides keep improving the decision loop.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

from fast_causal_bci import FastCausalModel, FeatureStream
from realtime.eeg_source import EEGSource
from realtime.online_smoothing import OnlineSmoother

IDLE_LABEL = 2


@dataclass
class Decision:
    stream_time_s: float
    raw_label: int
    confidence: float
    smoothed_label: int
    feature_ms: float
    decision_ms: float
    end_to_end_ms: float
    # Extra wall-clock time between this decision's raw prediction and the
    # moment its smoothed counterpart became available (>= 0; 0 for
    # smoothing_window=1, where every decision resolves immediately).
    smoothing_wall_delay_ms: float
    # end_to_end_ms + smoothing_wall_delay_ms: the full processing latency
    # a live consumer (e.g. a UI) actually experiences for this decision.
    total_latency_ms: float


@dataclass
class _PendingRecord:
    stream_time_s: float
    raw_label: int
    confidence: float
    feature_ms: float
    decision_ms: float
    end_to_end_ms: float
    arrived: float


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
        self._pending: "deque[_PendingRecord]" = deque()

    def run(self, source: EEGSource,
            stop_check: Optional[Callable[[], bool]] = None) -> Iterator[Decision]:
        """
        Consume the source, yielding one fully-resolved Decision per raw
        prediction that has been smoothed - i.e. each yielded Decision's
        raw_label/confidence/stream_time_s and its smoothed_label always
        refer to the SAME point in the stream (see online_smoothing.py for
        why smoothing has to lag behind the raw predictions by
        smoothing_window//2 steps). Call flush() after the source is
        exhausted for the trailing predictions still pending resolution.

        stop_check: polled once per chunk; return True to end the loop
        early (e.g. a UI "Stop" button), without needing to close the
        source itself.
        """
        if list(source.channels) != list(self.model.channels):
            raise ValueError(
                f"Channel/order mismatch: source={source.channels}, model={self.model.channels}"
            )
        if not (source.sampling_rate == self.model.fs):
            raise ValueError(
                f"Sampling-rate mismatch: source={source.sampling_rate}, model={self.model.fs}"
            )

        for chunk in source.chunks(self.chunk_samples):
            if stop_check and stop_check():
                break

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

            self._pending.append(_PendingRecord(
                stream_time_s=self.stream.samples_seen / self.model.fs,
                raw_label=raw_label,
                confidence=confidence,
                feature_ms=(features_ready - arrived) * 1000,
                decision_ms=(decided - features_ready) * 1000,
                end_to_end_ms=(decided - arrived) * 1000,
                arrived=arrived,
            ))

            for smoothed_label in self.smoother.push(raw_label):
                yield self._resolve(smoothed_label)

    def flush(self) -> List[Decision]:
        """Resolve every remaining pending record once the source is exhausted."""
        return [self._resolve(label) for label in self.smoother.flush()]

    def _resolve(self, smoothed_label: int) -> Decision:
        record = self._pending.popleft()
        emitted = self.clock()
        smoothing_wall_delay_ms = max(0.0, (emitted - record.arrived) * 1000 - record.end_to_end_ms)
        return Decision(
            stream_time_s=record.stream_time_s,
            raw_label=record.raw_label,
            confidence=record.confidence,
            smoothed_label=smoothed_label,
            feature_ms=record.feature_ms,
            decision_ms=record.decision_ms,
            end_to_end_ms=record.end_to_end_ms,
            smoothing_wall_delay_ms=smoothing_wall_delay_ms,
            total_latency_ms=record.end_to_end_ms + smoothing_wall_delay_ms,
        )
