"""
FileReplaySource: replay a recorded EDF as if it were arriving live.

Moved and renamed from fast_causal_bci.RecordedReplaySource (Phase B naming)
- the chunk-emission logic in `chunks()` is unchanged from the original,
already-tested implementation: absolute scheduling deadlines are used
instead of a fixed per-chunk sleep, so processing time and scheduling jitter
don't accumulate into growing drift over a long replay.

fast_causal_bci.py keeps a `RecordedReplaySource = FileReplaySource` alias
so its own CLI (`main()`, `run_decision_source()`) keeps working unchanged;
that file's filter/feature/model/LOSO core was not touched by this move.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np


class FileReplaySource:
    """Expose an in-memory EDF recording as device-like sample chunks."""

    is_live = False

    def __init__(self, signals, channels, fs, realtime_pace=False,
                 max_seconds=None, clock=time.monotonic, sleeper=time.sleep):
        self.signals = signals
        self.channels = list(channels)
        self.sampling_rate = float(fs)
        self.realtime_pace = bool(realtime_pace)
        self.max_seconds = max_seconds
        self.clock = clock
        self.sleeper = sleeper
        self.last_scheduled_at = None
        self.last_emitted_at = None
        self.last_lateness_ms = None

    @property
    def channel_names(self) -> Sequence[str]:
        return self.channels

    def chunks(self, chunk_samples):
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        n = min(len(self.signals[ch]) for ch in self.channels)
        if self.max_seconds is not None:
            n = min(n, int(round(self.max_seconds * self.sampling_rate)))
        deadline = self.clock()
        for start in range(0, n, chunk_samples):
            end = min(start + chunk_samples, n)
            if self.realtime_pace:
                # Absolute deadlines avoid accumulating processing/sleep drift.
                deadline += (end - start) / self.sampling_rate
                remaining = deadline - self.clock()
                if remaining > 0:
                    self.sleeper(remaining)
            emitted = self.clock()
            self.last_scheduled_at = deadline if self.realtime_pace else emitted
            self.last_emitted_at = emitted
            self.last_lateness_ms = max(0.0, (emitted - self.last_scheduled_at) * 1000)
            yield {ch: np.asarray(self.signals[ch][start:end]) for ch in self.channels}

    def close(self) -> None:
        """No resources to release for an in-memory replay source."""
        return None

    def __enter__(self) -> "FileReplaySource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
