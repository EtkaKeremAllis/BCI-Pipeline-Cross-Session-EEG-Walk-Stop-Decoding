"""
OnlineSmoother: a causal, delayed counterpart to the offline pipeline's
apply_temporal_smoothing() (centered majority vote, tie keeps the raw
prediction, no WALK priority).

The offline function is centered: smoothed[i] depends on raw labels both
before AND after index i, which a live stream can't provide (the future
hasn't arrived yet). Rather than inventing a different, weaker algorithm
just to be "causal", OnlineSmoother delays each decision by window_size//2
chunks - just long enough to have seen the same "future" context the
offline centered window uses - then applies the exact same rule. The result
is byte-for-byte identical to apply_temporal_smoothing() given the same
input (see EEG real time/tests/test_online_smoothing.py), at the cost of
that small, fixed extra latency (`window_size//2 * step_seconds`: 50 ms for
window 3 or 100 ms for window 5 at this pipeline's 50 ms decision step).

Keeps the full raw-label history for the current session. Sessions are
minutes long, not days, so this is a deliberate simplicity/memory tradeoff,
not an oversight - a bounded ring buffer would add real complexity for no
benefit at this scale.
"""
from __future__ import annotations


class OnlineSmoother:
    def __init__(self, window_size: int):
        if window_size not in (1, 3, 5):
            raise ValueError(f"window_size must be 1, 3, or 5, got {window_size!r}")
        self.window_size = window_size
        self.half = window_size // 2
        self._raw: list[int] = []
        self._next_emit_idx = 0

    def push(self, raw_label: int) -> list[int]:
        """Add one new raw prediction. Returns the (usually 0 or 1) smoothed
        labels that became ready to emit as a result - a list because the
        very first push() after construction can make more than one ready
        when window_size == 1."""
        self._raw.append(int(raw_label))
        return self._drain_ready()

    def flush(self) -> list[int]:
        """Call once the stream has ended: emit every remaining pending
        index using a truncated (shorter) window, matching
        apply_temporal_smoothing()'s behavior at the tail of a finite
        sequence. Safe to call multiple times (returns [] once drained)."""
        ready = []
        n = len(self._raw)
        while self._next_emit_idx < n:
            ready.append(self._smooth_at(self._next_emit_idx))
            self._next_emit_idx += 1
        return ready

    def _drain_ready(self) -> list[int]:
        ready = []
        n = len(self._raw)
        while self._next_emit_idx < n and self._next_emit_idx + self.half < n:
            ready.append(self._smooth_at(self._next_emit_idx))
            self._next_emit_idx += 1
        return ready

    def _smooth_at(self, i: int) -> int:
        n = len(self._raw)
        start = max(0, i - self.half)
        end = min(n, i + self.half + 1)
        window = self._raw[start:end]

        counts: dict[int, int] = {}
        for label in window:
            counts[label] = counts.get(label, 0) + 1
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]

        if len(candidates) == 1:
            return candidates[0]
        return self._raw[i]  # tie -> keep the raw prediction, no WALK priority
