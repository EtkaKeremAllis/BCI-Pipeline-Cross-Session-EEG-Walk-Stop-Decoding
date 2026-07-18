"""
EEGSource: the source-agnostic streaming interface Phase B is built on.

This extends fast_causal_bci.py's ChunkSource Protocol (kept untouched there
for backward compatibility) with what Phase B additionally needs:

  - `channel_names`: an explicit alias for `channels`, matching the name used
    across the rest of Phase B's modules.
  - `is_live`: distinguishes a recorded replay from a real device at runtime,
    for logging/UI (e.g. the web UI should visibly flag replay vs. live).
  - `close()` / context-manager support: recorded replay has nothing to
    release, but a real device source will (closing a socket, serial port,
    SDK handle, ...). Defining the shape now means a future LiveDeviceSource
    doesn't require touching any code that already consumes an EEGSource.

Any object satisfying this Protocol (structurally - no inheritance required)
can be passed anywhere Phase B expects an EEGSource, including
fast_causal_bci.py's own ChunkSource-typed functions (predict_features,
run_decision_source, ...), since it is a superset of ChunkSource's shape.
"""
from __future__ import annotations

from typing import Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class EEGSource(Protocol):
    """Source boundary shared by recorded replay and a future live EEG device."""

    sampling_rate: float
    channels: Sequence[str]
    is_live: bool

    @property
    def channel_names(self) -> Sequence[str]:
        ...

    def chunks(self, chunk_samples: int) -> Iterator[Mapping[str, np.ndarray]]:
        """Yield successive chunk_samples-sized sample dicts, one per channel."""
        ...

    def close(self) -> None:
        """Release any resources held by this source. No-op for replay sources."""
        ...

    def __enter__(self) -> "EEGSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
