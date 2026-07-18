"""
OutputSink: pluggable destinations for DecisionEngine output (Phase B slice 5).

Live WALK/STOP/IDLE decisions need to go somewhere - a terminal for local
debugging, a CSV file for offline analysis (the same shape
fast_causal_bci.run_decision_source() already writes, plus the smoothed
label DecisionEngine adds), a live web UI, and eventually a real actuator.
Each destination implements the same small OutputSink shape so
DecisionEngine's consumer loop doesn't need to know which one(s) it's
writing to, or how many.
"""
from __future__ import annotations

import csv
import queue
import sys
from pathlib import Path
from typing import List, Optional, Protocol, TextIO

from realtime.decision_engine import Decision

LABELS = {0: "STOP", 1: "WALK", 2: "IDLE"}


class OutputSink(Protocol):
    def write(self, decision: Decision) -> None:
        """Handle one new decision. Called once per DecisionEngine.run() yield."""
        ...

    def close(self) -> None:
        """Release any resources (file handles, connections, ...). No-op if none."""
        ...

    def __enter__(self) -> "OutputSink":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ConsoleOutput:
    """Print each decision to a stream (stdout by default) - local debugging."""

    def __init__(self, stream: TextIO = sys.stdout):
        self.stream = stream

    def write(self, decision: Decision) -> None:
        raw = LABELS[decision.raw_label]
        smoothed = LABELS[decision.smoothed_label] if decision.smoothed_label is not None else "-"
        print(
            f"t={decision.stream_time_s:7.2f}s  raw={raw:4s}  smoothed={smoothed:4s}  "
            f"conf={decision.confidence:.3f}  end_to_end={decision.end_to_end_ms:.2f}ms",
            file=self.stream,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> "ConsoleOutput":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class CSVOutput:
    """Write each decision as one CSV row - the same fields fast_causal_bci's
    own output_csv already writes, plus the smoothed label DecisionEngine adds."""

    FIELDNAMES = [
        "stream_time_s", "raw_label", "confidence", "smoothed_label",
        "feature_ms", "decision_ms", "end_to_end_ms",
    ]

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def write(self, decision: Decision) -> None:
        self._writer.writerow({
            "stream_time_s": decision.stream_time_s,
            "raw_label": LABELS[decision.raw_label],
            "confidence": decision.confidence,
            "smoothed_label": "" if decision.smoothed_label is None else LABELS[decision.smoothed_label],
            "feature_ms": decision.feature_ms,
            "decision_ms": decision.decision_ms,
            "end_to_end_ms": decision.end_to_end_ms,
        })

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CSVOutput":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class WebSocketOutput:
    """
    Broadcast each decision to every currently-subscribed live client.

    Deliberately has no actual network/websocket code here - that belongs to
    the web server in web_ui_live.py (Phase B slice 6), which will call
    subscribe()/unsubscribe() as browser clients connect/disconnect and
    forward each subscriber queue's items over its own websocket connection.
    Keeping the transport out of this class makes the output path testable
    (subscribe a plain queue.Queue, assert what arrives) without a running
    server or an open socket.
    """

    def __init__(self):
        self._subscribers: List["queue.Queue"] = []

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def write(self, decision: Decision) -> None:
        message = {
            "stream_time_s": decision.stream_time_s,
            "raw_label": LABELS[decision.raw_label],
            "confidence": decision.confidence,
            "smoothed_label": None if decision.smoothed_label is None else LABELS[decision.smoothed_label],
            "end_to_end_ms": decision.end_to_end_ms,
        }
        for q in list(self._subscribers):
            q.put(message)

    def close(self) -> None:
        for q in list(self._subscribers):
            self.unsubscribe(q)

    def __enter__(self) -> "WebSocketOutput":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class HardwareOutput:
    """
    Placeholder for a real HID/vJoy actuator output.

    Not implemented - there is no hardware available to implement or test
    against yet (see the project's Phase-out-of-scope note on real hardware
    output). This class exists only to prove the OutputSink shape holds up
    for a fourth, very different kind of destination (a physical device,
    not a file/stream/queue) before real hardware arrives, so
    DecisionEngine's call site never needs to change when it does.
    Raises immediately on construction rather than failing later on first
    write(), so misuse can't be missed.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HardwareOutput is an unimplemented placeholder - no HID/vJoy device "
            "is available to build or test against yet. Wire a real device driver "
            "here once one exists."
        )

    def write(self, decision: Decision) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MultiOutput:
    """Fan a single decision stream out to several sinks at once (e.g. a
    live ConsoleOutput plus a CSVOutput for the session log)."""

    def __init__(self, sinks: List[OutputSink]):
        self.sinks = list(sinks)

    def write(self, decision: Decision) -> None:
        for sink in self.sinks:
            sink.write(decision)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

    def __enter__(self) -> "MultiOutput":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
