"""
Verify web_ui_live.py's AppState + HTTP handler against real data, over a
real (loopback) HTTP connection - not just calling AppState methods
directly, to prove the whole server wiring (routes, JSON (de)serialization,
background worker thread) actually works.

Run from the "EEG real time" directory:
    pytest tests/test_web_ui_live.py -v
"""
import http.client
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_REALTIME_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(EEG_REALTIME_DIR)

if EEG_REALTIME_DIR not in sys.path:
    sys.path.insert(0, EEG_REALTIME_DIR)

from realtime.web_ui_live import AppState, make_handler  # noqa: E402

MODEL_DIR = os.path.join(EEG_REALTIME_DIR, "models", "ses-01-to-ses-02")
TEST_EDF = os.path.join(REPO_ROOT, "sub-02", "ses-02", "eeg",
                         "sub-02_ses-02_task-training_eeg.edf")
TEST_EVENTS = os.path.join(REPO_ROOT, "sub-02", "ses-02", "eeg",
                            "sub-02_ses-02_task-training_acq-rexcommand_events.tsv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_DIR) and os.path.exists(TEST_EDF)),
    reason="real model/recording not present in this checkout",
)


class _Defaults:
    edf = model = events = None


@pytest.fixture
def server():
    state = AppState()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, _Defaults()))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield port
    httpd.shutdown()
    httpd.server_close()


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def _post(port, path, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps(payload).encode()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    result = resp.read()
    conn.close()
    return resp.status, result


def _wait_until_finished(port, timeout=30):
    deadline = time.time() + timeout
    snapshot = None
    while time.time() < deadline:
        status, body = _get(port, "/api/status")
        assert status == 200
        snapshot = json.loads(body)
        if snapshot["finished"]:
            return snapshot
        time.sleep(0.2)
    raise AssertionError(f"Replay did not finish within {timeout}s: {snapshot}")


def test_index_page_serves_html(server):
    status, body = _get(server, "/")
    assert status == 200
    assert b"Phase B Live BCI" in body


def test_full_replay_via_http_reaches_decision_engine(server):
    status, body = _post(server, "/api/start", {
        "edf": TEST_EDF, "model": MODEL_DIR, "events": TEST_EVENTS,
        "max_seconds": "5", "smoothing_window": 3,
    })
    assert status == 200, body

    snapshot = _wait_until_finished(server)
    assert snapshot["error"] is None
    assert snapshot["latest"] is not None
    assert len(snapshot["history"]) > 0

    latest = snapshot["latest"]
    assert latest["raw_label"] in ("STOP", "WALK", "IDLE")
    assert latest["smoothed_label"] in ("STOP", "WALK", "IDLE")
    assert latest["truth"] in ("STOP", "WALK", "IDLE", "")
    assert isinstance(latest["total_latency_ms"], float)
    assert latest["total_latency_ms"] >= 0


def test_stop_endpoint_ends_replay_early(server):
    status, body = _post(server, "/api/start", {
        "edf": TEST_EDF, "model": MODEL_DIR, "events": None,
        "max_seconds": None, "smoothing_window": 1,
    })
    assert status == 200, body

    time.sleep(0.5)  # let the worker actually get going
    status, _ = _post(server, "/api/stop", {})
    assert status == 200

    snapshot = _wait_until_finished(server)
    # The full sub-02/ses-02 recording is ~250s of real-time-paced replay;
    # stopping after 0.5s must finish far short of that, proving stop_check
    # (wired through DecisionEngine.run()) actually cuts the loop short.
    assert snapshot["progress"] < 0.2


def test_cannot_start_while_already_running(server):
    status, _ = _post(server, "/api/start", {
        "edf": TEST_EDF, "model": MODEL_DIR, "events": None,
        "max_seconds": None, "smoothing_window": 1,
    })
    assert status == 200

    status, body = _post(server, "/api/start", {
        "edf": TEST_EDF, "model": MODEL_DIR, "events": None,
        "max_seconds": None, "smoothing_window": 1,
    })
    assert status == 400
    assert b"already running" in body

    _post(server, "/api/stop", {})
    _wait_until_finished(server)
