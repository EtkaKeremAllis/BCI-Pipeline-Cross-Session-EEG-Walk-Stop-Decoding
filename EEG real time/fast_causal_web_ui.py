"""Dependency-free local Web UI for live causal BCI predictions."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fast_causal_bci import (
    FastCausalModel, FeatureStream, LABELS, RecordedReplaySource,
    event_label_at, load_recording,
)
from parse_events import parse_events


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fast Causal BCI</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#e8eef7;background:#07111f}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 20% 0,#173256 0,#07111f 45%);min-height:100vh}
.wrap{max-width:1180px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:end;gap:20px}
h1{margin:0;font-size:28px}.muted{color:#91a4bd}.badge{padding:7px 12px;border-radius:999px;background:#17283d;color:#a9bad0}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:18px;margin-top:20px}.card{background:rgba(15,29,47,.92);border:1px solid #263d59;border-radius:18px;padding:20px;box-shadow:0 18px 50px #0005}
label{font-size:12px;color:#91a4bd;display:block;margin:10px 0 5px}input{width:100%;border:1px solid #304962;background:#091726;color:#e8eef7;border-radius:9px;padding:10px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}button{border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}.start{background:#4bd5a0;color:#062117}.stop{background:#ff6b76;color:#2b070b;margin-left:8px}
.state{min-height:230px;display:grid;place-items:center;text-align:center;border-radius:16px;background:#0a1727;border:1px solid #263d59;transition:.2s}.state .word{font-size:68px;font-weight:900;letter-spacing:3px}.state.walk{background:#08271f;border-color:#2ed299}.state.stop{background:#32131a;border-color:#ff6270}.state.idle{background:#1c2330}.confidence{font-size:17px;color:#bdcbe0}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.metric{padding:12px;background:#091726;border-radius:12px}.metric b{display:block;font-size:21px;margin-top:5px}.metric span{font-size:11px;color:#8fa2ba}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #24384f}th{color:#8fa2ba}.scroll{max-height:330px;overflow:auto}
.bar{height:7px;border-radius:9px;background:#091726;overflow:hidden;margin-top:12px}.bar i{display:block;height:100%;background:#4bd5a0;width:0}.error{color:#ff8790;min-height:20px;margin-top:8px;font-size:13px}
@media(max-width:850px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.state .word{font-size:50px}}
</style></head><body><div class="wrap">
<div class="top"><div><h1>Fast Causal BCI</h1><div class="muted">9 channels · lfilter · 200 ms window · 50 ms decision</div></div><div id="status" class="badge">READY</div></div>
<div class="grid"><section class="card"><div id="state" class="state idle"><div><div class="word" id="word">IDLE</div><div class="confidence" id="confidence">No prediction yet</div></div></div>
<div class="metrics"><div class="metric"><span>EEG time</span><b id="stream">0.00 s</b></div><div class="metric"><span>End-to-end</span><b id="latency">—</b></div><div class="metric"><span>Feature</span><b id="feature">—</b></div><div class="metric"><span>Source jitter</span><b id="jitter">—</b></div></div>
<div class="bar"><i id="progress"></i></div></section>
<section class="card"><form id="form"><label>EDF path</label><input id="edf" value="__EDF__"><label>Model folder</label><input id="model" value="__MODEL__"><label>Events TSV (optional)</label><input id="events" value="__EVENTS__"><div class="row"><div><label>Max duration (s, empty=all)</label><input id="maxSeconds" value="60"></div><div><label>Playback</label><input value="Real time (1×)" disabled></div></div><div style="margin-top:16px"><button class="start" type="submit">▶ Start</button><button class="stop" type="button" id="stop">■ Stop</button></div><div id="error" class="error"></div></form></section>
<section class="card" style="grid-column:1/-1"><h3 style="margin-top:0">Recent predictions</h3><div class="scroll"><table><thead><tr><th>EEG time</th><th>Prediction</th><th>Confidence</th><th>Truth</th><th>Latency</th></tr></thead><tbody id="rows"></tbody></table></div></section></div></div>
<script>
const $=id=>document.getElementById(id);let timer=null;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function start(e){e.preventDefault();$('error').textContent='';let body={edf:$('edf').value,model:$('model').value,events:$('events').value,max_seconds:$('maxSeconds').value||null};let r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let j=await r.json();if(!r.ok)$('error').textContent=j.error||'Failed to start';poll()}
async function stop(){await fetch('/api/stop',{method:'POST'});poll()}
async function poll(){try{let j=await(await fetch('/api/status')).json();$('status').textContent=j.running?'RUNNING':(j.finished?'FINISHED':'READY');if(j.error)$('error').textContent=j.error;let p=j.latest;if(p){let cls=p.prediction.toLowerCase();$('state').className='state '+cls;$('word').textContent=p.prediction;$('confidence').textContent=(p.confidence*100).toFixed(1)+'% confidence';$('stream').textContent=p.stream_time_s.toFixed(2)+' s';$('latency').textContent=p.end_to_end_ms.toFixed(2)+' ms';$('feature').textContent=p.feature_ms.toFixed(2)+' ms';$('jitter').textContent=(p.source_lateness_ms??0).toFixed(2)+' ms';$('progress').style.width=(j.progress*100).toFixed(1)+'%'}$('rows').innerHTML=(j.history||[]).slice().reverse().map(x=>`<tr><td>${x.stream_time_s.toFixed(2)} s</td><td>${esc(x.prediction)}</td><td>${(x.confidence*100).toFixed(1)}%</td><td>${esc(x.truth||'—')}</td><td>${x.end_to_end_ms.toFixed(2)} ms</td></tr>`).join('')}catch(e){}clearTimeout(timer);timer=setTimeout(poll,100)}
$('form').addEventListener('submit',start);$('stop').addEventListener('click',stop);poll();
</script></body></html>"""


class AppState:
    def __init__(self):
        self.lock = threading.Lock(); self.stop_event = threading.Event()
        self.thread = None; self.running = False; self.finished = False
        self.error = None; self.latest = None; self.history = deque(maxlen=100)
        self.progress = 0.0

    def snapshot(self):
        with self.lock:
            return {"running": self.running, "finished": self.finished,
                    "error": self.error, "latest": self.latest,
                    "history": list(self.history), "progress": self.progress}

    def start(self, config):
        with self.lock:
            if self.running: raise RuntimeError("A replay is already running")
            self.stop_event.clear(); self.running=True; self.finished=False
            self.error=None; self.latest=None; self.history.clear(); self.progress=0
        self.thread=threading.Thread(target=self._worker,args=(config,),daemon=True);self.thread.start()

    def stop(self): self.stop_event.set()

    def _worker(self, cfg):
        try:
            model=FastCausalModel.load(cfg["model"])
            signals,channels,fs=load_recording(cfg["edf"],model.channels)
            max_seconds=float(cfg["max_seconds"]) if cfg.get("max_seconds") else None
            total=min(len(signals[ch]) for ch in channels)/fs
            if max_seconds is not None: total=min(total,max_seconds)
            source=RecordedReplaySource(signals,channels,fs,True,max_seconds)
            events=parse_events(cfg["events"]) if cfg.get("events") else None
            stream=FeatureStream(model.channels,model.fs,model.window_seconds,model.context_seconds)
            chunk_samples=max(1,int(round(model.step_seconds*model.fs)))
            for chunk in source.chunks(chunk_samples):
                if self.stop_event.is_set(): break
                arrived=time.perf_counter();stream.push(chunk)
                if not stream.ready: continue
                features=stream.features();ready=time.perf_counter()
                pred,proba=model.predict_features(features.reshape(1,-1));decided=time.perf_counter()
                t=stream.samples_seen/model.fs;truth=event_label_at(t,events) if events else None
                record={"stream_time_s":t,"prediction":LABELS[int(pred[0])],
                  "confidence":float(proba[0,int(pred[0])]),"truth":"" if truth is None else LABELS[int(truth)],
                  "feature_ms":(ready-arrived)*1000,"decision_ms":(decided-ready)*1000,
                  "end_to_end_ms":(decided-arrived)*1000,"source_lateness_ms":source.last_lateness_ms}
                with self.lock:
                    self.latest=record;self.history.append(record);self.progress=min(1,t/total)
        except Exception as exc:
            with self.lock:self.error=f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:self.running=False;self.finished=True


def make_handler(state, defaults):
    page=(HTML.replace("__EDF__",str(defaults.edf or "")).replace("__MODEL__",str(defaults.model or "")).replace("__EVENTS__",str(defaults.events or ""))).encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send_json(self,obj,status=200):
            data=json.dumps(obj).encode();self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if urlparse(self.path).path=="/api/status": return self.send_json(state.snapshot())
            if urlparse(self.path).path=="/":
                self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(page)));self.end_headers();return self.wfile.write(page)
            self.send_error(404)
        def do_POST(self):
            path=urlparse(self.path).path
            if path=="/api/stop": state.stop();return self.send_json({"ok":True})
            if path=="/api/start":
                try:
                    length=int(self.headers.get("Content-Length","0"));cfg=json.loads(self.rfile.read(length));state.start(cfg);return self.send_json({"ok":True})
                except Exception as exc:return self.send_json({"error":str(exc)},400)
            self.send_error(404)
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8766);p.add_argument("--edf");p.add_argument("--model");p.add_argument("--events");args=p.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),make_handler(AppState(),args))
    print(f"Fast Causal BCI UI: http://{args.host}:{args.port}")
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=="__main__":main()
