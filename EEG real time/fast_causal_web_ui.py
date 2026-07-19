"""Dependency-free local Web UI for live causal BCI predictions."""
from __future__ import annotations

import argparse
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from fast_causal_bci import (
    FastCausalModel, RecordedReplaySource, load_recording, run_decision_source,
)
from parse_events import parse_events


HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
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
<div class="top"><div><h1>Fast Causal BCI</h1><div class="muted">9 kanal · lfilter · 200 ms pencere · 50 ms karar</div></div><div id="status" class="badge">HAZIR</div></div>
<div class="grid"><section class="card"><div id="state" class="state idle"><div><div class="word" id="word">IDLE</div><div class="confidence" id="confidence">Henüz tahmin yok</div></div></div>
<div class="metrics"><div class="metric"><span>EEG zamanı</span><b id="stream">0.00 s</b></div><div class="metric"><span>Smoothing dahil</span><b id="latency">—</b></div><div class="metric"><span>Feature</span><b id="feature">—</b></div><div class="metric"><span>Source jitter</span><b id="jitter">—</b></div></div>
<div class="bar"><i id="progress"></i></div></section>
<section class="card"><form id="form"><label>EDF yolu</label><input id="edf" value="__EDF__"><label>Model klasörü</label><input id="model" value="__MODEL__"><label>Events TSV (opsiyonel)</label><input id="events" value="__EVENTS__"><div class="row"><div><label>Maksimum süre (sn, boş=tümü)</label><input id="maxSeconds" value="60"></div><div><label>Oynatma</label><input value="Gerçek zaman (1×)" disabled></div></div><div style="margin-top:16px"><button class="start" type="submit">▶ Başlat</button><button class="stop" type="button" id="stop">■ Durdur</button></div><div id="error" class="error"></div></form></section>
<section class="card" style="grid-column:1/-1"><h3 style="margin-top:0">Son tahminler</h3><div class="scroll"><table><thead><tr><th>EEG zamanı</th><th>Tahmin</th><th>Smoothed</th><th>Güven</th><th>Gerçek</th><th>Toplam gecikme</th></tr></thead><tbody id="rows"></tbody></table></div></section></div></div>
<script>
const $=id=>document.getElementById(id);let timer=null;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function start(e){e.preventDefault();$('error').textContent='';let body={edf:$('edf').value,model:$('model').value,events:$('events').value,max_seconds:$('maxSeconds').value||null};let r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let j=await r.json();if(!r.ok)$('error').textContent=j.error||'Başlatılamadı';poll()}
async function stop(){await fetch('/api/stop',{method:'POST'});poll()}
async function poll(){try{let j=await(await fetch('/api/status')).json();$('status').textContent=j.running?'ÇALIŞIYOR':(j.finished?'TAMAMLANDI':'HAZIR');if(j.error)$('error').textContent=j.error;let p=j.latest;if(p){let shown=p.smoothed_prediction||p.prediction;let cls=shown.toLowerCase();$('state').className='state '+cls;$('word').textContent=shown;$('confidence').textContent='%'+(p.confidence*100).toFixed(1)+' güven';$('stream').textContent=p.stream_time_s.toFixed(2)+' s';$('latency').textContent=p.total_latency_ms.toFixed(2)+' ms';$('feature').textContent=p.feature_ms.toFixed(2)+' ms';$('jitter').textContent=(p.source_lateness_ms??0).toFixed(2)+' ms';$('progress').style.width=(j.progress*100).toFixed(1)+'%'}$('rows').innerHTML=(j.history||[]).slice().reverse().map(x=>`<tr><td>${x.stream_time_s.toFixed(2)} s</td><td>${esc(x.prediction)}</td><td>${esc(x.smoothed_prediction||'—')}</td><td>%${(x.confidence*100).toFixed(1)}</td><td>${esc(x.truth||'—')}</td><td>${x.total_latency_ms.toFixed(2)} ms</td></tr>`).join('')}catch(e){}clearTimeout(timer);timer=setTimeout(poll,100)}
$('form').addEventListener('submit',start);$('stop').addEventListener('click',stop);poll();
</script></body></html>"""


class AppState:
    def __init__(self):
        self.lock = threading.Lock(); self.stop_event = threading.Event()
        self.thread = None; self.running = False; self.finished = False
        self.error = None; self.latest = None; self.history = deque(maxlen=100)
        self.progress = 0.0; self.total = 1.0

    def snapshot(self):
        with self.lock:
            return {"running": self.running, "finished": self.finished,
                    "error": self.error, "latest": self.latest,
                    "history": list(self.history), "progress": self.progress}

    def start(self, config):
        with self.lock:
            if self.running: raise RuntimeError("Bir replay zaten çalışıyor")
            self.stop_event.clear(); self.running=True; self.finished=False
            self.error=None; self.latest=None; self.history.clear(); self.progress=0
        self.thread=threading.Thread(target=self._worker,args=(config,),daemon=True);self.thread.start()

    def stop(self): self.stop_event.set()

    def _on_record(self, record):
        with self.lock:
            self.latest=record;self.history.append(record)
            self.progress=min(1,record["stream_time_s"]/self.total)

    def _worker(self, cfg):
        try:
            model=FastCausalModel.load(cfg["model"])
            signals,channels,fs=load_recording(cfg["edf"],model.channels)
            max_seconds=float(cfg["max_seconds"]) if cfg.get("max_seconds") else None
            total=min(len(signals[ch]) for ch in channels)/fs
            if max_seconds is not None: total=min(total,max_seconds)
            self.total=total
            source=RecordedReplaySource(
                signals, channels, fs, realtime_pace=True, max_seconds=max_seconds
            )
            events=parse_events(cfg["events"]) if cfg.get("events") else None
            run_decision_source(
                model, source, output_csv=None, events=events,
                on_decision=self._on_record, stop_check=self.stop_event.is_set,
            )
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
