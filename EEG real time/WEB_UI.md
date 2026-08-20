# Live prediction Web UI

The UI replays an EDF recording at real EEG speed and shows WALK/STOP
predictions, confidence score, EEG time, and latency metrics live in the
browser.

## Setup

```powershell
python -m pip install -r ../requirements.txt
```

## Running

From this folder:

```powershell
python fast_causal_web_ui.py --port 8766
```

Then open `http://127.0.0.1:8766` in your browser. In the UI, pick the EDF
file and the matching model folder, then click **Start**. An events TSV is
optional; if supplied, the ground-truth label is also shown in the recent
predictions table.

Default field values can also be passed on the command line:

```powershell
python fast_causal_web_ui.py --port 8766 `
  --edf "C:\\data\\recording.edf" `
  --model "models\\ses-01-to-ses-02" `
  --events "C:\\data\\events.tsv"
```

The server binds to `127.0.0.1` only by default, for security.

## Verification

In a 3-second real-time-paced EDF smoke test, the HTTP UI returned 200, 41
predictions reached the UI API, and no errors occurred. This UI does not
change the model - the reported cross-session balanced accuracy stays at
**73.77%**.
