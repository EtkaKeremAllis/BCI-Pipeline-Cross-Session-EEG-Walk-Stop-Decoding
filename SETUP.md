# Setup Guide

This guide installs the dependencies required by `bci_web_ui.py` and
`bci_pipeline_v2.9.py` in an isolated Python virtual environment.

The Web UI itself uses only the Python standard library. NumPy and SciPy are
required by the signal-processing pipeline. `pdfplumber` is required only for
legacy PDF event files, but it is included in `requirements.txt` so every Web
UI workflow works after one installation.

## 1. Download the repository

Choose one method.

### Option A: Download ZIP

1. Open the repository on GitHub.
2. Select **Code → Download ZIP**.
3. Extract the archive to a normal user-writable directory, for example:

   ```text
   C:\BCI-Pipeline-Cross-Session-EEG-Walk-Stop-Decoding-master
   ```

### Option B: Clone with Git

```powershell
git clone https://github.com/EtkaKeremAllis/BCI-Pipeline-Cross-Session-EEG-Walk-Stop-Decoding.git
cd BCI-Pipeline-Cross-Session-EEG-Walk-Stop-Decoding
```

## 2. Verify the repository contents

Open PowerShell in the extracted or cloned repository directory:

```powershell
cd "C:\BCI-Pipeline-Cross-Session-EEG-Walk-Stop-Decoding-master"
Get-ChildItem
```

The directory should contain at least:

```text
bci_web_ui.py
bci_pipeline_v2.9.py
modern_bci_v2.py
edf_reader.py
parse_events.py
requirements.txt
```

Do not create the virtual environment in `C:\` unless the repository itself is
located there. Run the following commands from the repository directory.

## 3. Check Python

The currently verified environment uses Python 3.13.14. Python 3.10–3.13 is
recommended, but not every version in this range has been systematically tested.

### Windows

Try both commands:

```powershell
py --version
python --version
```

If either command prints Python 3.10 or newer, use that command in the next
step.

If Windows reports `Python was not found`, Python may be installed without
being available on `PATH`. Locate the executable or use its full path:

```powershell
& "C:\Users\YOUR_NAME\AppData\Local\Programs\Python\Python313\python.exe" --version
```

Replace `YOUR_NAME` and `Python313` with the actual installation path.

If Python is not installed, install it from
[python.org](https://www.python.org/downloads/windows/). Enable **Add Python to
PATH** and install the Python launcher when those options are shown.

If the Microsoft Store opens instead of Python, disable the `python.exe` and
`python3.exe` aliases under:

```text
Settings → Apps → Advanced app settings → App execution aliases
```

### Linux/macOS

```bash
python3 --version
```

## 4. Create a virtual environment

A virtual environment keeps this project's packages separate from the global
Python installation.

### Windows with the Python launcher

```powershell
py -m venv .venv
```

### Windows with `python` on PATH

```powershell
python -m venv .venv
```

### Windows with a full Python path

```powershell
& "C:\path\to\python.exe" -m venv .venv
```

Confirm that the environment was created:

```powershell
Test-Path .\.venv\Scripts\python.exe
```

The expected output is:

```text
True
```

### Linux/macOS

```bash
python3 -m venv .venv
```

## 5. Install dependencies

### Recommended Windows method: no activation required

This method avoids PowerShell execution-policy problems:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Optional Windows activation

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, allow scripts only for the current
PowerShell process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This does not permanently change the system execution policy.

### Linux/macOS

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 6. Verify the installation

### Windows

```powershell
.\.venv\Scripts\python.exe -c "import numpy, scipy, pdfplumber; print('Dependencies OK')"
.\.venv\Scripts\python.exe bci_pipeline_v2.9.py --help
```

### Linux/macOS

```bash
python -c "import numpy, scipy, pdfplumber; print('Dependencies OK')"
python bci_pipeline_v2.9.py --help
```

## 7. Start the Web UI

### Windows

```powershell
.\.venv\Scripts\python.exe -X utf8 bci_web_ui.py
```

### Linux/macOS

```bash
python -X utf8 bci_web_ui.py
```

Open the following address in a browser:

```text
http://127.0.0.1:8765
```

The server listens only on the local computer by default. EEG data is
processed locally and is not uploaded by the Web UI.

Stop the server with `Ctrl+C` in the terminal.

## Troubleshooting

### `Python was not found`

Python is not installed or is not available on `PATH`. Use the full path to
`python.exe`, install Python from python.org, or correct the Windows app
execution aliases as described above.

### `Activate.ps1 is not recognized`

The virtual environment was not created in the current directory, or the
terminal is not in the repository directory. Check:

```powershell
Get-Location
Test-Path .\.venv\Scripts\Activate.ps1
```

Activation is optional. You can always use:

```powershell
.\.venv\Scripts\python.exe -X utf8 bci_web_ui.py
```

### PowerShell blocks script execution

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `ModuleNotFoundError`

Confirm that commands use the virtual environment's interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip list
```

### Port 8765 is already in use

An earlier Web UI process may still be running. Return to its terminal and
press `Ctrl+C`, then start the server again.

### PDF event parsing fails

Confirm that `pdfplumber` is installed in the same virtual environment:

```powershell
.\.venv\Scripts\python.exe -c "import pdfplumber; print(pdfplumber.__version__)"
```

TSV event files do not require PDF parsing.
