import csv
import os
import re

REQUIRED_COLUMNS = {'onset', 'duration', 'trial_type'}


def _parse_tsv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter='\t')
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: events TSV is missing required column(s) {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        events = []
        for line_no, row in enumerate(reader, start=2):  # header is line 1
            try:
                onset = float(row['onset'])
                duration = float(row['duration'])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}: line {line_no}: onset/duration must be numeric "
                    f"(got onset={row.get('onset')!r}, duration={row.get('duration')!r})"
                )

            trial_type = (row.get('trial_type') or '').strip()
            if not trial_type:
                raise ValueError(f"{path}: line {line_no}: empty trial_type")

            events.append((onset, duration, trial_type))
    return events


def _parse_pdf(path):
    import pdfplumber  # optional dependency - only needed for legacy PDF exports

    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text()
    lines = text.split('\n')
    events = []
    started = False
    pattern = re.compile(r'^([\d.]+)\s+([\d.]+)\s+(\S+)$')
    for line in lines:
        if line.strip().startswith('onset'):
            started = True
            continue
        if started:
            m = pattern.match(line.strip())
            if m:
                onset, duration, trial_type = m.groups()
                events.append((float(onset), float(duration), trial_type))
            else:
                break
    return events


def parse_events(path):
    """Parse a BIDS-style events file into (onset, duration, trial_type) tuples.

    Format is chosen by file extension: '.tsv' is read as tab-separated text
    (the BIDS standard), '.pdf' falls back to the legacy PDF-export parser
    used for older recordings whose events were only available as PDF.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.tsv':
        return _parse_tsv(path)
    if ext == '.pdf':
        return _parse_pdf(path)
    raise ValueError(
        f"{path}: unsupported events file extension {ext!r}. Expected '.tsv' or '.pdf'."
    )


if __name__ == '__main__':
    cmd_events = parse_events('/mnt/user-data/uploads/sub-01_ses-01_task-training_acq-rexcommand_events.tsv')
    state_events = parse_events('/mnt/user-data/uploads/sub-01_ses-01_task-training_acq-rexstate_events.tsv')
    print('rexcommand events:', len(cmd_events))
    for e in cmd_events:
        print(e)
    print()
    print('rexstate events:', len(state_events))
    for e in state_events:
        print(e)
