import pdfplumber
import re

def parse_events(path):
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
