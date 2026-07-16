"""
Scan a directory tree for validate_timeline output (timeline_metrics.json)
produced by bci_pipeline_v*.py, and collect every run into one comparison
table (CSV + Markdown) - so results from many different configuration runs
(different channel sets, held-out sessions, seeds, etc.) don't have to be
compared by hand, file by file.

Usage:
    python aggregate_results.py <root_dir> [--output-dir DIR]

Handles two timeline_metrics.json shapes:
  - the pipeline's native `run_validate_timeline` output: fields live at the
    top level (balanced_accuracy, walk_recall, stop_recall, collapse_warning,
    deployment_accuracy_non_idle, total_windows, ...).
  - a "benchmark-enriched" shape some experiment scripts produce (e.g. a
    LOSO sweep across many held-out sessions / train-session counts /
    seeds): the same kind of fields nested under a top-level "metadata" key,
    plus extra provenance (holdout_session, model_seed, combo_index,
    train_sessions, commit_sha, ...).
Both shapes are normalized into the same row schema below.
"""
import argparse
import csv
import json
import os

ROW_FIELDS = [
    'source_path', 'holdout_session', 'n_train_sessions', 'train_sessions',
    'combo_index', 'model_seed', 'commit_sha', 'accuracy', 'balanced_accuracy',
    'walk_recall', 'stop_recall', 'n_windows', 'collapse_warning',
]

MARKDOWN_COLUMNS = [
    'source_path', 'holdout_session', 'n_train_sessions', 'model_seed',
    'accuracy', 'balanced_accuracy', 'walk_recall', 'stop_recall',
    'n_windows', 'collapse_warning',
]


def _extract_row(data, source_path):
    """Normalize either timeline_metrics.json shape into one flat row dict."""
    if 'metadata' in data:
        m = data['metadata']
        return {
            'source_path': source_path,
            'holdout_session': m.get('holdout_session'),
            'n_train_sessions': m.get('n_train_sessions'),
            'train_sessions': m.get('train_sessions'),
            'combo_index': m.get('combo_index'),
            'model_seed': m.get('model_seed'),
            'commit_sha': m.get('commit_sha'),
            'accuracy': m.get('accuracy'),
            'balanced_accuracy': m.get('balanced_accuracy'),
            'walk_recall': m.get('walk_recall'),
            'stop_recall': m.get('stop_recall'),
            'n_windows': m.get('n_windows'),
            'collapse_warning': m.get('collapse_warning'),
        }

    # Native run_validate_timeline output: no "metadata" wrapper, and
    # "accuracy" is called deployment_accuracy_non_idle.
    return {
        'source_path': source_path,
        'holdout_session': None,
        'n_train_sessions': None,
        'train_sessions': None,
        'combo_index': None,
        'model_seed': None,
        'commit_sha': None,
        'accuracy': data.get('deployment_accuracy_non_idle'),
        'balanced_accuracy': data.get('balanced_accuracy'),
        'walk_recall': data.get('walk_recall'),
        'stop_recall': data.get('stop_recall'),
        'n_windows': data.get('total_windows'),
        'collapse_warning': data.get('collapse_warning'),
    }


def find_timeline_metrics(root_dir):
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if 'timeline_metrics.json' in filenames:
            yield os.path.join(dirpath, 'timeline_metrics.json')


def collect_rows(root_dir):
    rows = []
    for path in sorted(find_timeline_metrics(root_dir)):
        with open(path) as f:
            data = json.load(f)
        rel_path = os.path.relpath(path, root_dir)
        rows.append(_extract_row(data, rel_path))
    return rows


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    def fmt(value):
        if isinstance(value, float):
            return f'{value:.4f}'
        return '' if value is None else str(value)

    lines = [
        '| ' + ' | '.join(MARKDOWN_COLUMNS) + ' |',
        '|' + '|'.join(['---'] * len(MARKDOWN_COLUMNS)) + '|',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row[col]) for col in MARKDOWN_COLUMNS) + ' |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate validate_timeline results (timeline_metrics.json) "
                    "from a directory tree of experiment runs into one table.")
    parser.add_argument('root_dir', help="Directory to scan recursively.")
    parser.add_argument('--output-dir', default='.',
                         help="Where to write aggregated_results.csv/.md (default: current directory).")
    args = parser.parse_args()

    rows = collect_rows(args.root_dir)
    if not rows:
        raise SystemExit(f"No timeline_metrics.json files found under {args.root_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'aggregated_results.csv')
    md_path = os.path.join(args.output_dir, 'aggregated_results.md')
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    accs = [r['balanced_accuracy'] for r in rows if r['balanced_accuracy'] is not None]
    n_collapsed = sum(1 for r in rows if r['collapse_warning'])
    print(f"[*] Found {len(rows)} run(s) under {args.root_dir}")
    if accs:
        print(f"[*] Balanced accuracy: mean={sum(accs) / len(accs):.4f}, "
              f"min={min(accs):.4f}, max={max(accs):.4f}")
    print(f"[*] Collapsed runs: {n_collapsed}/{len(rows)}")
    print(f"[+] Wrote {csv_path}")
    print(f"[+] Wrote {md_path}")


if __name__ == '__main__':
    main()
