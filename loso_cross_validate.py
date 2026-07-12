"""
loso_cross_validate.py
=======================
Leave-one-subject-out (LOSO) cross-validation runner.

For each subject present in a --dataset-list CSV (the same subject,session,
edf,events format used by `bci_pipeline_v*.py --mode train_multi`), trains
a pooled model on every OTHER subject's sessions, then evaluates it on that
held-out subject's own session(s) via validate_timeline-style full-timeline
inference. This gives a per-subject generalization estimate that a plain
train/test split within one subject's own sessions cannot.

Usage:
    python loso_cross_validate.py --dataset-list train_list.txt \
        --dataset-dir . --output-dir loso_results

Outputs (in --output-dir):
    loso_results.csv       one row per (subject, session) test evaluation
    loso_fold_summary.csv  one row per held-out subject (mean over its sessions)
    loso_summary.json      overall mean/std across folds + full detail
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import statistics


def _load_pipeline_module():
    """Locate and import the main pipeline script without hardcoding a
    version number (it has already been renamed once, v2.8 -> v2.9)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(here, "bci_pipeline_v*.py"))
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected exactly one bci_pipeline_v*.py in {here}, found: {candidates}"
        )
    spec = importlib.util.spec_from_file_location("bci_pipeline", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_dataset_list(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'subject', 'session', 'edf', 'events'}
    if not rows or required - set(rows[0].keys()):
        raise SystemExit(f"{path}: expected columns {sorted(required)}")
    return rows


def _write_fold_dataset_list(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'session', 'edf', 'events'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_loso(dataset_list_csv, output_dir, dataset_dir='.', **train_kwargs):
    bp = _load_pipeline_module()
    rows = _read_dataset_list(dataset_list_csv)
    subjects = sorted({r['subject'] for r in rows})
    if len(subjects) < 2:
        raise SystemExit(
            f"LOSO needs at least 2 distinct subjects in {dataset_list_csv}, "
            f"found {len(subjects)}: {subjects}"
        )

    os.makedirs(output_dir, exist_ok=True)
    per_session_rows = []

    for held_out in subjects:
        print("=" * 70)
        print(f"LOSO FOLD: held-out subject = {held_out}")
        print("=" * 70)

        train_rows = [r for r in rows if r['subject'] != held_out]
        test_rows = [r for r in rows if r['subject'] == held_out]

        fold_dir = os.path.join(output_dir, f"fold_{held_out}")
        os.makedirs(fold_dir, exist_ok=True)
        model_dir = os.path.join(fold_dir, "model")

        fold_list_path = os.path.join(fold_dir, f"train_list_excluding_{held_out}.csv")
        _write_fold_dataset_list(train_rows, fold_list_path)

        bp.run_train_multi(fold_list_path, model_dir, dataset_dir=dataset_dir, **train_kwargs)

        for test_row in test_rows:
            edf_path = bp._resolve_path(test_row['edf'], dataset_dir)
            events_path = bp._resolve_path(test_row['events'], dataset_dir)
            session_tag = f"{test_row['subject']}_{test_row['session']}"
            eval_dir = os.path.join(fold_dir, f"eval_{session_tag}")

            bp.run_validate_timeline(edf_path, events_path, model_dir, eval_dir)

            metrics_path = os.path.join(eval_dir, 'timeline_metrics.json')
            with open(metrics_path) as f:
                metrics = json.load(f)

            per_session_rows.append({
                'held_out_subject': held_out,
                'session': test_row['session'],
                'total_windows': metrics['total_windows'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'deployment_accuracy_non_idle': metrics['deployment_accuracy_non_idle'],
                'walk_recall': metrics['walk_recall'],
                'stop_recall': metrics['stop_recall'],
                'collapse_warning': metrics['collapse_warning'],
                'dominant_prediction_fraction': metrics['dominant_prediction_fraction'],
            })

    detail_path = os.path.join(output_dir, 'loso_results.csv')
    with open(detail_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_session_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_session_rows)

    fold_summaries = []
    for held_out in subjects:
        sessions = [r for r in per_session_rows if r['held_out_subject'] == held_out]
        fold_summaries.append({
            'held_out_subject': held_out,
            'n_sessions': len(sessions),
            'mean_balanced_accuracy': statistics.fmean(r['balanced_accuracy'] for r in sessions),
            'mean_deployment_accuracy_non_idle': statistics.fmean(
                r['deployment_accuracy_non_idle'] for r in sessions),
            'any_collapse_warning': any(r['collapse_warning'] for r in sessions),
        })

    fold_summary_path = os.path.join(output_dir, 'loso_fold_summary.csv')
    with open(fold_summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(fold_summaries)

    fold_balanced_accs = [f['mean_balanced_accuracy'] for f in fold_summaries]
    fold_deploy_accs = [f['mean_deployment_accuracy_non_idle'] for f in fold_summaries]
    overall = {
        'n_folds': len(subjects),
        'subjects': subjects,
        'mean_balanced_accuracy': statistics.fmean(fold_balanced_accs),
        'std_balanced_accuracy': statistics.pstdev(fold_balanced_accs) if len(fold_balanced_accs) > 1 else 0.0,
        'mean_deployment_accuracy_non_idle': statistics.fmean(fold_deploy_accs),
        'std_deployment_accuracy_non_idle': statistics.pstdev(fold_deploy_accs) if len(fold_deploy_accs) > 1 else 0.0,
        'folds': fold_summaries,
    }
    summary_path = os.path.join(output_dir, 'loso_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(overall, f, indent=2)

    print("\n" + "=" * 70)
    print("LOSO SUMMARY")
    print("=" * 70)
    for f_ in fold_summaries:
        print(f"  [{f_['held_out_subject']}] balanced_acc={f_['mean_balanced_accuracy']:.2%}  "
              f"deploy_acc={f_['mean_deployment_accuracy_non_idle']:.2%}  "
              f"collapse={'YES' if f_['any_collapse_warning'] else 'no'}")
    print(f"\nMean balanced accuracy across {len(subjects)} folds: "
          f"{overall['mean_balanced_accuracy']:.2%} (+/- {overall['std_balanced_accuracy']:.2%})")
    print(f"Outputs: {detail_path}, {fold_summary_path}, {summary_path}")

    return overall


def main():
    parser = argparse.ArgumentParser(description="Leave-one-subject-out cross-validation")
    parser.add_argument('--dataset-list', required=True)
    parser.add_argument('--dataset-dir', default='.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--channel-set', default='motor3',
                         choices=['motor3', 'motor5', 'motor9', 'motor13', 'all_eeg'])
    parser.add_argument('--n-features-select', type=int, default=45)
    parser.add_argument('--lda-shrinkage', type=float, default=0.0)
    parser.add_argument('--idle-distance-threshold', type=float, default=999.0)
    parser.add_argument('--confidence-threshold', type=float, default=0.45)
    parser.add_argument('--channel-normalization', choices=['none', 'zscore'], default='zscore')
    parser.add_argument('--balance-classes', choices=['none', 'downsample', 'class_weight'], default='none')
    parser.add_argument('--balance-subjects', choices=['none', 'downsample'], default='none')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_loso(
        args.dataset_list, args.output_dir, dataset_dir=args.dataset_dir,
        channel_set=args.channel_set, n_features_select=args.n_features_select,
        lda_shrinkage=args.lda_shrinkage, idle_distance_threshold=args.idle_distance_threshold,
        confidence_threshold=args.confidence_threshold,
        channel_normalization=args.channel_normalization,
        balance_classes=args.balance_classes, balance_subjects=args.balance_subjects,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
