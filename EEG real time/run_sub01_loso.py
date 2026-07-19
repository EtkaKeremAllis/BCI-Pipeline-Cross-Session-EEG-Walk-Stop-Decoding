"""Strict session-level LOSO benchmark for sub-01 sessions 01 through 08."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from fast_causal_bci import event_label_at, extract_causal_features, load_recording
from modern_bci_v2 import SimpleLDA
from parse_events import parse_events


CHANNELS = ["FC3", "FC1", "FCz", "FC2", "FC4", "C3", "Cz", "C4", "CPz"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sub01_loso")
    )
    parser.add_argument("--window", type=float, default=0.2)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--context", type=float, default=1.0)
    parser.add_argument("--n-features", type=int, default=24)
    parser.add_argument("--shrinkage", type=float, default=0.2)
    args = parser.parse_args()

    started = time.perf_counter()
    sessions = [f"ses-{index:02d}" for index in range(1, 9)]
    features = {}
    for session in sessions:
        eeg_dir = args.data_root / session / "eeg"
        edf = next(eeg_dir.glob("*_eeg.edf"))
        events_path = next(eeg_dir.glob("*rexcommand_events.pdf"))
        signals, channels, fs = load_recording(edf, CHANNELS)
        events = parse_events(events_path)
        test_X, test_y, test_times = extract_causal_features(
            signals, channels, fs, events, args.window, args.step,
            training=False, context_seconds=args.context,
        )
        # FastCausalModel.train uses non-overlapping window-sized training
        # steps. Reuse the identical causal 50 ms stream at matching 200 ms
        # timestamps, but retain only windows whose complete context has one
        # label. No held-out information enters this deterministic selection.
        aligned = np.isclose(
            test_times / args.window, np.round(test_times / args.window), atol=1e-7
        )
        context_clean = np.asarray([
            event_label_at(t - args.context + 1 / fs, events) == int(label)
            for t, label in zip(test_times, test_y)
        ])
        train_mask = aligned & context_clean
        train_X, train_y = test_X[train_mask], test_y[train_mask]
        features[session] = (train_X, train_y, test_X, test_y)
        print(f"{session}: train={len(train_y)}, test={len(test_y)}", flush=True)

    rows = []
    for held_out in sessions:
        train_sessions = [session for session in sessions if session != held_out]
        X = np.vstack([features[session][0] for session in train_sessions])
        y = np.concatenate([features[session][1] for session in train_sessions])
        test_X, test_y = features[held_out][2:]

        scores = (X[y == 1].mean(0) - X[y == 0].mean(0)) ** 2 / (X.var(0) + 1e-9)
        selected = np.argsort(scores)[::-1][:min(args.n_features, X.shape[1])]
        mean = X[:, selected].mean(0)
        std = X[:, selected].std(0) + 1e-8
        lda = SimpleLDA(shrinkage=args.shrinkage)
        lda.fit((X[:, selected] - mean) / std, y)
        prediction = lda.predict((test_X[:, selected] - mean) / std)

        stop_recall = float(np.mean(prediction[test_y == 0] == 0))
        walk_recall = float(np.mean(prediction[test_y == 1] == 1))
        row = {
            "held_out": held_out,
            "n_train": int(len(y)),
            "n_test": int(len(test_y)),
            "accuracy": float(np.mean(prediction == test_y)),
            "balanced_accuracy": (stop_recall + walk_recall) / 2,
            "stop_recall": stop_recall,
            "walk_recall": walk_recall,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    metric_names = ["accuracy", "balanced_accuracy", "stop_recall", "walk_recall"]
    macro = {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows], ddof=1)),
        }
        for metric in metric_names
    }
    result = {
        "protocol": "8-fold strict leave-one-session-out",
        "subject": "sub-01",
        "sessions": sessions,
        "channels": CHANNELS,
        "window_seconds": args.window,
        "step_seconds": args.step,
        "context_seconds": args.context,
        "n_features": args.n_features,
        "shrinkage": args.shrinkage,
        "session_macro": macro,
        "folds": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# sub-01 ses-01..ses-08 Strict LOSO", "",
        "Held-out session feature selection, normalization ve model fit islemlerine katilmadi.", "",
        f"Konfigurasyon: 9 kanal, window={args.window}s, step={args.step}s, "
        f"context={args.context}s, n_features={args.n_features}, shrinkage={args.shrinkage}.", "",
        "| Held-out | N train | N test | Accuracy | Balanced acc. | STOP recall | WALK recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['held_out']} | {row['n_train']} | {row['n_test']} | "
            f"{row['accuracy']*100:.2f}% | {row['balanced_accuracy']*100:.2f}% | "
            f"{row['stop_recall']*100:.2f}% | {row['walk_recall']*100:.2f}% |"
        )
    lines.extend(["", "## Session-macro", ""])
    for metric in metric_names:
        lines.append(
            f"- {metric}: {macro[metric]['mean']*100:.2f}% "
            f"(session SD {macro[metric]['std']*100:.2f} puan)"
        )
    lines.extend(["", f"Toplam sure: {result['elapsed_seconds']:.2f} saniye.", ""])
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["session_macro"], indent=2))


if __name__ == "__main__":
    main()
