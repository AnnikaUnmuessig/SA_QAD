#!/usr/bin/env python3
"""
compare_results.py

Parses the JSON result files produced by lm_eval_hf.py for a "baseline"
run and a "qad" run, and prints a side-by-side accuracy table with the
delta (QAD - baseline) per task.

Usage:
    python compare_results.py --results_dir ./eval_results \
        --tasks minerva_math500 aime24 aime25 gpqa_diamond_generative_n_shot livecodebench ifeval
"""

import argparse
import json
from pathlib import Path


# Primary metric key to report per task family. lm-eval-harness result
# dicts are nested as results[task][metric_name]; the exact metric name
# varies by task, so we try a few common ones in order.
CANDIDATE_METRICS = ["acc,none", "acc", "exact_match,none", "exact_match",
                     "pass@1,none", "pass@1", "inst_level_strict_acc,none"]


def find_results_json(run_dir: Path):
    """lm_eval stamps a timestamp onto the output filename and nests it
    under <model_name_sanitized>/, so the exact path can't be predicted --
    glob for it and take the most recently written match.
    """
    if not run_dir.exists():
        return None
    matches = list(run_dir.glob("**/results_*.json"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_score(run_dir: Path):
    """run_dir is the directory that was passed as --output_path for this run."""
    path = find_results_json(Path(run_dir))
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", {})
    if not results:
        return None
    task_results = next(iter(results.values()))
    for metric in CANDIDATE_METRICS:
        if metric in task_results:
            return task_results[metric]
    for v in task_results.values():
        if isinstance(v, (int, float)):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for task in args.tasks:
        baseline_score = load_score(results_dir / f"baseline_{task}")
        qad_score = load_score(results_dir / f"qad_{task}")
        delta = (qad_score - baseline_score) if (baseline_score is not None and qad_score is not None) else None
        rows.append((task, baseline_score, qad_score, delta))

    # ---- print table ----
    header = f"{'Task':35s} {'Baseline':>10s} {'QAD':>10s} {'Delta':>10s}"
    print(header)
    print("-" * len(header))
    for task, base, qad, delta in rows:
        base_s = f"{base*100:.2f}" if base is not None else "N/A"
        qad_s = f"{qad*100:.2f}" if qad is not None else "N/A"
        delta_s = f"{delta*100:+.2f}" if delta is not None else "N/A"
        print(f"{task:35s} {base_s:>10s} {qad_s:>10s} {delta_s:>10s}")

    valid = [d for _, _, _, d in rows if d is not None]
    if valid:
        avg = sum(valid) / len(valid)
        print("-" * len(header))
        print(f"{'Average delta':35s} {'':>10s} {'':>10s} {avg*100:>+10.2f}")


if __name__ == "__main__":
    main()