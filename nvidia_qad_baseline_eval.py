#NVIDIA provides python lm_eval_hf.py for LLM evaluation
#This script aims to evaluate the performance of QAD vs BF16 and PTQ on AIME25 and GPQA-D
#Doubts/Risks: Since LLM is 1.5B, benchmarks might be too complex?

#Citation QAD paper about evaluation:
#take top 10 checkpoints by lowest validation loss, 
#then pick whichever performs best on average across the benchmark suite

#average over multiple sampling runs per problem rather than a single pass@1: 
# 48 runs for AIME24/AIME25, 12 for LiveCodeBench, 20 for GPQA-D, 5 for IFEval 
# (for the Llama/9B/AceReason models); 16 runs for AIME25, 8 for LiveCodeBench-v5, 
# 8 for GPQA-D, 5 for AA-LCR/SciCode (for Nemotron 3 Nano).

#Sampling settings: T=0.6, top-p=0.95 for the first group; T=1.0, top-p=1.0 for Nemotron 3 Nano (take same setting for 1.5B model)


#!/usr/bin/env python3

"""
run_kaggle_qad_eval.py

Baseline vs QAD accuracy comparison. Pure Python version -- no .sh file,
callable directly from a Kaggle notebook cell or the command line.

Usage:
    python run_kaggle_qad_eval.py \
        --baseline_model meta-llama/Llama-3.2-1B-Instruct \
        --qad_model /kaggle/working/qad-output \
        --mode cheap

Or from a notebook:
    from run_kaggle_qad_eval import run_comparison
    run_comparison(
        baseline_model="meta-llama/Llama-3.2-1B-Instruct",
        qad_model="/kaggle/working/qad-output",
        mode="cheap",
    )
"""

import argparse
import subprocess
import sys
from pathlib import Path

from compare_results import load_score  # reuse the parsing logic

CHEAP_TASKS = ["gsm8k"]
FULL_TASKS = ["gsm8k", "aime24", "aime25", "minerva_math500",
              "gpqa_diamond_generative_n_shot", "livecodebench", "ifeval"]


def run_lm_eval(model_path: str, tag: str, task: str, out_dir: Path,
                 batch_size: int, quant_cfg: str | None, eval_dir: Path):
    out_path = out_dir / f"{tag}_{task}.json"
    
    # Define generation arguments
    gen_kwargs = "temperature=1.0,top_p=1.0" #adapt to sampling settings of Nemetron 3 Nano as this is the closest model in terms of size to 1.5B model
    
    cmd = [
        sys.executable, "lm_eval_hf.py",
        "--model", "hf",
        "--tasks", task,
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--batch_size", str(batch_size),
        "--output_path", str(out_path),
        # Pass the arguments via --gen_kwargs
        "--gen_kwargs", gen_kwargs
    ]
    
    if quant_cfg and tag == "qad":
        cmd += ["--quant_cfg", quant_cfg]

    print(f"=== [{tag}] task={task} ===")
    subprocess.run(cmd, cwd=eval_dir, check=True)
    return out_path


def run_comparison(baseline_model: str, qad_model: str, mode: str = "cheap",
                    batch_size: int = 8, out_dir: str = "./eval_results",
                    eval_dir: str = ".", quant_cfg: str | None = None):
    out_dir = Path(out_dir)
    eval_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = CHEAP_TASKS if mode == "cheap" else FULL_TASKS

    print(f">>> BASELINE: {baseline_model}")
    for task in tasks:
        run_lm_eval(baseline_model, "baseline", task, out_dir, batch_size, quant_cfg, eval_dir)

    print(f">>> QAD: {qad_model}")
    for task in tasks:
        run_lm_eval(qad_model, "qad", task, out_dir, batch_size, quant_cfg, eval_dir) #from library lm-eval-hf.py from NVIDIA Model Optimizer

    print(">>> Summary:")
    header = f"{'Task':35s} {'Baseline':>10s} {'QAD':>10s} {'Delta':>10s}"
    print(header)
    print("-" * len(header))
    deltas = []
    for task in tasks:
        base = load_score(out_dir / f"baseline_{task}.json")
        qad = load_score(out_dir / f"qad_{task}.json")
        delta = (qad - base) if (base is not None and qad is not None) else None
        if delta is not None:
            deltas.append(delta)
        base_s = f"{base*100:.2f}" if base is not None else "N/A"
        qad_s = f"{qad*100:.2f}" if qad is not None else "N/A"
        delta_s = f"{delta*100:+.2f}" if delta is not None else "N/A"
        print(f"{task:35s} {base_s:>10s} {qad_s:>10s} {delta_s:>10s}")
    if deltas:
        avg = sum(deltas) / len(deltas)
        print("-" * len(header))
        print(f"{'Average delta':35s} {'':>10s} {'':>10s} {avg*100:>+10.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--qad_model", default="/kaggle/working/qad-output")
    ap.add_argument("--mode", choices=["cheap", "full"], default="cheap")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="./eval_results")
    ap.add_argument("--eval_dir", default=".", help="path to Model-Optimizer/examples/llm_eval")
    ap.add_argument("--quant_cfg", default=None)
    args = ap.parse_args()

    run_comparison(
        baseline_model=args.baseline_model,
        qad_model=args.qad_model,
        mode=args.mode,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        eval_dir=args.eval_dir,
        quant_cfg=args.quant_cfg,
    )