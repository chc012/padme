#!/usr/bin/env python3
"""Score one evaluator against the benchmark.

    python tools/score_one.py --dataset data/my_run/eval_dataset.json \
                              --model fireworks_ai/accounts/fireworks/models/gpt-oss-120b

Each trajectory is scored **alone** and the preference is recovered from which score is
higher -- see `src/metaeval/scoring.py` for why that asymmetry is the experiment rather than
an oversight.

**For more than one evaluator use `tools/run_evaluators.py`.** It scores a whole roster over
one shared worker pool, applies per-lane rate limits, and checkpoints every call so a long
sweep can be interrupted. This script is the single-model path: the thing to reach for when
iterating on a prompt or checking that a new provider answers at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metaeval.scoring import (  # noqa: E402
    _score_model, compute_metrics, evaluate, load_pairs, print_metrics, save_json)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="blind dataset from tools/build_eval_dataset.py")
    ap.add_argument("--answers", default=None,
                    help="default: the answers file beside --dataset "
                         "(eval_dataset.json -> eval_answers.json)")
    ap.add_argument("--model", required=True, help="evaluator, in litellm form")
    ap.add_argument("--output", default="data/results.json")
    ap.add_argument("--runs", "-k", type=int, default=1,
                    help="independent scoring runs per response (default 1). At temperature "
                         "0, k>1 measures residual provider nondeterminism, not sampling "
                         "spread -- see SCORE_CALL.")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    dataset = load_pairs(a.dataset, a.answers)
    print(f"Loaded {len(dataset)} labelled pair(s) from {a.dataset}")
    print(f"Model: {a.model}  runs/response: {a.runs}")

    results = evaluate(dataset, lambda entry, key: _score_model(entry, key, a.model),
                       num_runs=a.runs, max_workers=a.workers)
    save_json(results, a.output)

    metrics = compute_metrics(results)
    metrics_path = str(Path(a.output).parent / "metrics.json")
    save_json(metrics, metrics_path)

    print_metrics(metrics)
    print(f"\nResults -> {a.output}")
    print(f"Metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
