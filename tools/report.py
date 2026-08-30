#!/usr/bin/env python3
"""Derived statistics for the evaluator sweep: correlations, commit rate, and the slices.

    python tools/report.py --data dataset

`run_evaluators.py --report-only` prints the headline table. This prints what that table does
not: the tie-rate and score-scale correlations, accuracy-when-committed, and accuracy sliced by
domain, criterion, level contrast and agent model.

It makes no model calls and needs no credentials. Everything comes from three shipped files:
`*.results.json` carries the per-pair scores keyed by `id`, the blind dataset carries
`context.domain` and `criterion_name`, and the answers file carries `levels` and `agent`. The
join is on `id`.

Spearman is computed here rather than imported so this script has no dependency beyond the
standard library -- rank the two columns, then Pearson on the ranks.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

# Every tool puts the repo root on `sys.path` at module level, whether or not its imports are
# lazy today -- `tests/test_tools_run_as_scripts.py` pins that invariant, so adding a
# `from src.metaeval ...` import here later cannot break running this as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GAPS = ("bad-ok", "ok-good", "bad-good")


def _ranks(xs: list[float]) -> list[float]:
    """Ranks, averaging ties -- which matters here: several rows share a tie rate."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def gap_of(levels: list[str]) -> str:
    """The level contrast, order-independent -- `levels` is slot-ordered, the gap is not."""
    s = set(levels or [])
    for g in GAPS:
        if s == set(g.split("-")):
            return g
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="dataset",
                    help="the data release root; supplies the defaults below "
                         "(default: dataset)")
    ap.add_argument("--dataset", help="the blind eval_dataset.json "
                                     "(default: <data>/benchmark/eval_dataset.json)")
    ap.add_argument("--answers", help="default: eval_answers.json beside --dataset")
    ap.add_argument("--results", help="directory of *.results.json "
                                      "(default: <data>/evaluations)")
    args = ap.parse_args()

    root = Path(args.data)
    dataset = Path(args.dataset) if args.dataset else root / "benchmark" / "eval_dataset.json"
    results = Path(args.results) if args.results else root / "evaluations"
    if not dataset.is_file():
        print(f"no dataset at {dataset}; pass --data <release root> or --dataset <file>")
        return 2
    if not results.is_dir():
        print(f"no results directory at {results}; pass --data <release root> or --results <dir>")
        return 2

    ds = json.loads(dataset.read_text())
    ans_path = Path(args.answers) if args.answers else dataset.with_name("eval_answers.json")
    ans = {a["id"]: a for a in json.loads(ans_path.read_text())}

    # id -> the four axes the paper slices on.
    axis = {
        e["id"]: {
            "domain": (e.get("context") or {}).get("domain", "?"),
            "criterion": e.get("criterion_name", "?"),
            "gap": gap_of((ans.get(e["id"]) or {}).get("levels")),
            "agent": (ans.get(e["id"]) or {}).get("agent", "?"),
        }
        for e in ds
    }

    rows = []
    for f in sorted(results.glob("*.results.json")):
        label = f.name[: -len(".results.json")]
        recs = json.loads(f.read_text())
        runs = [(r["id"], run) for r in recs for run in r.get("runs", [])]
        if not runs:
            continue
        scored = [(i, x) for i, x in runs if x.get("score_1") is not None]
        committed = [(i, x) for i, x in scored if x.get("higher_score") is not None]
        vals = {x[k] for _, x in runs for k in ("score_1", "score_2") if x.get(k) is not None}

        slices: dict[str, dict[str, tuple[int, int]]] = {}
        for name in ("domain", "criterion", "gap", "agent"):
            acc = collections.Counter()
            tot = collections.Counter()
            for i, x in runs:
                key = axis.get(i, {}).get(name, "?")
                tot[key] += 1
                acc[key] += 1 if x.get("correct") else 0
            slices[name] = {k: (acc[k], tot[k]) for k in tot}

        rows.append({
            "label": label,
            "acc": sum(1 for _, x in runs if x.get("correct")) / len(runs) * 100,
            "tie": sum(1 for _, x in scored if x.get("higher_score") is None)
                   / len(runs) * 100,
            "errs": len(runs) - len(scored),
            "commit": len(committed) / len(runs) * 100,
            "acc_committed": (sum(1 for _, x in committed if x.get("correct"))
                              / len(committed) * 100) if committed else float("nan"),
            "lvls": len(vals),
            "slices": slices,
        })

    if not rows:
        print(f"no *.results.json in {results}")
        return 2
    rows.sort(key=lambda r: -r["acc"])

    print(f"\n{len(rows)} evaluators x {len(ds)} pairs\n")
    print(f"  {'evaluator':24}{'acc':>7}{'tie%':>7}{'errs':>6}{'commit%':>9}"
          f"{'acc|commit':>12}{'lvls':>6}")
    print("  " + "-" * 71)
    for r in rows:
        print(f"  {r['label']:24}{r['acc']:7.1f}{r['tie']:7.1f}{r['errs']:6d}"
              f"{r['commit']:9.1f}{r['acc_committed']:12.1f}{r['lvls']:6d}")

    acc = [r["acc"] for r in rows]
    print("\n  CORRELATIONS WITH ACCURACY (Spearman, n = %d)" % len(rows))
    print(f"    tie rate                 rho = {spearman([r['tie'] for r in rows], acc):+.3f}")
    print(f"    distinct score values    rho = {spearman([float(r['lvls']) for r in rows], acc):+.3f}")

    c = sorted(r["acc_committed"] for r in rows if r["acc_committed"] == r["acc_committed"])
    worst = min(rows, key=lambda r: r["acc_committed"])
    rest = [r["acc_committed"] for r in rows if r["label"] != worst["label"]]
    print("\n  ACCURACY WHEN COMMITTED (ties and non-answers excluded)")
    print(f"    all {len(c)}:  mean {statistics.mean(c):.1f}, range {min(c):.1f}-{max(c):.1f}")
    print(f"    excluding {worst['label']} ({worst['acc_committed']:.1f}): "
          f"mean {statistics.mean(rest):.1f}, range {min(rest):.1f}-{max(rest):.1f}")
    print("    A near-constant commit accuracy beside a 48-point accuracy spread is what makes")
    print("    the ranking a RESOLUTION ranking rather than a correctness one.")

    for name, title in (("domain", "DOMAIN"), ("criterion", "CRITERION"),
                        ("gap", "LEVEL CONTRAST"), ("agent", "AGENT MODEL")):
        present = {k for r in rows for k in r["slices"][name]}
        # Contrasts read narrow -> wide, which is the axis they are reported on; the rest
        # have no natural order, so alphabetical.
        keys = [g for g in GAPS if g in present] if name == "gap" else sorted(present)
        print(f"\n  ACCURACY BY {title}")
        print("    " + f"{'evaluator':24}" + "".join(f"{k[:14]:>16}" for k in keys))
        for r in rows:
            cells = []
            for k in keys:
                hit, tot = r["slices"][name].get(k, (0, 0))
                cells.append(f"{hit / tot * 100:15.1f}%" if tot else f"{'-':>16}")
            print("    " + f"{r['label']:24}" + "".join(cells))
        means = []
        for k in keys:
            hits = sum(r["slices"][name].get(k, (0, 0))[0] for r in rows)
            tots = sum(r["slices"][name].get(k, (0, 0))[1] for r in rows)
            means.append(f"{hits / tots * 100:15.1f}%" if tots else f"{'-':>16}")
        print("    " + f"{'-- mean over evaluators':24}" + "".join(means))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
