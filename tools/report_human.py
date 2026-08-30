#!/usr/bin/env python3
"""Human validation: does the synthetic label agree with the annotator majority?

    python tools/report_human.py --data dataset

Reports, per filter arm, how often the steering label matches the majority vote of three
annotators; then panel reliability -- unanimity, pairwise agreement, Cohen's kappa and
Krippendorff's alpha -- and agreement conditioned on annotator confidence.

The arms are nested. Every one of the 150 sampled pairs is in D0; D1 drops the pairs judge1
rejected; D2 drops the ones judge2 then rejected. So a rising number across the arms is the
filter cascade buying label validity, and the D0 column is what the labels are worth unfiltered.

Annotators are identified only as `annotator_N` within a panel, in an order that carries no
information about who they were. Nothing here needs their identity: every statistic is either
per-panel or over unordered pairs of raters.

No model calls, no credentials. Alpha and kappa are implemented here so this needs nothing
beyond the standard library.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = (("D0", "all sampled"), ("D1", "judge1 kept"), ("D2", "both judges kept"))


def majority(votes: dict[str, int]) -> int | None:
    """The reference label. None when three raters split with no majority (impossible at n=3
    on a binary choice, but the guard keeps this honest for other panel sizes)."""
    top, n = collections.Counter(votes.values()).most_common(1)[0]
    return top if n * 2 > len(votes) else None


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] / n * cb[k] / n for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def krippendorff_alpha(units: list[list[int]]) -> float:
    """Nominal alpha over units that may have differing numbers of raters.

    Computed from observed and expected disagreement rather than from a coincidence matrix,
    which is the same quantity and easier to read: Do averages within-unit mismatch, De is
    the mismatch expected if every rating were drawn from the pooled marginal.
    """
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    num = den = 0.0
    for u in units:
        m = len(u)
        pairs = sum(1 for x, y in itertools.permutations(u, 2) if x != y)
        num += pairs / (m - 1)
        den += m
    do = num / den
    pooled = collections.Counter(v for u in units for v in u)
    total = sum(pooled.values())
    de = 1 - sum((c / total) ** 2 for c in pooled.values())
    de *= total / (total - 1)
    return 1 - do / de if de else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="dataset",
                    help="the data release root (default: dataset)")
    args = ap.parse_args()
    H = Path(args.data) / "human_study"
    if not H.is_dir():
        print(f"no human_study/ under {args.data}; pass --data <release root>")
        return 2

    key = {r["id"]: r for r in json.loads((H / "answer_key.json").read_text())}
    panels = sorted(p.name for p in H.glob("panel_*"))
    votes: dict[str, dict[str, int]] = collections.defaultdict(dict)
    conf: dict[str, dict[str, int]] = collections.defaultdict(dict)
    panel_of: dict[str, str] = {}
    for p in panels:
        for f in sorted((H / p).glob("annotator_*.json")):
            for r in json.loads(f.read_text()):
                votes[r["id"]][f.stem] = r["judge_preference"]
                conf[r["id"]][f.stem] = r.get("confidence")
                panel_of[r["id"]] = p

    arms = {
        "D0": list(key),
        "D1": [i for i in key if key[i]["filter_outcome"] != "judge1_reject"],
        "D2": [i for i in key if key[i]["filter_outcome"] == "kept"],
    }

    print(f"\n{len(key)} pairs, {len(panels)} panels, "
          f"{sum(1 for p in panels for _ in (H / p).glob('annotator_*.json'))} annotators\n")

    print("  LABEL ACCURACY vs ANNOTATOR MAJORITY")
    print(f"    {'arm':6}{'n':>6}{'accuracy':>11}{'alpha(label,maj)':>19}")
    for arm, note in ARMS:
        ids = arms[arm]
        hit = sum(1 for i in ids if majority(votes[i]) == key[i]["correct_response"])
        a = krippendorff_alpha([[key[i]["correct_response"], majority(votes[i])] for i in ids])
        print(f"    {arm:6}{len(ids):6d}{hit / len(ids) * 100:10.1f}%{a:19.3f}"
              f"   {note}")
    print("    Rising accuracy across the arms is the veto cascade removing pairs whose")
    print("    steering direction annotators did not see.")

    print("\n  PANEL RELIABILITY (all sampled pairs)")
    for p in panels:
        ids = [i for i in key if panel_of[i] == p]
        raters = sorted({r for i in ids for r in votes[i]})
        unan = sum(1 for i in ids if len(set(votes[i].values())) == 1)
        a = krippendorff_alpha([list(votes[i].values()) for i in ids])
        print(f"    {p}: n={len(ids)}  unanimous {unan}/{len(ids)} "
              f"({unan / len(ids) * 100:.0f}%)  alpha {a:+.3f}")
        for x, y in itertools.combinations(raters, 2):
            both = [i for i in ids if x in votes[i] and y in votes[i]]
            va, vb = [votes[i][x] for i in both], [votes[i][y] for i in both]
            agree = sum(1 for u, v in zip(va, vb) if u == v) / len(both) * 100
            print(f"        {x} vs {y}:  agree {agree:5.1f}%   kappa {cohen_kappa(va, vb):+.3f}")
    allids = list(key)
    unan = sum(1 for i in allids if len(set(votes[i].values())) == 1)
    print(f"    pooled: unanimous {unan}/{len(allids)} ({unan / len(allids) * 100:.0f}%)  "
          f"alpha {krippendorff_alpha([list(votes[i].values()) for i in allids]):+.3f}")

    scale = sorted({c for d in conf.values() for c in d.values() if c is not None})
    if scale:
        # The interface recorded 0 / 50 / 100; the three points mean guess / leaning /
        # certain. Divide by 50 to read these as the 0-1-2 scale.
        print(f"\n  CONFIDENCE (interface scale {scale} = guess / leaning / certain)")
        for arm, _ in ARMS:
            vals = [c for i in arms[arm] for c in conf[i].values() if c is not None]
            m = statistics.mean(vals)
            print(f"    {arm:6} n={len(vals):4d}  mean {m:6.1f}  ({m / 50:.2f} on 0-2)")
        certain = [i for i in allids
                   if all(c == scale[-1] for c in conf[i].values() if c is not None)]
        rest = [i for i in allids if i not in set(certain)]
        for label, ids in (("all raters certain", certain), ("anyone unsure", rest)):
            if not ids:
                continue
            a = krippendorff_alpha([list(votes[i].values()) for i in ids])
            hit = sum(1 for i in ids if majority(votes[i]) == key[i]["correct_response"])
            print(f"    {label:20} n={len(ids):4d}  alpha {a:+.3f}  "
                  f"label accuracy {hit / len(ids) * 100:.1f}%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
