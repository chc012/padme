"""Run-directory layout, and the end-of-run summary.

One module so the writer and every reader share one set of constants. A writer and reader
that disagree about the layout fail *silently* -- as "0 trajectory files" rather than as an
error -- which is why `run_pipeline.Store` imports the names from here instead of restating
them.

Everything here reads a run directory or a run blob. Nothing here computes a cost, an
accuracy or a comparison; those live in `tools/report.py`.
"""

from __future__ import annotations

import json
from pathlib import Path


# Files that live in a run directory but are not trajectories.
#
# `generated_instructions.json` was missing, and the test that guards this list enumerated only
# three pool files so it could not catch the omission. Harmless in effect -- the structural
# `"turns" not in d` check rejects it anyway -- but this list is what keeps the report from
# parsing multi-megabyte files to discover they are not trajectories, and the comment called it
# a contract.
NON_TRAJECTORY = (".pairs.json", "judge_votes.json", "dataset.json", "summary.json",
                  # written by tools/run_pipeline.py
                  "rejects.json", "pair_index.json", "pipeline_run.json",
                  "usage_checkpoints.json",
                  "generated_instructions.json")

# Trajectories live in their own subdirectory. At the full tau3 space a flat run directory
# holds 2,250 trajectory files beside 1,125 pair files and six bookkeeping files, which is
# unnavigable; the subdirectory also means a stray `*.json` in the run root cannot be mistaken
# for a trajectory. `run_pipeline.Store.traj_dir` writes here, and a test pins that the two
# agree -- a writer and reader that disagree about the layout fail *silently*, as "0 trajectory
# files" rather than as an error.
TRAJECTORY_SUBDIR = "trajectories"

# Pairs get the same treatment, and for the same reason. At the full tau3 space a run holds
# 1,673 pair files -- one per draw, not per cell -- which is what made the run root unreadable
# even after the trajectories moved out.
PAIRS_SUBDIR = "pairs"


def pair_files(runs: Path) -> list[Path]:
    """Every `*.pairs.json` in a run directory, in either layout, each cell exactly once.

    Eleven call sites globbed the run root independently. That is the same shape of duplication
    that once made a trajectory read silently report 0 tokens for a directory holding 7,020,191,
    so this is the one implementation and they all route through it. Flat runs -- anything written before
    the subdirectories existed -- keep working unmigrated.

    **De-duplicated by filename, with the subdirectory winning.** The first version concatenated
    the two layouts, which double-counts any cell present in both -- and both is a reachable
    state, not a hypothetical: a resume writes to `pairs/` while an older
    run wrote the deliberately identical filename to the root, and a migration done with `cp`
    rather than `mv` leaves both. Every reader that returns a
    *list* rather than a dict keyed by pair id inherits the duplication -- `Store.write_dataset`
    and `Store.n_pairs` among them -- and a doubled `dataset.json` reads exactly like
    `filter_retries > 0` when it is not.

    Filename-keyed rather than `trajectory_files`' either/or, because pair filenames are
    cell-scoped: a half-migrated directory still reads completely, which either/or would not
    give.
    """
    sub = runs / PAIRS_SUBDIR
    by_name = {f.name: f for f in sorted(runs.glob("*.pairs.json"))}
    if sub.is_dir():
        by_name.update({f.name: f for f in sorted(sub.glob("*.pairs.json"))})
    return [by_name[n] for n in sorted(by_name)]


def trajectory_files(runs: Path) -> list[Path]:
    """Every trajectory in a run directory, in either layout.

    Runs produced before the subdirectory existed are flat, and they are not migrated. So both
    layouts are read, and this is the one implementation of that: four call sites globbed the
    run root independently, and the last time one of them reimplemented a trajectory read it
    silently reported 0 tokens for a directory holding 7,020,191.
    """
    sub = runs / TRAJECTORY_SUBDIR
    if sub.is_dir():
        found = sorted(f for f in sub.glob("*.json")
                       if not f.name.endswith(NON_TRAJECTORY))
        if found:
            return found
    return sorted(f for f in runs.glob("*.json")
                  if not f.name.endswith(NON_TRAJECTORY))


def pooled_run(runs: Path) -> dict | None:
    """`pipeline_run.json` if this directory came from the pool, else None.

    Its presence changes what may be *printed*, not just what is added. A pooled run has no
    per-stage wall clock -- once pair 7's instruction generation overlaps pair 2's judging,
    no interval belongs to a stage -- so the columns that report one must be suppressed
    rather than left to print a number that no longer means what its header says.
    """
    p = runs / "pipeline_run.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None



def _count_pairs_note(pooled: dict) -> int:
    """Pairs built, from the run record rather than the filesystem."""
    return (pooled.get("kept") or 0) + (pooled.get("rejected") or 0)



def _title(text: str, note: str = "") -> None:
    print("\n" + "=" * 92)
    print(f"  {text}" + (f"{note:>{max(1, 90 - len(text))}}" if note else ""))
    print("=" * 92)



def _print_run(pooled: dict | None) -> None:
    """How the run was configured and what it produced.

    Config travels with the numbers because a pipeline clock without its concurrency is not
    reproducible, and because concurrency moves result stability -- temperature 0 is not
    bitwise reproducible on Fireworks and MoE routing depends on batch composition.
    """
    if pooled is None:
        return
    cfg = pooled.get("config") or {}
    u = pooled.get("usage") or {}
    cum = pooled.get("cumulative") or {}
    inv = cum.get("n_invocations") or 1
    # Two simulations per pair, by construction: one per steering level in the cell's gap.
    su = cum.get("stage_usage") or pooled.get("stage_usage") or {}
    n_sims = (pooled.get("n_tasks") or 0) * 2 if (su.get("trajectory") or {}) else 0
    per = cum.get("per_datapoint") or pooled.get("per_datapoint") or {}
    chains = sorted(v["service_s"] for v in per.values() if v.get("service_s"))

    _title("RUN", f"{inv} invocations" if inv > 1 else "")

    def row(k, v, extra=""):
        print(f"  {k:28s} {v:>12s}   {extra}".rstrip())

    row("tasks", str(pooled.get("n_tasks", 0)))
    row("kept / rejected / failed",
        f"{pooled.get('kept', 0)} / {pooled.get('rejected', 0)} / {pooled.get('failed', 0)}")
    n_pairs = _count_pairs_note(pooled)
    if n_sims:
        mult = f"{n_sims / n_pairs:.1f} per pair" if n_pairs else ""
        row("simulations", str(n_sims), mult)
    # Cumulative once a directory was built by more than one invocation: a resume that redoes
    # nothing has an empty meter, and quoting that as the cost understates it by everything.
    # Cumulative wall clock is a sum of separate sessions, so it is labelled by invocations
    # rather than presented as one elapsed figure.
    if inv > 1:
        row("tokens (all invocations)", f"{cum.get('total_tokens', 0):,}",
            f"latest alone {u.get('total_tokens', 0):,}")
        row("wall (summed sessions)", f"{cum.get('session_wall_s', 0):.0f}s")
    else:
        row("tokens", f"{u.get('total_tokens', 0):,}")
        row("elapsed", f"{u.get('wall_s', 0):.0f}s", f"at {cfg.get('workers')} workers")
    if chains:
        mid = chains[len(chains) // 2]
        row("chain per pair", f"{mid:.1f}s",
            f"median ({chains[0]:.1f}s - {chains[-1]:.1f}s), concurrency-independent")
    row("workers / sim / serverless",
        f"{cfg.get('workers')} / {cfg.get('sim_concurrency')} / "
        f"{cfg.get('serverless_concurrency')}")
    row("seed", str(cfg.get("seed")))
    if cfg.get("filter_retries"):
        row("filter-retries", str(cfg["filter_retries"]))
    if u.get("n_error"):
        row("errored calls", str(u["n_error"]))


def _print_retention(pooled: dict | None) -> None:
    """Retention per stratum, all three axes in one table.

    **Reported, never enforced.** Unequal retention is the measurement: 100% with high
    evaluator accuracy means a criterion too easy to discriminate, low retention with low
    accuracy means one that is hard to label *and* hard to score. `draws/kept` counts attempts
    rather than cells, so it keeps rising under `--filter-retries` even as retention approaches
    1.0 -- which is what stops the finding vanishing into the retry budget.
    """
    if pooled is None:
        return
    strata = pooled.get("strata") or {}
    if not strata:
        return
    _title("RETENTION", "reported, never enforced")
    print(f"  {'axis':11s} {'stratum':24s} {'kept':>8s} {'rate':>7s} {'draws/kept':>11s}")
    for axis in ("criterion", "domain", "gap"):
        for name, r in (strata.get(axis) or {}).items():
            cells = r.get("cells") or r.get("attempts") or 0
            rate = f"{r['retention']:.0%}" if r.get("retention") is not None else "-"
            mult = (f"{r['attempts_per_kept']:.2f}x"
                    if r.get("attempts_per_kept") is not None else "none kept")
            kept = f"{r.get('kept', 0)}/{cells}"
            print(f"  {axis:11s} {name:24s} {kept:>8s} {rate:>7s} {mult:>11s}")



def print_summary(run: dict) -> None:
    """Every block of a finished run, from the run blob alone.

    The single entry point for "print a pooled run". A run that finishes silently is
    unpleasant to run, and the funnel counts are how you know it was sane.

    **The blob is built first and printed from, rather than printed from live objects**, so
    what you read on the terminal is exactly what is on disk in `pipeline_run.json`.
    """
    _print_run(run)
    _print_retention(run)
