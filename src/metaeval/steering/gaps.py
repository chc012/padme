"""Assign each cell one of the three level gaps, with balanced counts.

A criterion has three steering levels, so a cell could in principle be simulated at all
three and yield three pairs: two narrow gaps (bad-ok, ok-good) and one wide (bad-good).
**It is simulated at two, and this module decides which two.** That is a 3:2 saving on the
expensive stage, and it is only sound because the gap assignment carries no information
about the task -- see below.

**Stratify what we report; randomize what we want to average out.**

Gap size is a *reported axis*: evaluator accuracy as a function of how close the two
trajectories are is a headline result. Drawing it at random would leave the cells uneven --
over 12 cells a fair draw can easily land 8/15/13 across three gaps -- and an uneven cell is
a weaker estimate for no gain. A balanced rotation gives exact counts and costs nothing,
because the assignment is arbitrary *with respect to the task*: which gap a cell gets carries
no information about that cell.

The agent stays randomized (`pick_agent`) because model identity is a nuisance variable to be
averaged over, not a curve to report.

The rotation is seeded and derived from the cell's own identity, so a run is reproducible from
the stored record rather than from whatever order the loop happened to take. `run_pipeline`
stores each assignment in `pair_index.json` and reads it back rather than re-cycling, because
a narrowed re-run would otherwise reassign gaps and silently change the dataset's shape.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..schema.pairs import PAIR_LEVELS

# (bad, ok), (ok, good), (bad, good) -- two narrow, one wide.
GAPS: tuple[tuple[str, str], ...] = tuple(PAIR_LEVELS)
GAP_NAMES = {("bad", "ok"): "narrow_low", ("ok", "good"): "narrow_high",
             ("bad", "good"): "wide"}


@dataclass(frozen=True)
class GroupPlan:
    """One cell: what to run, and which two of its three levels to run it at."""

    domain: str
    criterion: str
    task_id: str
    task_index: int
    gap: tuple[str, str]

    @property
    def gap_name(self) -> str:
        return GAP_NAMES[self.gap]

    def as_dict(self) -> dict:
        return {"domain": self.domain, "criterion": self.criterion,
                "task_id": self.task_id, "task_index": self.task_index,
                "gap": list(self.gap), "gap_name": self.gap_name}


def balanced_cycle(n: int, options: list, seed: int = 42, salt: str = "") -> list:
    """`n` picks from `options`, as evenly as the counts allow, order shuffled.

    Even counts come first, then the order is shuffled so the assignment does not
    correlate with whatever order the caller enumerated groups in -- otherwise every
    first-listed domain would systematically get the same gap.

    With `n` not divisible by `len(options)` the remainder is spread by taking a
    shuffled prefix, so the cells differ by at most one.
    """
    if not options:
        raise ValueError("no options to choose from")
    rng = random.Random(f"{seed}:{salt}")
    full, rest = divmod(n, len(options))
    picks = list(options) * full
    if rest:
        extra = list(options)
        rng.shuffle(extra)
        picks += extra[:rest]
    rng.shuffle(picks)
    return picks


def plan_groups(domains: list[str], criteria: list[str], task_ids: dict[str, list[str]],
                *, seed: int = 42) -> list[GroupPlan]:
    """One `GroupPlan` per (domain, criterion, task), with gaps balanced.

    `task_ids` maps a domain to the task ids to use for it, so task selection stays the
    caller's business -- this module only decides *which gap a cell gets*.

    Gaps are balanced **within each criterion** rather than only overall. Balancing globally
    can leave one criterion lopsided, and accuracy by (criterion, gap) is a cell we report.
    """
    plans: list[GroupPlan] = []
    for criterion in criteria:
        cells = [(d, t, i) for d in domains
                 for i, t in enumerate(task_ids.get(d, []))]
        gaps = balanced_cycle(len(cells), list(GAPS), seed=seed, salt=criterion)
        for (domain, task_id, task_index), gap in zip(cells, gaps):
            plans.append(GroupPlan(domain=domain, criterion=criterion, task_id=task_id,
                                   task_index=task_index, gap=gap))
    return plans


def gap_counts(plans: list[GroupPlan]) -> dict[str, int]:
    out: dict[str, int] = {name: 0 for name in GAP_NAMES.values()}
    for p in plans:
        out[p.gap_name] += 1
    return out
