"""Tests for gap stratification.

The rule these encode: **stratify what we report, randomize what we want to average
out.** Gap size is a reported axis, so its cells must come out even; agent identity is a
nuisance variable, so it stays randomized (`pick_agent`, tested in test_steering.py).
"""

from __future__ import annotations

import pytest

from src.metaeval.steering.gaps import (
    GAPS,
    GroupPlan,
    balanced_cycle,
    gap_counts,
    plan_groups,
)

DOMAINS = ["airline", "telecom", "banking_knowledge", "retail"]
CRITERIA = ["friendliness", "communication_clarity", "task_resolution"]
TASKS = {d: [f"{d}-t{i}" for i in range(3)] for d in DOMAINS}


def test_the_target_shape_is_36_groups():
    plans = plan_groups(DOMAINS, CRITERIA, TASKS)
    assert len(plans) == 4 * 3 * 3 == 36


def test_gaps_are_exactly_balanced_overall():
    """The whole point of stratifying rather than drawing at random."""
    plans = plan_groups(DOMAINS, CRITERIA, TASKS)
    assert gap_counts(plans) == {"narrow_low": 12, "narrow_high": 12, "wide": 12}


def test_gaps_are_balanced_within_each_criterion():
    """Balancing only globally can leave a criterion lopsided, and per-criterion
    accuracy by gap is a cell we report."""
    plans = plan_groups(DOMAINS, CRITERIA, TASKS)
    for criterion in CRITERIA:
        sub = [p for p in plans if p.criterion == criterion]
        assert gap_counts(sub) == {"narrow_low": 4, "narrow_high": 4, "wide": 4}, criterion


def test_the_assignment_is_reproducible_from_the_seed():
    a = plan_groups(DOMAINS, CRITERIA, TASKS, seed=7)
    b = plan_groups(DOMAINS, CRITERIA, TASKS, seed=7)
    assert [p.as_dict() for p in a] == [p.as_dict() for p in b]


def test_a_different_seed_gives_a_different_assignment_but_the_same_balance():
    a = plan_groups(DOMAINS, CRITERIA, TASKS, seed=1)
    b = plan_groups(DOMAINS, CRITERIA, TASKS, seed=2)
    assert [p.gap for p in a] != [p.gap for p in b]
    assert gap_counts(a) == gap_counts(b)


def test_gap_does_not_correlate_with_position_in_the_domain_list():
    """Without a shuffle, every first-listed domain would draw the same gap, tying a
    reported axis to an arbitrary enumeration order."""
    plans = plan_groups(DOMAINS, CRITERIA, TASKS)
    for criterion in CRITERIA:
        first = [p.gap_name for p in plans
                 if p.criterion == criterion and p.domain == DOMAINS[0]]
        assert len(set(first)) > 1 or len(first) == 1, (criterion, first)


def test_cells_differ_by_at_most_one_when_n_is_not_divisible():
    """5 groups over 3 gaps must be 2/2/1, never 3/1/1."""
    picks = balanced_cycle(5, list(GAPS), seed=42)
    counts = sorted(picks.count(g) for g in GAPS)
    assert counts == [1, 2, 2]
    assert len(picks) == 5


def test_balanced_cycle_rejects_an_empty_option_list():
    with pytest.raises(ValueError, match="no options"):
        balanced_cycle(3, [])


def test_every_gap_is_one_of_the_three_real_pairs():
    plans = plan_groups(DOMAINS, CRITERIA, TASKS)
    assert {p.gap for p in plans} == set(GAPS)


# --- the gap travels with the plan, and only two levels are ever simulated ---

def test_a_plan_serialises_its_gap_and_its_name():
    """The assignment has to survive to disk, because it is not recomputable: a re-run over
    a narrowed set of cells would re-cycle and hand the same cell a different gap."""
    plan = GroupPlan(domain="airline", criterion="friendliness", task_id="t0",
                     task_index=0, gap=("bad", "good"))
    d = plan.as_dict()
    assert d["gap"] == ["bad", "good"]
    assert d["gap_name"] == "wide"


def test_every_planned_gap_is_one_of_the_three():
    """A gap outside `GAPS` would have no `gap_name`, so `as_dict` would raise on it -- but
    only at write time, after the simulations were paid for."""
    for plan in plan_groups(DOMAINS, CRITERIA, TASKS):
        assert plan.gap in GAPS
        assert plan.gap_name in {"narrow_low", "narrow_high", "wide"}


def test_each_level_is_used_about_equally_across_the_run():
    """Every level appears in exactly two of the three gaps, so balanced gaps give balanced
    level counts -- and a lopsided level count would mean some steering instruction is barely
    exercised."""
    import collections

    plans = plan_groups(DOMAINS, CRITERIA, TASKS)
    levels = collections.Counter(l for p in plans for l in p.gap)
    assert set(levels) == {"bad", "ok", "good"}
    assert max(levels.values()) - min(levels.values()) <= 2, levels
    # Two levels per cell, never three. This is the 3:2 saving on the expensive stage.
    assert sum(levels.values()) == 2 * len(plans)


def test_the_runner_simulates_exactly_the_assigned_gap():
    """Guards the saving where it is actually spent.

    `run_pipeline` simulates `cell.levels`, and `Cell.levels` is the gap -- so the pair of
    levels chosen here is the pair of simulations paid for. A third level would be a 50% cost
    increase on the dominant stage, and nothing in the output would say so.
    """
    from tools.run_pipeline import Cell

    cell = Cell(domain="airline", criterion="friendliness", task_index=0, task_id="t0",
                gap=("ok", "good"), agent="a")
    assert cell.levels == ("ok", "good")
    assert len(cell.levels) == 2
