"""Tests for `tools/run_pipeline.py`.

None of the four pre-existing orchestrators has a test, because each fuses its control flow
to litellm. The pool takes its four model-facing stages as injected callables specifically
so the state machine can be driven here: every test below runs the real resume logic against
real files in `tmp_path`, with fake stages, and spends nothing.

What matters most is the resume behaviour, because getting it wrong is expensive rather than
merely wrong: a trajectory pair is ~195k tokens, so re-running P2 to recover a judge call
burns ~97% of a task's cost for nothing.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from tools.run_pipeline import (
    Cell,
    Limits,
    Outcome,
    Stages,
    Store,
    run_task,
    strata,
)

JUDGES = {"judge1": "model-a", "judge2": "model-b"}


def _cell(domain="airline", criterion="friendliness", idx=0, gap=("ok", "good")):
    return Cell(domain=domain, criterion=criterion, task_index=idx, task_id="t-1",
                gap=gap, agent="gpt-oss-120b")


class FakeTraj:
    """Stands in for a `Trajectory`. Only needs to survive save/load and carry turns."""

    def __init__(self, level: str):
        self.level = level
        self.turns = []


def _stages(*, instructions=None, trajectory=None, build_pair=None, judge=None,
            sides=(1, 1), intent=1, calls=None):
    """Fake stages. `sides` is what (judge1, judge2) return; `intent` is the label."""
    calls = calls if calls is not None else []

    def _instr(cell):
        calls.append(("P1", cell.key))
        return {lvl: f"be {lvl}" for lvl in ("bad", "ok", "good")}

    def _traj(cell, level, body):
        calls.append(("P2", cell.key, level))
        return FakeTraj(level), 1

    def _pair(cell, trajs):
        calls.append(("P3", cell.key))
        return {"id": f"pair-{cell.key}", "correct_response": intent,
                "criterion_name": cell.criterion}

    order = {"judge1": sides[0], "judge2": sides[1]}

    def _judge(name, pair, cell):
        calls.append(("judge", name, pair["id"], cell.key))
        return {"side": order[name], "a_was": 1, "attempts": 1, "reasoning": "because"}

    return Stages(instructions=instructions or _instr,
                  trajectory=trajectory or _traj,
                  build_pair=build_pair or _pair,
                  judge=judge or _judge), calls


def _patch_schema(monkeypatch):
    """`run_task` imports save/load_trajectory from the schema package; the fake trajectory
    is not a pydantic model, so those two are stubbed to plain json."""
    import src.metaeval.schema as schema

    monkeypatch.setattr(schema, "save_trajectory",
                        lambda t, p: p.write_text(json.dumps({"level": t.level})),
                        raising=False)
    monkeypatch.setattr(schema, "load_trajectory",
                        lambda p: FakeTraj(json.loads(p.read_text())["level"]),
                        raising=False)


def _run(tmp_path, monkeypatch, **kw):
    _patch_schema(monkeypatch)
    cell = kw.pop("cell", None) or _cell()
    stages, calls = _stages(**kw)
    store = Store(tmp_path)
    out = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    return out, calls, store


# --- the happy path ---

def test_a_pair_both_judges_back_is_kept(tmp_path, monkeypatch):
    out, calls, store = _run(tmp_path, monkeypatch, sides=(1, 1), intent=1)
    assert out.state == "kept" and out.phase == "P5"
    assert [c[0] for c in calls] == ["P1", "P2", "P2", "P3", "judge", "judge"]


def test_only_the_two_levels_the_gap_needs_are_simulated(tmp_path, monkeypatch):
    """Two simulations per pair, not three. The third level bought pairs no human read."""
    out, calls, _ = _run(tmp_path, monkeypatch, cell=_cell(gap=("bad", "good")))
    levels = [c[2] for c in calls if c[0] == "P2"]
    assert levels == ["bad", "good"], "exactly the assigned gap, in worse->better order"


def test_every_artifact_lands_on_disk(tmp_path, monkeypatch):
    out, _, store = _run(tmp_path, monkeypatch)
    assert store.instr_path.is_file()
    assert store.pair_path(_cell()).is_file()
    assert store.votes_path.is_file()
    assert store.index_path.is_file()
    for lvl in ("ok", "good"):
        assert store.traj_path(_cell(), lvl).is_file()


# --- rejection: a completed task, not a failure ---

def test_judge1_disagreeing_rejects_and_stops_before_judge2(tmp_path, monkeypatch):
    """The cascade's entire cost saving: judge2 never sees judge1's rejects."""
    out, calls, store = _run(tmp_path, monkeypatch, sides=(2, 1), intent=1)
    assert out.state == "rejected" and out.phase == "P4"
    judged = [c[1] for c in calls if c[0] == "judge"]
    assert judged == ["judge1"], "judge2 must not run on a pair judge1 rejected"


def test_judge2_disagreeing_rejects_at_p5(tmp_path, monkeypatch):
    out, calls, store = _run(tmp_path, monkeypatch, sides=(1, 2), intent=1)
    assert out.state == "rejected" and out.phase == "P5"
    assert [c[1] for c in calls if c[0] == "judge"] == ["judge1", "judge2"]


def test_a_reject_is_recorded_with_why(tmp_path, monkeypatch):
    """Rejects go to their own file: they are data about the filter, not lost work."""
    out, _, store = _run(tmp_path, monkeypatch, sides=(2, 1), intent=1)
    blob = json.loads(store.rejects_path.read_text())
    row = blob["rejected"][0]
    assert row["rejected_by"] == "judge1"
    assert row["judge_side"] == 2 and row["intent"] == 1
    assert row["criterion"] == "friendliness"


def test_a_reject_is_never_retried(tmp_path, monkeypatch):
    """Re-running must not re-judge a rejected pair -- it is a completed task."""
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    cell = _cell()
    stages, calls = _stages(sides=(2, 1), intent=1)
    run_task(cell, stages, store, JUDGES, log=lambda m: None)
    calls.clear()
    second = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert second.state == "rejected"
    assert calls == [], "the whole task resumed from disk, including the vote"


# --- resume ---

def test_second_run_repeats_nothing(tmp_path, monkeypatch):
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    stages, calls = _stages()
    run_task(cell, stages, store, JUDGES, log=lambda m: None)
    calls.clear()
    out = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert out.state == "kept"
    assert calls == []
    assert set(out.resumed) >= {"P1", "P3", "P2:ok", "P2:good"}


def test_a_failed_second_simulation_does_not_discard_the_first(tmp_path, monkeypatch):
    """The real hazard, and the reason each trajectory is saved as it completes.

    The standalone runner gates *all* saving on the whole group succeeding, so a failure on
    the second level throws away the first level's trajectory too -- ~97k tokens already
    paid for. Here the first survives on disk and the resume re-runs only the second.
    """
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()          # gap ok->good, so levels are ok, good

    state = {"fail_good": True}
    calls: list = []

    def flaky_traj(c, level, body):
        calls.append(("P2", c.key, level))
        if level == "good" and state["fail_good"]:
            raise RuntimeError("DEPLOYMENT_SCALING_UP")
        return FakeTraj(level), 1

    stages, _ = _stages(trajectory=flaky_traj, calls=calls)
    first = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert first.state == "failed" and first.phase == "P2"
    assert store.traj_path(cell, "ok").is_file(), "the level that SUCCEEDED must persist"
    assert not store.traj_path(cell, "good").is_file()

    state["fail_good"] = False
    calls.clear()
    second = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert second.state == "kept"
    assert [c[2] for c in calls if c[0] == "P2"] == ["good"], \
        "only the failed level re-runs; the other is reused from disk"


def test_an_existing_pair_short_circuits_the_simulations(tmp_path, monkeypatch):
    """A `PairwiseEntry` embeds both trajectories in full, so once the pair exists the
    trajectory files are byproducts. Checking P2 first cost a real ~40k-token simulation in
    the smoke run -- and left the pair's embedded copy disagreeing with the file on disk.
    """
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    stages, calls = _stages()
    run_task(cell, stages, store, JUDGES, log=lambda m: None)

    store.traj_path(cell, "ok").unlink()            # byproduct, not the data
    calls.clear()
    out = run_task(cell, stages, store, JUDGES, log=lambda m: None)

    assert out.state == "kept"
    assert calls == [], "nothing re-runs: the pair already holds both trajectories"


def test_a_failure_at_p5_does_not_re_buy_the_trajectories(tmp_path, monkeypatch):
    """~97% of a task's cost is its two trajectories. A judge failure must cost one call."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()

    state = {"fail": True}
    calls: list = []

    def flaky_judge(name, pair, cell):
        calls.append(("judge", name, pair["id"], cell.key))
        if name == "judge2" and state["fail"]:
            raise RuntimeError("rate limit")
        return {"side": 1, "a_was": 1, "attempts": 1, "reasoning": ""}

    stages, _ = _stages(judge=flaky_judge, calls=calls)
    first = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert first.state == "failed" and first.phase == "P5"

    state["fail"] = False
    calls.clear()
    second = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert second.state == "kept"
    assert not any(c[0] == "P2" for c in calls), "trajectories must be reused"
    assert [c[1] for c in calls if c[0] == "judge"] == ["judge2"], "judge1's vote reused"


def test_an_empty_instruction_is_not_a_completed_phase(tmp_path, monkeypatch):
    """A failed generation writes empty bodies. Treating that as done would simulate a
    group whose steering text is the empty string."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    store.put_instructions(cell, {"ok": "", "good": "", "bad": ""}, {})
    assert store.instructions_for(cell) is None

    stages, calls = _stages()
    run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert any(c[0] == "P1" for c in calls), "must regenerate, not simulate empty steering"


def test_the_pair_file_is_never_rewritten(tmp_path, monkeypatch):
    """Pair ids are uuids, so rebuilding a pair would mint a new id and orphan its votes."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    stages, calls = _stages()
    run_task(cell, stages, store, JUDGES, log=lambda m: None)
    before = store.pair_path(cell).read_text()

    calls.clear()
    run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert store.pair_path(cell).read_text() == before
    assert not any(c[0] == "P3" for c in calls)


# --- failures ---

@pytest.mark.parametrize("phase", ["P1", "P2", "P3"])
def test_a_stage_failure_is_recorded_not_raised(tmp_path, monkeypatch, phase):
    """One task's failure must not take down the pool."""
    _patch_schema(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("provider down")

    kw = {"P1": {"instructions": boom}, "P2": {"trajectory": boom},
          "P3": {"build_pair": boom}}[phase]
    stages, _ = _stages(**kw)
    out = run_task(_cell(), stages, Store(tmp_path), JUDGES, log=lambda m: None)
    assert out.state == "failed" and out.phase == phase
    assert "provider down" in out.detail


def test_a_failure_is_written_to_the_index(tmp_path, monkeypatch):
    """The index is the resume state, so a failure has to be visible in it."""
    _patch_schema(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("nope")

    stages, _ = _stages(instructions=boom)
    store = Store(tmp_path)
    run_task(_cell(), stages, store, JUDGES, log=lambda m: None)
    row = json.loads(store.index_path.read_text())["airline.friendliness.t0"]
    assert row["state"] == "failed" and row["phase"] == "P1"


def test_trajectory_attempt_count_is_kept(tmp_path, monkeypatch):
    """A pair that only looks slow because it retried must be distinguishable."""
    _patch_schema(monkeypatch)

    def retried(cell, level, body):
        return FakeTraj(level), 3

    stages, _ = _stages(trajectory=retried)
    out = run_task(_cell(), stages, Store(tmp_path), JUDGES, log=lambda m: None)
    assert out.attempts == {"P2:ok": 3, "P2:good": 3}


# --- the index as the join key, and gap preservation ---

def test_the_index_records_the_group_to_pair_join(tmp_path, monkeypatch):
    """Stage 1's usage is keyed by group, stages 4-5 by pair id. Without this mapping the
    per-datapoint roll-up cannot span the two stages the cost claim rests on."""
    out, _, store = _run(tmp_path, monkeypatch)
    row = json.loads(store.index_path.read_text())["airline.friendliness.t0"]
    assert row["pair_id"] == out.pair_id
    assert row["gap"] == ["ok", "good"]


def test_the_index_preserves_the_gap_for_a_later_run(tmp_path, monkeypatch):
    """`balanced_cycle` balances over the combos of one invocation, so a narrowed re-run
    would otherwise reassign a gap that trajectories were already generated for."""
    out, _, store = _run(tmp_path, monkeypatch, cell=_cell(gap=("bad", "ok")))
    row = json.loads(store.index_path.read_text())["airline.friendliness.t0"]
    assert row["gap"] == ["bad", "ok"]


# --- votes merge across judges without losing either ---

def test_votes_merge_by_pair_and_judge(tmp_path, monkeypatch):
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    for idx in (0, 1):
        cell = _cell(idx=idx)
        stages, _ = _stages()
        run_task(cell, stages, store, JUDGES, log=lambda m: None)
    blob = json.loads(store.votes_path.read_text())
    keys = {(v["id"], v["judge"]) for v in blob["votes"]}
    assert len(keys) == 4, "two pairs x two judges, nothing overwritten"
    assert set(blob["judges"]) == {"judge1", "judge2"}


# --- concurrency ---

def test_the_sim_semaphore_caps_concurrent_simulations(tmp_path, monkeypatch):
    """The limit that actually binds: every simulation needs the user simulator, the only
    dedicated deployment, so its replica count is the pipeline's ceiling."""
    _patch_schema(monkeypatch)
    from concurrent.futures import ThreadPoolExecutor

    live, peak, lock = [0], [0], threading.Lock()

    def slow_traj(cell, level, body):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        import time
        time.sleep(0.02)
        with lock:
            live[0] -= 1
        return FakeTraj(level), 1

    store = Store(tmp_path)
    limits = Limits.of(sim=2, serverless=8)
    cells = [_cell(idx=i) for i in range(8)]

    def work(c):
        stages, _ = _stages(trajectory=slow_traj)
        return run_task(c, stages, store, JUDGES, limits=limits, log=lambda m: None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(work, cells))

    assert all(o.state == "kept" for o in outs)
    assert peak[0] <= 2, f"sim concurrency exceeded its cap: {peak[0]}"


def test_unlimited_limits_do_not_block(tmp_path, monkeypatch):
    """`Limits()` with no semaphores is the test default and must be a pass-through."""
    _patch_schema(monkeypatch)
    stages, _ = _stages()
    out = run_task(_cell(), stages, Store(tmp_path), JUDGES, limits=Limits(),
                   log=lambda m: None)
    assert out.state == "kept"


def test_concurrent_tasks_do_not_lose_each_others_writes(tmp_path, monkeypatch):
    """Instructions, votes, rejects and the index are shared read-modify-write files. Under
    a pool, several workers finishing a phase at once would each write from a stale read."""
    _patch_schema(monkeypatch)
    from concurrent.futures import ThreadPoolExecutor

    store = Store(tmp_path)
    cells = [_cell(idx=i) for i in range(12)]

    def work(c):
        stages, _ = _stages()
        return run_task(c, stages, store, JUDGES, log=lambda m: None)

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(work, cells))

    index = json.loads(store.index_path.read_text())
    instr = json.loads(store.instr_path.read_text())["records"]
    votes = json.loads(store.votes_path.read_text())["votes"]
    assert len(index) == 12, f"index lost rows: {len(index)}"
    assert len(instr) == 12, f"instructions lost rows: {len(instr)}"
    assert len(votes) == 24, f"votes lost rows: {len(votes)}"


# --- strata: reported, never enforced ---

def _outcome(criterion, state, gap=("ok", "good"), domain="airline", attempt=1):
    return Outcome(cell=_cell(domain=domain, criterion=criterion, gap=gap), state=state,
                   attempt=attempt)


def test_strata_report_retention_without_capping_anything():
    """The measured pattern: friendliness keeps everything, task_resolution does not. Both
    are findings, so the pool records the ratio and generates no quota."""
    outs = ([_outcome("friendliness", "kept") for _ in range(12)]
            + [_outcome("task_resolution", "kept") for _ in range(7)]
            + [_outcome("task_resolution", "rejected") for _ in range(5)])
    by_c = strata(outs)["criterion"]
    assert by_c["friendliness"] == {"cells": 12, "attempts": 12, "kept": 12, "rejected": 0,
                                    "failed": 0, "retention": 1.0,
                                    "attempts_per_kept": 1.0}
    assert by_c["task_resolution"]["retention"] == pytest.approx(0.583, abs=1e-3)
    assert by_c["task_resolution"]["attempts_per_kept"] == pytest.approx(1.71, abs=1e-2)


def test_retries_raise_the_cost_multiplier_without_hiding_it_in_retention():
    """**The finding must survive `--filter-retries`.**

    Retries push retention toward 1.0 by construction, so if the cost multiplier were also
    computed per cell, a criterion needing three draws per pair would read identically to one
    that passes first time -- and the differential-retention result the paper rests on would
    disappear into the retry budget. Retention counts cells; the multiplier counts draws.
    """
    # Two criteria, both end up fully kept -- but one needed three draws each.
    easy = [_outcome("friendliness", "kept", attempt=1) for _ in range(6)]
    hard = [_outcome("task_resolution", "kept", attempt=3) for _ in range(6)]
    by_c = strata(easy + hard)["criterion"]

    assert by_c["friendliness"]["retention"] == 1.0
    assert by_c["task_resolution"]["retention"] == 1.0, "retries flatten retention"
    # ...and here is where the difference is still visible.
    assert by_c["friendliness"]["attempts_per_kept"] == 1.0
    assert by_c["task_resolution"]["attempts_per_kept"] == 3.0
    assert by_c["task_resolution"]["cells"] == 6, "6 cells"
    assert by_c["task_resolution"]["attempts"] == 18, "but 18 draws paid for"


def test_strata_cover_domain_and_gap_too():
    outs = [_outcome("friendliness", "kept", gap=("bad", "good"), domain="retail"),
            _outcome("friendliness", "rejected", gap=("bad", "ok"), domain="telecom")]
    st = strata(outs)
    assert set(st) == {"criterion", "domain", "gap"}
    assert st["gap"]["bad-good"]["kept"] == 1
    assert st["domain"]["telecom"]["rejected"] == 1


def test_strata_with_nothing_kept_reports_none_not_zero():
    """A stratum the filter cannot confirm at all is a finding about steering. Reporting
    `attempts_per_kept` as 0 would read as free rather than impossible."""
    st = strata([_outcome("task_resolution", "rejected")])
    row = st["criterion"]["task_resolution"]
    assert row["retention"] == 0.0 and row["attempts_per_kept"] is None


def test_strata_of_nothing_is_empty():
    assert all(v == {} for v in strata([]).values())


def test_failures_are_counted_apart_from_rejections():
    """A reject is a filter decision; a failure is a lost task. Pooling them would make a
    broken deployment look like a discriminating filter."""
    st = strata([_outcome("friendliness", "rejected"), _outcome("friendliness", "failed")])
    row = st["criterion"]["friendliness"]
    assert row["rejected"] == 1 and row["failed"] == 1 and row["kept"] == 0


# --- Store, in isolation ---

def test_store_tolerates_a_corrupt_shared_file(tmp_path):
    """A half-written file from a kill must not crash the next run."""
    store = Store(tmp_path)
    store.votes_path.write_text("{not json")
    assert store.votes_for("pair-1") == {}


def test_store_filenames_match_the_existing_convention(tmp_path):
    """Downstream report tools glob these names, so they are part of the contract."""
    store, cell = Store(tmp_path), _cell()
    assert store.traj_path(cell, "ok").name == "airline.friendliness.t0.ok.json"
    assert store.pair_path(cell).name == "airline.friendliness.t0.pairs.json"


# --- the tau3 accounting bridge ---
#
# Trajectory calls happen inside tau3 and never pass through `timed_call`, so this function is
# the *only* thing that puts the pipeline's most expensive stage into the meter. If it
# silently records nothing, the per-datapoint roll-up shows a pair costing a few thousand
# tokens when it cost ~97,000, and nothing else in the system would notice.

def _turn(role, prompt=None, completion=None, secs=None, extra_none=False):
    if extra_none:
        return type("T", (), {"role": role, "extra": None})()
    extra = {}
    if prompt is not None:
        extra["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion}
    if secs is not None:
        extra["generation_time_seconds"] = secs
    return type("T", (), {"role": role, "extra": extra})()


def _traj_with(turns):
    return type("Tr", (), {"turns": turns})()


def _fold(turns, cell=None, level="ok"):
    from src.metaeval.usage import UsageMeter
    from tools.run_pipeline import _record_trajectory_usage

    m = UsageMeter().start()
    _record_trajectory_usage(m, _traj_with(turns), cell or _cell(), level)
    return m


def test_trajectory_usage_is_folded_into_the_meter():
    m = _fold([_turn("assistant", 3000, 500, 4.0), _turn("user", 900, 60, None)])
    s = m.summary()
    assert s["prompt_tokens"] == 3900 and s["completion_tokens"] == 560
    assert s["n_calls"] == 2


def test_folded_records_carry_the_chain_key_and_stage():
    """Keyed by group like every other phase, or the per-datapoint chain fragments."""
    m = _fold([_turn("assistant", 100, 10, 1.0)])
    r = m.records[0]
    assert r.role == "trajectory"
    assert r.item_id == _cell().key
    assert r.label == "ok", "the level, so bad/ok/good spend is separable"


def test_turns_without_usage_are_skipped_not_recorded_as_zero():
    """Tool results involve no LLM call. Recording them would inflate the call count and
    drag the mean tokens-per-call down toward zero."""
    m = _fold([_turn("assistant", 100, 10, 1.0), _turn("tool"), _turn("tool")])
    assert m.summary()["n_calls"] == 1


def test_assistant_latency_is_kept_and_user_latency_is_absent():
    """tau3 times assistant messages only, so the trajectory stage's service time is a
    floor. It must not be invented for the turns that lack it."""
    m = _fold([_turn("assistant", 100, 10, 7.5), _turn("user", 50, 5, None)])
    lat = sorted(r.latency_s for r in m.records)
    assert lat == [0.0, 7.5]
    assert m.summary()["service_s"] == pytest.approx(7.5)


def test_folding_survives_a_trajectory_with_no_turns():
    assert _fold([]).summary()["n_calls"] == 0
    from src.metaeval.usage import UsageMeter
    from tools.run_pipeline import _record_trajectory_usage
    m = UsageMeter().start()
    _record_trajectory_usage(m, type("Tr", (), {})(), _cell(), "ok")   # no .turns at all
    assert m.summary()["n_calls"] == 0


def test_folding_survives_a_none_or_odd_extra():
    """`extra` is a passthrough of whatever tau3 put on the message; it is not guaranteed
    to be a dict, and an accounting helper must not take down the simulation that produced
    the data."""
    m = _fold([_turn("assistant", extra_none=True), _turn("assistant", 100, 10, 1.0)])
    assert m.summary()["n_calls"] == 1


def test_folded_reasoning_is_marked_unavailable_not_zero():
    """tau3 keeps no per-message reasoning trace, so the split cannot be derived here.
    Saying 0 would claim the agent did no reasoning."""
    m = _fold([_turn("assistant", 100, 50, 1.0)])
    assert m.records[0].reasoning_source == "unavailable"


# --- build_cells: the gap must survive a re-run ---

def _patch_task_lookup(monkeypatch):
    """`build_cells` resolves tasks and agents through tau3; neither is what is under test."""
    import src.metaeval.sources.tau2_source as ts

    fake_task = type("T", (), {"id": "task-9"})()
    monkeypatch.setattr(ts, "get_task", lambda d, index=0, task_id=None: fake_task)
    monkeypatch.setattr(ts, "pick_agent", lambda d, t, c, s: "gpt-oss-120b")


def test_build_cells_assigns_a_gap_to_every_cell(tmp_path, monkeypatch):
    from tools.run_pipeline import build_cells

    _patch_task_lookup(monkeypatch)
    cells = build_cells(["airline"], ["friendliness", "task_resolution"], [0, 1], 42,
                        Store(tmp_path))
    assert len(cells) == 4
    for c in cells:
        assert len(c.gap) == 2 and c.agent == "gpt-oss-120b"


def test_build_cells_preserves_a_gap_already_on_disk(tmp_path, monkeypatch):
    """`balanced_cycle` balances over the combos of *one invocation*, so a narrowed re-run
    would otherwise reassign a gap that trajectories were already generated for -- silently
    changing the dataset's shape."""
    from tools.run_pipeline import build_cells

    _patch_task_lookup(monkeypatch)
    store = Store(tmp_path)
    first = build_cells(["airline"], ["friendliness"], [0, 1, 2], 42, store)
    for c in first:
        store.update_index(c, state="kept")

    # A narrower re-run: one task index instead of three.
    again = build_cells(["airline"], ["friendliness"], [1], 42, store)
    prior = {c.key: c.gap for c in first}
    assert again[0].gap == prior[again[0].key], "the recorded gap must win"


def test_build_cells_is_deterministic_for_the_same_inputs(tmp_path, monkeypatch):
    from tools.run_pipeline import build_cells

    _patch_task_lookup(monkeypatch)
    a = build_cells(["airline"], ["friendliness"], [0, 1, 2], 42, Store(tmp_path / "a"))
    b = build_cells(["airline"], ["friendliness"], [0, 1, 2], 42, Store(tmp_path / "b"))
    assert [c.gap for c in a] == [c.gap for c in b]


def test_the_judge_stage_receives_the_cell_on_the_resume_path(tmp_path, monkeypatch):
    """The bug this signature exists to prevent.

    `run_task` short-circuits P3 when the pair is already on disk, so anything the judge needs
    that was only computed *during* `build_pair` is absent exactly when resuming. The previous
    design carried the accounting key in a `pair_id -> group_key` dict populated by
    `build_pair`; on resume that dict was empty and the judge fell back to the pair uuid, so
    P4/P5 recorded under a different key than P1/P2 for the same datapoint -- the precise
    fragmentation the key was unified to fix, on the path the tool is built for.

    Passing the cell makes it structurally impossible: it is in scope at the call site.
    """
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()

    seen: list = []

    def judge(name, pair, c):
        seen.append((name, c.key))
        return {"side": 1, "a_was": 1, "attempts": 1, "reasoning": ""}

    stages, _ = _stages(judge=judge)
    run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert seen == [("judge1", cell.key), ("judge2", cell.key)]

    # Now the resume path: pair on disk, votes cleared, so only P4/P5 run.
    store.votes_path.unlink()
    seen.clear()
    out = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert out.state == "kept"
    assert "P3" in out.resumed, "the pair came from disk, so build_pair did not run"
    assert seen == [("judge1", cell.key), ("judge2", cell.key)], \
        "the cell must still reach the judge when build_pair is skipped"


def test_a_retried_simulation_is_visible_in_the_accounting(tmp_path, monkeypatch):
    """The stage that is 80% of spend reported its failures nowhere.

    `run_steered_resilient` discards a failed simulation's conversation and re-runs, so the
    tokens are unknowable -- but the time is not, and `_record_trajectory_usage` folded in only
    the survivor with `attempt=1`. Net effect: `n_error` stayed 0 and `item_stage_matrix`'s
    `retries` was structurally blind to exactly the case its docstring promises to
    distinguish, "a pair that only looks slow because it retried".
    """
    from src.metaeval.usage import UsageMeter
    from tools.run_pipeline import _record_trajectory_usage

    m = UsageMeter().start()
    # Two failed attempts, then the survivor on attempt 3.
    for att in (1, 2):
        m.record_error(RuntimeError("DEPLOYMENT_SCALING_UP"), model="tau3:sim",
                       latency_s=31.0, role="trajectory", label="ok",
                       item_id=_cell().key, attempt=att)
    _record_trajectory_usage(m, _traj_with([_turn("assistant", 3000, 400, 5.0)]),
                             _cell(), "ok", attempt=3)

    row = m.item_stage_matrix()[_cell().key]
    assert row["n_calls"] == 3
    assert row["n_error"] == 2, "the lost simulations are counted"
    assert row["retries"] == 2, "and are distinguishable from a genuinely slow pair"
    assert row["service_s"] == pytest.approx(67.0), "62s of wasted time is not hidden"
    assert row["stage_tokens"]["trajectory"] == 3400, "only the survivor's tokens are known"


def test_a_first_try_simulation_records_no_retries(tmp_path):
    """The common case must not look retried."""
    from src.metaeval.usage import UsageMeter
    from tools.run_pipeline import _record_trajectory_usage

    m = UsageMeter().start()
    _record_trajectory_usage(m, _traj_with([_turn("assistant", 100, 10, 1.0)]),
                             _cell(), "ok", attempt=1)
    assert m.item_stage_matrix()[_cell().key]["retries"] == 0


# --- cross-tool compatibility, which Store's docstring claims ---

def test_dataset_json_is_written_for_the_report_tools(tmp_path, monkeypatch):
    """`dataset.json` is the run directory's public output: it is what
    `tools/build_eval_dataset.py` reads, and the pool once did not write one.

    The failure mode is why this is pinned rather than left to the reader. Every consumer of
    a run directory guards with `is_file()`, because a run may legitimately be mid-flight --
    so a missing dataset produces an *empty kept set* rather than an error, and every count
    downstream reads 0 with nothing anywhere saying why.
    """
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    for idx in (0, 1):
        cell = _cell(idx=idx)
        stages, _ = _stages()
        run_task(cell, stages, store, JUDGES, log=lambda m: None)

    store.write_dataset()
    ds = json.loads((tmp_path / "dataset.json").read_text())
    assert isinstance(ds, list) and len(ds) == 2, "a flat concatenation of the pair files"
    assert {e["id"] for e in ds} == {f"pair-{_cell(idx=i).key}" for i in (0, 1)}


def test_the_kept_set_is_derivable_from_what_the_pool_writes(tmp_path, monkeypatch):
    """The join the report tools actually perform: dataset.json x judge_votes.json."""
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    stages, _ = _stages(sides=(1, 1), intent=1)
    run_task(_cell(), stages, store, JUDGES, log=lambda m: None)
    store.write_dataset()

    ds = json.loads((tmp_path / "dataset.json").read_text())
    votes = json.loads(store.votes_path.read_text())["votes"]
    intent = {e["id"]: e["correct_response"] for e in ds}
    kept = {v["id"] for v in votes
            if v["judge"] == "judge1" and v["side"] == intent.get(v["id"])}
    assert len(kept) == 1


def test_write_dataset_tolerates_a_corrupt_pair_file(tmp_path, monkeypatch):
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    stages, _ = _stages()
    run_task(_cell(), stages, store, JUDGES, log=lambda m: None)
    (tmp_path / "broken.pairs.json").write_text("{not json")
    store.write_dataset()
    assert len(json.loads((tmp_path / "dataset.json").read_text())) == 1


def test_the_instructions_file_is_readable_by_the_standalone_runner(tmp_path, monkeypatch):
    """Two producers, one filename, two shapes: the pool writes `records` as a dict keyed by
    group (so it can merge across resumes), the standalone generator writes a list. Iterating
    a dict yields string keys, so `r["domain"]` raised TypeError on a dict-shaped file.
    The reader normalises both, and this pins the shape the pipeline actually writes.
    """
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    stages, _ = _stages()
    run_task(_cell(), stages, store, JUDGES, log=lambda m: None)

    payload = json.loads(store.instr_path.read_text())
    raw = payload["records"]
    records = list(raw.values()) if isinstance(raw, dict) else raw
    assert [r["domain"] for r in records] == ["airline"], "the normalisation the reader does"
    assert records[0]["criterion"] == "friendliness"


# --- robustness: run_task must not raise, and writes must not lose data ---

def test_a_malformed_pair_file_fails_the_task_not_the_run(tmp_path, monkeypatch):
    """`run_task` is documented "Never raises", and `main` catches only KeyboardInterrupt --
    so an escaping exception would kill the process before `pipeline_run.json` is written and
    discard every Outcome already collected. A pair file left in place by an earlier run,
    under the same (deliberately identical) filename, is enough to trigger it.
    """
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    # A pair with no `correct_response` -- previously a bare KeyError out of run_task.
    store.put_pair(cell, {"id": "p-1"})

    stages, _ = _stages()
    out = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert out.state == "failed" and out.phase == "P3"
    assert "KeyError" in out.detail


def test_a_pair_file_holding_a_bare_string_does_not_raise(tmp_path, monkeypatch):
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    store.pair_path(cell).write_text(json.dumps(["not a dict"]))
    stages, _ = _stages()
    out = run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert out.state == "failed"


def test_writes_are_atomic_so_a_kill_cannot_truncate(tmp_path):
    """`write_text` truncates in place and `_read` swallows JSONDecodeError, which together
    are a silent total-loss path: a kill mid-write leaves the file truncated, the next read
    returns `{}`, and the next write persists only the new row. Every prior vote gone, no
    error. `os.replace` means a reader sees either the whole old file or the whole new one.
    """
    import os

    store = Store(tmp_path)
    target = tmp_path / "judge_votes.json"
    Store._write(target, {"votes": [{"id": "a", "judge": "judge1"}]})

    # No temp file survives a successful write.
    assert not list(tmp_path.glob(".*.tmp")), "the temp file must be renamed, not left behind"
    assert json.loads(target.read_text())["votes"][0]["id"] == "a"

    # And the replace is a rename, not a truncate-then-fill.
    inode_before = os.stat(target).st_ino
    Store._write(target, {"votes": [{"id": "b", "judge": "judge1"}]})
    assert os.stat(target).st_ino != inode_before, "replaced, so no window of partial content"


def test_a_truncated_file_does_not_silently_empty_the_merge(tmp_path):
    """The read side still degrades to a default -- that is deliberate, so one corrupt file
    cannot halt a long run -- but the atomic write means the corruption should not arise from
    our own writes in the first place. This pins the pairing rather than the tolerance.
    """
    store = Store(tmp_path)
    store.votes_path.write_text('{"votes": [{"id": "a", "judge": "judge1", "side": 1}]}')
    assert store.votes_for("a") == {"judge1": 1}
    store.votes_path.write_text('{"votes": [{"id": "a", "judge')      # truncated
    assert store.votes_for("a") == {}, "tolerated, not crashed"


def test_the_dry_run_plan_matches_what_resume_actually_does(tmp_path, monkeypatch):
    """`--dry-run` is documented as "what resume would skip", and it applied a laxer rule for
    P1: key-membership, where resume uses `instructions_for`, which rejects a record whose body
    is empty ("An empty body is a failed generation, not a completed phase"). So a blank
    instruction planned as `nothing, complete` and then spent a generator call.
    """
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()

    # A record present but blank -- exactly what a failed generation leaves behind.
    store.put_instructions(cell, {"bad": "", "ok": "", "good": ""}, {})
    assert store.instructions_for(cell) is None, "resume treats this as incomplete"

    from tools.run_pipeline import _dry_run
    _dry_run([cell], store, JUDGES)
    # The real run agrees: it regenerates rather than skipping.
    stages, calls = _stages()
    run_task(cell, stages, store, JUDGES, log=lambda m: None)
    assert any(c[0] == "P1" for c in calls)


def test_the_dry_run_counts_a_complete_instruction_as_done(tmp_path, monkeypatch, capsys):
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    store.put_instructions(cell, {"bad": "a", "ok": "b", "good": "c"}, {})

    from tools.run_pipeline import _dry_run
    _dry_run([cell], store, JUDGES)
    out = capsys.readouterr().out
    assert "P1" not in out.split("would run:")[1], "P1 is not pending when the bodies are real"


# --- a smoke run must not contaminate a real one ---

def test_a_limited_run_on_the_default_gets_its_own_directory():
    """`--limit N` on the default `--out-dir` redirects to `smoke-N/`.

    `pipeline_run.json` accumulates by design, so a smoke run sharing the real directory
    would fold its tokens into `cumulative` and they would be quoted as part of what the
    real dataset cost. Same failure `run_evaluators.py` documents at its own redirect.
    """
    from tools.run_pipeline import _DEFAULT_OUT, smoke_out
    assert smoke_out(_DEFAULT_OUT, 2) == _DEFAULT_OUT / "smoke-2"
    assert smoke_out(_DEFAULT_OUT, 36) == _DEFAULT_OUT / "smoke-36"


def test_a_full_run_on_the_default_is_not_redirected():
    from tools.run_pipeline import _DEFAULT_OUT, smoke_out
    assert smoke_out(_DEFAULT_OUT, None) == _DEFAULT_OUT
    assert smoke_out(_DEFAULT_OUT, 0) == _DEFAULT_OUT, "0 is not a limit"


def test_limit_is_redirected_whatever_the_out_dir_says(tmp_path):
    """**This test used to assert the exemption**, which was safe while an out-dir was a flag
    you had to type: setting one was a deliberate act. In the YAML era `out_dir` is written in
    every config, so it never matched the built-in default, the redirect never fired, and a
    `limit` run wrote its tokens into the real run's accumulating `pipeline_run.json`. A limited
    run is a smoke run whatever the out-dir says."""
    from tools.run_pipeline import smoke_out, _DEFAULT_OUT
    assert smoke_out(tmp_path, 2) == tmp_path / "smoke-2"
    assert smoke_out(_DEFAULT_OUT, 2) == _DEFAULT_OUT / "smoke-2"
    assert smoke_out(tmp_path, None) == tmp_path, "a full run is never redirected"


# --- filter retries: re-draw a rejected pair ---
#
# The fake `build_pair` above returns a CONSTANT pair id, which is fine for a single draw but
# would make a retry look like a resume: `votes_for` would find the previous attempt's votes
# and skip the judges. The real `pairs_from_group` mints a uuid4 per pair, so these fakes mint
# a fresh id per draw too.

def _retry_stages(verdicts, intent=1, calls=None):
    """`verdicts` is what judge1 returns on draw 1, 2, 3...; judge2 always agrees."""
    calls = calls if calls is not None else []
    draw = {"n": 0}

    def _instr(cell):
        calls.append(("P1", cell.key))
        return {lvl: f"be {lvl}" for lvl in ("bad", "ok", "good")}

    def _traj(cell, level, body):
        calls.append(("P2", cell.key, level))
        return FakeTraj(level), 1

    def _pair(cell, trajs):
        draw["n"] += 1
        calls.append(("P3", cell.key, draw["n"]))
        return {"id": f"pair-{cell.key}-d{draw['n']}", "correct_response": intent,
                "criterion_name": cell.criterion}

    def _judge(name, pair, cell):
        calls.append(("judge", name, pair["id"]))
        if name == "judge1":
            i = min(draw["n"], len(verdicts)) - 1
            return {"side": verdicts[i], "a_was": 1, "attempts": 1, "reasoning": "r"}
        return {"side": intent, "a_was": 1, "attempts": 1, "reasoning": "r"}

    return Stages(instructions=_instr, trajectory=_traj, build_pair=_pair,
                  judge=_judge), calls


def _run_retry(tmp_path, monkeypatch, verdicts, retries, intent=1):
    _patch_schema(monkeypatch)
    stages, calls = _retry_stages(verdicts, intent=intent)
    store = Store(tmp_path)
    out = run_task(_cell(), stages, store, JUDGES, log=lambda m: None,
                   filter_retries=retries)
    return out, calls, store


def test_zero_retries_is_exactly_todays_behaviour(tmp_path, monkeypatch):
    """The default. A reject is terminal and nothing is re-drawn."""
    out, calls, _ = _run_retry(tmp_path, monkeypatch, verdicts=[2], retries=0)
    assert out.state == "rejected" and out.attempt == 1
    assert sum(1 for c in calls if c[0] == "P3") == 1, "one draw only"


def test_a_reject_is_redrawn_and_can_be_recovered(tmp_path, monkeypatch):
    """The measured case: rejected on draw 1, passes on draw 2 (55% of the time in practice)."""
    out, calls, _ = _run_retry(tmp_path, monkeypatch, verdicts=[2, 1], retries=1)
    assert out.state == "kept"
    assert out.attempt == 2, "and the attempt number is recorded"
    assert sum(1 for c in calls if c[0] == "P3") == 2


def test_retries_are_bounded(tmp_path, monkeypatch):
    """A cell the filter never accepts must terminate, not loop. 5 of 36 cells passed in
    neither of the two real runs, so this is the common case, not a corner."""
    out, calls, _ = _run_retry(tmp_path, monkeypatch, verdicts=[2, 2, 2, 2], retries=2)
    assert out.state == "rejected"
    assert out.attempt == 3, "1 initial draw + 2 retries"
    assert sum(1 for c in calls if c[0] == "P3") == 3, "and no more"


def test_p1_is_generated_once_and_reused_by_every_draw(tmp_path, monkeypatch):
    """**The choice that makes a retry interpretable.** Reusing the steering text means a
    re-draw is the same datapoint redrawn under an unchanged intended gap. Regenerating it
    would re-roll the intent, making it a different datapoint and erasing the reason the
    hard criteria stay visibly hard."""
    _, calls, _ = _run_retry(tmp_path, monkeypatch, verdicts=[2, 2, 1], retries=2)
    assert sum(1 for c in calls if c[0] == "P1") == 1
    assert sum(1 for c in calls if c[0] == "P2") == 6, "3 draws x 2 levels"


def test_each_draw_gets_its_own_trajectory_and_pair_files(tmp_path, monkeypatch):
    """A rejected draw's artifacts are evidence -- `rejects.json` points at them -- and pair
    ids are uuids, so overwriting would orphan the votes already cast."""
    _, _, store = _run_retry(tmp_path, monkeypatch, verdicts=[2, 1], retries=1)
    traj = tmp_path / "trajectories"
    assert (traj / "airline.friendliness.t0.ok.json").is_file()
    assert (traj / "airline.friendliness.t0.a2.ok.json").is_file()
    # Pairs moved to `pairs/` for the same reason trajectories did: 1,673 of them made the run
    # root unreadable. Every reader routes through `runs.pair_files`, which tries both
    # layouts, so the flat reference runs still read -- that is what the old version of this
    # assertion was protecting, and it is now protected by `pair_files` instead of by layout.
    pairs = tmp_path / "pairs"
    assert (pairs / "airline.friendliness.t0.pairs.json").is_file()
    assert (pairs / "airline.friendliness.t0.a2.pairs.json").is_file()
    assert not list(tmp_path.glob("*.pairs.json")), "nothing loose in the run root"


def test_attempt_one_keeps_the_original_filenames(tmp_path, monkeypatch):
    """Layout compatibility is load-bearing: a retries=0 run must be indistinguishable from
    what the serial tools and the already-validated pipeline36 directory contain, or every
    report tool needs changing."""
    _run_retry(tmp_path, monkeypatch, verdicts=[1], retries=2)
    names = {p.name for p in tmp_path.glob("airline.friendliness.t0*")}
    assert not any(".a" in n for n in names), f"no attempt infix on a first-draw pass: {names}"


def test_the_recorded_attempt_survives_to_the_index(tmp_path, monkeypatch):
    """Without this the human-alignment question -- does agreement measured on first-draw
    passers transfer to re-drawn ones -- is unanswerable after the fact."""
    _, _, store = _run_retry(tmp_path, monkeypatch, verdicts=[2, 2, 1], retries=2)
    row = json.loads((tmp_path / "pair_index.json").read_text())["airline.friendliness.t0"]
    assert row["attempt"] == 3 and row["state"] == "kept"


def test_every_draw_is_recorded_as_a_reject_not_just_the_last(tmp_path, monkeypatch):
    """Each failed draw is a filter decision on a real pair and is part of the 1-filter vs
    2-filter ablation's denominator."""
    _run_retry(tmp_path, monkeypatch, verdicts=[2, 2, 1], retries=2)
    rej = json.loads((tmp_path / "rejects.json").read_text())["rejected"]
    assert len(rej) == 2, "two rejected draws before the accepted third"
    assert sorted(r["attempt"] for r in rej) == [1, 2]


def test_the_nested_property_holds(tmp_path, monkeypatch):
    """One run at retries=2 must contain the retries=0 and retries=1 answers.

    Attempt 1 is the same draw at every setting, so truncating on the recorded attempt gives
    the lower settings for free -- which is what makes the ablation one run instead of three,
    and removes run-to-run nondeterminism (kappa 0.21) from the comparison.
    """
    out2, _, store = _run_retry(tmp_path, monkeypatch, verdicts=[2, 1], retries=2)
    rej = json.loads((tmp_path / "rejects.json").read_text())["rejected"]
    # retries=0 view: only draw 1 counts, and it was rejected.
    assert out2.state == "kept" and out2.attempt == 2
    assert [r["attempt"] for r in rej] == [1], "draw 1 is on record as a reject"
    # So the retries=0 result is derivable: this cell yields no pair at setting 0.
    kept_at_0 = out2.state == "kept" and out2.attempt <= 1
    assert kept_at_0 is False


def test_a_failure_mid_retry_is_still_a_failure_not_a_silent_reject(tmp_path, monkeypatch):
    """A lost deployment during draw 2 must not be recorded as the filter rejecting."""
    _patch_schema(monkeypatch)
    stages, calls = _retry_stages([2, 1])
    boom = {"n": 0}

    def _traj(cell, level, body):
        boom["n"] += 1
        if boom["n"] > 2:                     # draw 2's first simulation
            raise RuntimeError("DEPLOYMENT_SCALING_UP")
        return FakeTraj(level), 1

    stages.trajectory = _traj
    out = run_task(_cell(), stages, Store(tmp_path), JUDGES, log=lambda m: None,
                   filter_retries=1)
    assert out.state == "failed" and out.phase == "P2"


# --- turning retries on over a run that already finished ---

def test_retries_can_be_added_to_a_completed_run(tmp_path, monkeypatch):
    """**A finished retries=0 run can be re-opened at retries=N.**

    This is the property that makes the ablation affordable on data already collected: the
    kept cells resume for free and only the rejects are re-drawn. It is not a new mechanism --
    per-phase resume plus the attempt loop gives it -- but it is load-bearing enough to pin,
    because it is the difference between the experiment costing 11 re-draws and costing 36.
    """
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    keeper, loser = _cell(criterion="friendliness"), _cell(criterion="task_resolution")

    # Pass 1, retries=0: one kept, one rejected.
    good, _ = _stages(sides=(1, 1), intent=1)
    run_task(keeper, good, store, JUDGES, log=lambda m: None, filter_retries=0)
    bad, _ = _stages(sides=(2, 1), intent=1)
    first = run_task(loser, bad, store, JUDGES, log=lambda m: None, filter_retries=0)
    assert first.state == "rejected" and first.attempt == 1

    # Pass 2, retries=1 over the same directory.
    kept_calls: list = []
    kept_stages, _ = _stages(sides=(1, 1), intent=1, calls=kept_calls)
    again = run_task(keeper, kept_stages, store, JUDGES, log=lambda m: None,
                     filter_retries=1)
    assert again.state == "kept" and again.attempt == 1
    assert kept_calls == [], "a kept cell must cost nothing when retries are switched on"

    # The reject re-draws, and its first draw is left exactly as it was.
    retry_stages, retry_calls = _retry_stages([1], intent=1)   # draw 2 passes
    second = run_task(loser, retry_stages, store, JUDGES, log=lambda m: None,
                      filter_retries=1)
    assert second.state == "kept" and second.attempt == 2
    assert sum(1 for c in retry_calls if c[0] == "P2") == 2, "only the new draw simulates"
    assert sum(1 for c in retry_calls if c[0] == "P1") == 0, "instructions came off disk"


def test_the_original_retries_zero_result_is_still_derivable_afterwards(tmp_path, monkeypatch):
    """Re-opening a run must not overwrite what it originally concluded.

    Attempt 1's pair and votes are read from disk and never regenerated, so truncating on the
    attempt number still reproduces the retries=0 answer. If a re-draw could touch attempt 1,
    adding retries would silently invalidate the very baseline the ablation compares against.
    """
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    bad, _ = _stages(sides=(2, 1), intent=1)
    run_task(cell, bad, store, JUDGES, log=lambda m: None)
    first_pair = json.loads(store.pair_path(cell).read_text())
    first_votes = json.loads(store.votes_path.read_text())["votes"]

    retry_stages, _ = _retry_stages([1], intent=1)
    out = run_task(cell, retry_stages, store, JUDGES, log=lambda m: None, filter_retries=2)
    assert out.state == "kept" and out.attempt == 2

    assert json.loads(store.pair_path(cell).read_text()) == first_pair, "attempt 1 untouched"
    now = json.loads(store.votes_path.read_text())["votes"]
    for v in first_votes:
        assert v in now, "the original votes must survive, they are the retries=0 result"


# --- the plan must match the run, including for terminal cells ---

def test_a_cell_rejected_at_p4_does_not_plan_a_p5_call(tmp_path, monkeypatch, capsys):
    """**The cascade stops at the first veto, so judge2 never runs on a P4 reject.**

    Listing it as pending overstated the work: pointed at the finished 36-pair directory the
    plan said "P5 x9" for nine calls that cannot happen. A plan a spend decision is made from
    has to be right about that.
    """
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    bad, _ = _stages(sides=(2, 1), intent=1)
    run_task(cell, bad, store, JUDGES, log=lambda m: None)

    from tools.run_pipeline import _dry_run
    _dry_run([cell], store, JUDGES, 0)
    out = capsys.readouterr().out
    assert "nothing, complete (rejected)" in out
    assert "would run: nothing" in out, out.split("would run:")[1][:60]


def test_a_finished_run_plans_nothing_at_zero_retries(tmp_path, monkeypatch, capsys):
    _patch_schema(monkeypatch)
    store = Store(tmp_path)
    keeper, loser = _cell(criterion="friendliness"), _cell(criterion="task_resolution")
    good, _ = _stages(sides=(1, 1), intent=1)
    run_task(keeper, good, store, JUDGES, log=lambda m: None)
    bad, _ = _stages(sides=(2, 1), intent=1)
    run_task(loser, bad, store, JUDGES, log=lambda m: None)

    from tools.run_pipeline import _dry_run
    _dry_run([keeper, loser], store, JUDGES, 0)
    out = capsys.readouterr().out
    assert "would run: nothing" in out


def test_the_plan_shows_redraws_for_rejects_when_retries_are_on(tmp_path, monkeypatch, capsys):
    """The other half of the same bug: at retries>0 the real work is re-drawing the rejects,
    and the plan showed it nowhere."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    bad, _ = _stages(sides=(2, 1), intent=1)
    run_task(cell, bad, store, JUDGES, log=lambda m: None)

    from tools.run_pipeline import _dry_run
    _dry_run([cell], store, JUDGES, 2)
    out = capsys.readouterr().out
    assert "up to 2 re-draw(s)" in out
    assert "P2 x4" in out, "2 re-draws x 2 levels"
    assert "UPPER bound" in out, "a draw that passes ends the cell early"


def test_a_kept_cell_never_plans_a_redraw(tmp_path, monkeypatch, capsys):
    """Retries apply to rejects only. Re-drawing a kept pair would spend the trajectories
    again and, worse, could replace a pair a human has already annotated."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    good, _ = _stages(sides=(1, 1), intent=1)
    run_task(cell, good, store, JUDGES, log=lambda m: None)

    from tools.run_pipeline import _dry_run
    _dry_run([cell], store, JUDGES, 2)
    out = capsys.readouterr().out
    assert "nothing, complete (kept)" in out
    assert "would run: nothing" in out


def test_a_partly_spent_retry_budget_plans_only_what_is_left(tmp_path, monkeypatch, capsys):
    """A cell already re-drawn twice and still rejected has 0 draws left at retries=2, so a
    resume must not start the budget over."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    stages, _ = _retry_stages([2, 2, 2], intent=1)
    out1 = run_task(cell, stages, store, JUDGES, log=lambda m: None, filter_retries=2)
    assert out1.state == "rejected" and out1.attempt == 3

    from tools.run_pipeline import _dry_run
    _dry_run([cell], store, JUDGES, 2)
    out = capsys.readouterr().out
    assert "would run: nothing" in out, "the budget is spent, not reset"


# --- the per-cell draw history ---

def test_the_index_records_every_draw_not_just_the_winner(tmp_path, monkeypatch):
    """Every draw's artifacts already survive on disk -- trajectories, pair, votes, and its
    `rejects.json` row. Nothing was ever discarded. But reading "what happened to this cell"
    meant joining three files, and the question that needs it most -- does human alignment
    measured on first-draw passers transfer to re-drawn pairs -- should not need a join."""
    _, _, store = _run_retry(tmp_path, monkeypatch, verdicts=[2, 2, 1], retries=2)
    row = json.loads((tmp_path / "pair_index.json").read_text())["airline.friendliness.t0"]
    draws = row["draws"]
    assert [d["attempt"] for d in draws] == [1, 2, 3]
    assert [d["verdict"] for d in draws] == ["rejected", "rejected", "kept"]
    assert draws[0]["rejected_by"] == "judge1"
    assert draws[-1]["rejected_by"] == ""
    # Each draw names its own pair, so the losers are addressable, not merely counted.
    assert len({d["pair_id"] for d in draws}) == 3


def test_a_single_draw_still_gets_a_history_of_one(tmp_path, monkeypatch):
    """At the default retries=0 the field must exist rather than being absent, or every
    consumer needs a branch."""
    _, _, store = _run_retry(tmp_path, monkeypatch, verdicts=[1], retries=0)
    row = json.loads((tmp_path / "pair_index.json").read_text())["airline.friendliness.t0"]
    assert row["draws"] == [{"attempt": 1, "pair_id": "pair-airline.friendliness.t0-d1",
                             "verdict": "kept", "rejected_by": "", "phase": "P5"}]


def test_the_history_is_idempotent_across_a_resume(tmp_path, monkeypatch):
    """A resume re-reads a draw from disk and re-records its verdict. Appending blindly would
    grow the history on every re-run and make `len(draws)` a count of invocations."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()
    bad, _ = _stages(sides=(2, 1), intent=1)
    for _ in range(3):
        run_task(cell, bad, store, JUDGES, log=lambda m: None)
    row = json.loads((tmp_path / "pair_index.json").read_text())[cell.key]
    assert len(row["draws"]) == 1, "three runs of one draw is still one draw"


def test_the_winning_pair_is_distinguishable_from_the_losers(tmp_path, monkeypatch):
    """The whole point: an evaluator sweep must score the kept pair, and a human annotating
    re-drawn pairs needs to know which is which."""
    out, _, store = _run_retry(tmp_path, monkeypatch, verdicts=[2, 1], retries=1)
    row = json.loads((tmp_path / "pair_index.json").read_text())["airline.friendliness.t0"]
    winners = [d["pair_id"] for d in row["draws"] if d["verdict"] == "kept"]
    losers = [d["pair_id"] for d in row["draws"] if d["verdict"] == "rejected"]
    assert winners == [out.pair_id] == [row["pair_id"]]
    assert out.pair_id not in losers


# --- configuration comes from one YAML file ---

def _write_cfg(tmp_path, body: str):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def test_an_empty_config_gets_every_default(tmp_path):
    from tools.run_pipeline import load_config
    c = load_config(_write_cfg(tmp_path, "{}"))
    assert c.seed == 2026
    assert c.sim_attempts == 10 and c.sim_backoff == 10.0
    assert c.task_indices is None, "null means every task, not task 0"
    assert c.filter_retries == 0
    assert len(c.criteria) == 3


def test_an_unknown_key_is_an_error_not_a_warning(tmp_path):
    """A typo that silently fell back to the default would be invisible in the output and
    would misattribute whatever the run then measured."""
    from tools.run_pipeline import load_config
    with pytest.raises(SystemExit) as e:
        load_config(_write_cfg(tmp_path, "sim_concurency: 8\n"))
    assert "unknown key" in str(e.value)


def test_criteria_are_a_list_of_name_and_description(tmp_path):
    from tools.run_pipeline import load_config
    c = load_config(_write_cfg(tmp_path, """
criteria:
  - name: brevity
    description: Whether the agent says it in as few words as carry the meaning.
"""))
    assert sorted(c.criteria) == ["brevity"]
    assert c.criteria["brevity"].description.startswith("Whether the agent")


def test_a_config_criterion_needs_a_description_unless_it_is_a_builtin(tmp_path):
    """`generate_instructions` reads only name and description, so a new criterion with no
    description would steer on an empty string."""
    from tools.run_pipeline import load_config
    with pytest.raises(SystemExit) as e:
        load_config(_write_cfg(tmp_path, "criteria:\n  - name: brevity\n"))
    assert "description" in str(e.value)


def test_naming_a_builtin_criterion_reuses_its_exact_text(tmp_path):
    """So the three shipped criteria keep the wording every prior run used."""
    from tools.run_pipeline import load_config
    from src.metaeval.steering import CRITERIA
    c = load_config(_write_cfg(tmp_path, "criteria:\n  - name: friendliness\n"))
    assert c.criteria["friendliness"].description == CRITERIA["friendliness"].description


def test_an_unknown_domain_is_rejected_with_the_available_list(tmp_path):
    from tools.run_pipeline import load_config
    with pytest.raises(SystemExit) as e:
        load_config(_write_cfg(tmp_path, "domains: [airline, atlantis]\n"))
    assert "atlantis" in str(e.value) and "airline" in str(e.value)


def test_nonsense_concurrency_is_rejected(tmp_path):
    from tools.run_pipeline import load_config
    with pytest.raises(SystemExit):
        load_config(_write_cfg(tmp_path, "workers: 0\n"))
    with pytest.raises(SystemExit):
        load_config(_write_cfg(tmp_path, "filter_retries: -1\n"))


def test_task_indices_must_be_integers(tmp_path):
    from tools.run_pipeline import load_config
    with pytest.raises(SystemExit) as e:
        load_config(_write_cfg(tmp_path, "task_indices: [0, 'two']\n"))
    assert "task_indices" in str(e.value)


def test_the_config_travels_into_the_run_record(tmp_path):
    """A pipeline clock without its concurrency is not reproducible, so the parameters are
    recorded beside the numbers they produced."""
    from tools.run_pipeline import load_config
    c = load_config(_write_cfg(tmp_path, "seed: 7\nworkers: 3\nfilter_retries: 2\n"))
    rec = c.as_record
    assert rec["seed"] == 7 and rec["workers"] == 3 and rec["filter_retries"] == 2
    assert sorted(rec["criteria"]) == rec["criteria"], "names only, not the full text"


def test_the_shipped_config_parses(tmp_path):
    """The file the tool defaults to must load, or every invocation fails."""
    from tools.run_pipeline import load_config
    c = load_config(Path("config/pipeline.yaml"))
    assert c.seed == 2026 and c.task_indices is None
    assert len(c.criteria) == 3 and len(c.domains) == 4


# --- the manual has to describe the interface that exists ---
#
# The flags moved into the YAML and the manual kept its flag table, so the documented command
# `run_pipeline.py --task-indices 0 1 2 --out-dir ...` had become an argparse error -- the first
# thing anyone following the manual would hit. Both halves of that drift are pinned here.

_DOC = Path("docs/REPRODUCE.md")


def test_the_manual_only_shows_flags_the_tool_accepts():
    import re
    real = {"--config", "--dry-run"}
    for line in _DOC.read_text().splitlines():
        if "run_pipeline.py" not in line:
            continue
        flags = set(re.findall(r"--[a-z][a-z-]+", line))
        assert flags <= real, f"{flags - real} does not exist any more: {line.strip()}"


def _documented_config_keys() -> set[str]:
    """Backticked keys in the config table of the generation section, and nowhere else.

    Scoped to the section: the regex matches any table row opening with a backticked lowercase
    word, and the manual has other such tables -- the sweep's per-lane caps list `gateway` and
    `gemini`, which are lanes rather than config keys.
    """
    import re
    text = _DOC.read_text()
    start = text.index("## 2. Generation")
    end = text.index("## 3.", start)
    return set(re.findall(r"^\| `([a-z_]+)` \|", text[start:end], re.M))


def test_every_key_the_manual_documents_is_a_real_config_key():
    """The other direction: a table row for a key `load_config` would reject as unknown."""
    from tools.run_pipeline import _DEFAULTS
    documented = _documented_config_keys()
    assert documented, "the key table vanished from the generation section"
    assert documented <= set(_DEFAULTS), f"not config keys: {documented - set(_DEFAULTS)}"
    assert set(_DEFAULTS) <= documented, f"undocumented: {set(_DEFAULTS) - documented}"


# --- trajectories live in a subdirectory ---

def test_trajectories_are_written_to_the_subdirectory(tmp_path, monkeypatch):
    """A flat run directory at the full tau3 space holds 2,250 trajectory files beside 1,125
    pair files and six bookkeeping files."""
    _run(tmp_path, monkeypatch)
    assert not list(tmp_path.glob("*.ok.json")), "nothing loose in the run root"
    assert (tmp_path / "trajectories" / "airline.friendliness.t0.ok.json").is_file()


def test_the_writer_and_the_readers_share_one_layout_constant():
    """**A writer and reader that disagree here fail silently** -- as "0 trajectory files"
    rather than as an error, which is how a whole stage's cost can read as unmeasured."""
    from src.metaeval.runs import TRAJECTORY_SUBDIR
    store = Store(Path("/tmp/_layout_check"))
    assert store.traj_dir.name == TRAJECTORY_SUBDIR


def test_the_pair_writer_and_readers_share_one_layout_constant():
    """Same hazard, same guard. A pair-layout disagreement is worse than the trajectory one:
    `dataset.json` would come back empty and every downstream count would read 0 pairs without
    an error anywhere."""
    from src.metaeval.runs import PAIRS_SUBDIR
    store = Store(Path("/tmp/_layout_check"))
    assert store.pairs_dir.name == PAIRS_SUBDIR


def test_pair_files_reads_both_layouts(tmp_path):
    """The flat reference runs -- `data/step3/runs_final` and `data/step5/pipeline36`, which the
    paper quotes -- are not migrated, so the reader has to cover both."""
    from src.metaeval.runs import pair_files
    (tmp_path / "flat.pairs.json").write_text("[]")
    assert [f.name for f in pair_files(tmp_path)] == ["flat.pairs.json"]
    sub = tmp_path / "pairs"
    sub.mkdir()
    (sub / "nested.pairs.json").write_text("[]")
    assert sorted(f.name for f in pair_files(tmp_path)) == ["flat.pairs.json", "nested.pairs.json"]


def test_a_cell_in_both_layouts_is_returned_once(tmp_path):
    """**The version of this the first test could not fail.** It used two distinct filenames, so
    "no double-counting" was unfalsifiable. The state that matters is the SAME cell in both
    places -- reachable when a resume writes `pairs/` beside a legacy orchestrator's root file,
    or when a migration is done with `cp`. Concatenating the layouts inflated `_count_pairs`, the
    denominator the per-pair cost claim is stated over, and put two entries per cell in
    `dataset.json` looking exactly like `filter_retries > 0`."""
    from src.metaeval.runs import pair_files
    name = "airline.friendliness.t0.pairs.json"
    (tmp_path / name).write_text(json.dumps([{"id": "p0"}]))
    sub = tmp_path / "pairs"
    sub.mkdir()
    (sub / name).write_text(json.dumps([{"id": "p0"}]))

    got = pair_files(tmp_path)
    assert len(got) == 1, f"one cell, one file: {[str(f) for f in got]}"
    assert got[0].parent.name == "pairs", "the subdirectory is the winner"
    assert Store(tmp_path).n_pairs() == 1, "the dataset.json denominator must not double"


def test_the_report_finds_trajectories_in_the_subdirectory(tmp_path, monkeypatch):
    """The end-to-end version of the constant check: write with the pool, read with the
    report, and require a non-zero count."""
    _run(tmp_path, monkeypatch)
    from src.metaeval.runs import trajectory_files
    found = trajectory_files(tmp_path)
    assert len(found) == 2, [f.name for f in found]
    assert all(f.parent.name == "trajectories" for f in found)


def test_a_flat_run_directory_is_still_read(tmp_path):
    """`data/step3/runs_final` and `data/step5/pipeline36` are flat and are not migrated;
    both are validated references the paper quotes."""
    from src.metaeval.runs import trajectory_files
    (tmp_path / "airline.friendliness.t0.ok.json").write_text('{"turns": []}')
    (tmp_path / "judge_votes.json").write_text("{}")
    (tmp_path / "airline.friendliness.t0.pairs.json").write_text("[]")
    found = trajectory_files(tmp_path)
    assert [f.name for f in found] == ["airline.friendliness.t0.ok.json"]


def test_bookkeeping_files_in_the_subdirectory_are_still_excluded(tmp_path):
    """The exclusion list applies in either layout, or a stray summary.json inside
    trajectories/ would be scanned as a trajectory."""
    from src.metaeval.runs import trajectory_files
    sub = tmp_path / "trajectories"
    sub.mkdir()
    (sub / "airline.friendliness.t0.ok.json").write_text('{"turns": []}')
    (sub / "summary.json").write_text("{}")
    assert [f.name for f in trajectory_files(tmp_path)] == ["airline.friendliness.t0.ok.json"]


def test_the_shipped_config_uses_the_exact_builtin_criterion_text(tmp_path):
    """**A paraphrase here silently changes the steering prompt for every future run.**

    Writing the descriptions into the config by hand did exactly that: two of three diverged
    and task_resolution was cut from 416 characters to 212, which would have broken
    comparability with every prior run while the config still looked correct. The shipped file
    names the criteria and nothing else, so the built-in text is reused rather than restated.
    """
    from src.metaeval.steering import CRITERIA
    from tools.run_pipeline import load_config
    c = load_config(Path("config/pipeline.yaml"))
    for name, crit in c.criteria.items():
        assert name in CRITERIA, f"{name} is not a built-in; is that deliberate?"
        assert crit.description == CRITERIA[name].description, (
            f"{name}: the config's description has drifted from the built-in")


# --- a killed invocation's tokens are recoverable ---
#
# `cumulative` sums the records in `runs`, and a record only lands there when a run reaches its
# final write. The full tau3 run's first pass built ~470 cells and was killed, so afterwards
# `generated_instructions.json` described 657 instruction calls beside 1,125 records -- every
# method-stage figure ~40% low, with scaling by record count as the only recourse.

class _FakeMeter:
    def __init__(self, tokens): self.tokens = tokens
    def summary(self): return {"total_tokens": self.tokens, "n_calls": 1}
    def stage_service(self): return {"judge": 1.0}
    def untimed_calls(self): return {}
    def by(self, key): return {}


def test_an_orphaned_checkpoint_is_recovered(tmp_path):
    from tools.run_pipeline import recover_checkpoints, run_record
    store = Store(tmp_path)
    store.write_checkpoint("killed1", run_record(_FakeMeter(500), {}, 470, "killed1"))
    got = recover_checkpoints(store, runs=[], this_run_id="current")
    assert [r["run_id"] for r in got] == ["killed1"]
    assert got[0]["usage"]["total_tokens"] == 500
    assert got[0]["recovered_from_checkpoint"] is True


def test_a_checkpoint_whose_run_completed_is_not_double_counted(tmp_path):
    """**The way this fix could do harm.** A completed run's record is in `runs`; if its
    checkpoint were also folded in, every token it spent would be counted twice -- worse than
    the undercount it replaces, and in the flattering direction."""
    from tools.run_pipeline import recover_checkpoints, run_record
    store = Store(tmp_path)
    store.write_checkpoint("done1", run_record(_FakeMeter(500), {}, 100, "done1"))
    runs = [{"run_id": "done1", "usage": {"total_tokens": 800}}]
    assert recover_checkpoints(store, runs, this_run_id="current") == []


def test_the_current_invocations_own_checkpoint_is_skipped(tmp_path):
    """Its real record is appended right after, so folding the checkpoint in would double it."""
    from tools.run_pipeline import recover_checkpoints, run_record
    store = Store(tmp_path)
    store.write_checkpoint("me", run_record(_FakeMeter(500), {}, 100, "me"))
    assert recover_checkpoints(store, runs=[], this_run_id="me") == []


def test_clearing_a_checkpoint_leaves_the_others(tmp_path):
    from tools.run_pipeline import run_record
    store = Store(tmp_path)
    store.write_checkpoint("a", run_record(_FakeMeter(1), {}, 1, "a"))
    store.write_checkpoint("b", run_record(_FakeMeter(2), {}, 1, "b"))
    store.clear_checkpoint("a")
    assert sorted(store.read_checkpoints()) == ["b"]


def test_the_checkpoint_file_is_not_mistaken_for_a_trajectory_or_a_pair():
    """A new file in the run root is scanned by the report unless it is excluded -- the same
    omission that once let `generated_instructions.json` be parsed as a trajectory."""
    from src.metaeval.runs import NON_TRAJECTORY
    assert "usage_checkpoints.json" in NON_TRAJECTORY


def test_a_run_past_the_checkpoint_interval_completes(tmp_path, monkeypatch):
    """**Every existing end-to-end test uses a handful of cells, so none crossed the checkpoint
    interval.** `run_cfg` was read by the in-loop checkpoint but assigned after the loop, so the
    100th completed task raised `UnboundLocalError` -- and the loop catches only
    `KeyboardInterrupt`, so it escaped `main` and killed the run before `pipeline_run.json`,
    `write_dataset` or `publish_stage_usage` ran. A checkpoint added to bound a kill to 100 cells
    guaranteed one at cell 100 instead.

    Drives `main` with 120 fake cells so the interval is crossed. `warm: false` in the config
    rather than a monkeypatch on `run_pipeline.warm`: `main` imports `warm` *locally*, so a
    module-attribute patch is shadowed and the test would make real network calls -- which is how
    an earlier version of this test passed only because the deployments happened to be warm.
    """
    import yaml
    import tools.run_pipeline as rp

    _patch_schema(monkeypatch)
    cells = [_cell(idx=i) for i in range(120)]
    stages, _ = _stages()

    cfg = yaml.safe_load(Path("config/pipeline.yaml").read_text())
    cfg["warm"] = False
    cfg["out_dir"] = str(tmp_path)
    cfg["limit"] = None
    cfg_path = tmp_path / "offline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(rp, "build_cells", lambda *a, **k: cells)
    monkeypatch.setattr(rp, "real_stages", lambda *a, **k: stages)
    monkeypatch.setattr(rp, "print_report", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(cfg_path)])

    assert rp.main() == 0, "a 120-cell run must not raise"
    assert (tmp_path / "pipeline_run.json").is_file(), "the run record never got written"
    assert (tmp_path / "dataset.json").is_file()
    blob = json.loads((tmp_path / "pipeline_run.json").read_text())
    assert blob["n_tasks"] == 120
    # The checkpoint fired at 100 and was cleared once its real record was in `runs`.
    assert json.loads((tmp_path / "usage_checkpoints.json").read_text()) == {}


def test_resuming_a_flat_run_does_not_rebuild_its_pairs(tmp_path, monkeypatch):
    """**The module docstring's "No rewrite of an existing pair file" applied only to the new
    layout.** `pair_for` read `pairs/` alone, so resuming a pre-subdirectory run -- including
    `data/step5/pipeline36`, a validated reference -- found nothing, rebuilt the pair with a
    fresh uuid, orphaned the votes already cast on the old id, and re-spent both judges. With
    the trajectories flat too, P2 re-ran as well: ~97% of a task's cost to recreate what was
    already on disk."""
    _patch_schema(monkeypatch)
    store, cell = Store(tmp_path), _cell()

    # A flat run: pair and trajectories in the root, as every pre-subdirectory run has them.
    (tmp_path / f"{cell.key}.pairs.json").write_text(json.dumps(
        [{"id": "original-uuid", "correct_response": 1, "criterion_name": cell.criterion}]))
    for lvl in ("ok", "good"):
        (tmp_path / f"{cell.key}.{lvl}.json").write_text(json.dumps({"level": lvl}))

    stages, calls = _stages()
    out = run_task(cell, stages, store, JUDGES, log=lambda m: None)

    assert out.pair_id == "original-uuid", "a fresh uuid means the old votes are orphaned"
    assert not any(c[0] == "P3" for c in calls), "the pair must be read back, not rebuilt"
    assert not any(c[0] == "P2" for c in calls), "and its trajectories not re-simulated"


def test_no_removed_flag_name_survives_in_run_pipeline():
    """The flags moved into the YAML, but their names lived on in comments and docstrings --
    `--filter-retries`, `--limit 2`, `--out-dir`, `--workers 6` -- describing current parameters
    under names the tool no longer accepts. This repo treats a comment that asserts absent
    behaviour as a defect, and the same drift in the manual is already pinned by
    `test_the_manual_only_shows_flags_the_tool_accepts`.

    Scoped to the flags THIS tool dropped. `--dataset` and `--answers` also appear, but they
    belong to `run_evaluators` and are legitimate cross-references.
    """
    from tools.run_pipeline import _DEFAULTS
    src = Path("tools/run_pipeline.py").read_text()
    removed = {"--" + k.replace("_", "-") for k in _DEFAULTS} | {"--no-warm"}
    found = sorted(f for f in removed if f in src)
    assert not found, f"removed flag name(s) still referenced: {found}"
