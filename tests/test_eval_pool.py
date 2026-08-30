"""The shared worker pool behind the evaluator sweep: order, admission, and resume.

These exist because the sweep grew from 36 pairs to 1,000, which turns three things that
were survivable into things that are not:

- a **serial** roster, where `gemini-2.5-flash` ran 957s of a 61-minute sweep with every
  Fireworks model idle, and where a dedicated deployment warmed at its turn had already
  scaled to zero (`qwen3-4b` and `qwen3-1p7b` each scored 0 of 36 that way);
- **no checkpoint**, so a crash at hour 3 of a 4-hour model discarded all of it;
- **no circuit breaker**, so one unreachable model works through 2,000 calls, each retried
  four times inside litellm, and then reappears in all three retry passes.

Every test here pins a property of the fix rather than an implementation detail, so the
scheduler can be rewritten without rewriting the file.
"""

import collections
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.metaeval.scoring import (
    Checkpoint,
    Task,
    evaluate,
    evaluate_models,
    provider_of,
)


# --- helpers ---------------------------------------------------------------------------

def _entries(n=3):
    return [{
        "id": f"pair-{i}",
        "criterion_name": "clarity",
        "criterion_description": "Is it clear?",
        "prompt": "policy text",
        "response_1": f"good {i}",
        "response_2": f"bad {i}",
        "correct_response": 1,
    } for i in range(n)]


MODELS = [("gpt-oss-20b", "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"),
          ("gemini-2.5-flash", "gemini/gemini-2.5-flash"),
          ("kimi-k3", "fireworks_ai/accounts/fireworks/models/kimi-k3")]


def _good(_label, _entry, key):
    """response_1 always wins, so every pair is scored correctly."""
    return 0.9 if key == "response_1" else 0.2


def _recorder():
    """A score_fn that logs every (label, id, key, run-ish) call it sees, thread-safely."""
    seen, lock = [], threading.Lock()

    def fn(label, entry, key):
        with lock:
            seen.append((label, entry["id"], key))
        return _good(label, entry, key)
    return fn, seen


# --- provider_of -----------------------------------------------------------------------

@pytest.mark.parametrize("model,provider", [
    ("fireworks_ai/accounts/fireworks/models/gpt-oss-20b", "fireworks_ai"),
    ("gemini/gemini-2.5-flash", "gemini"),
    # `provider_of` still answers "fireworks_ai" here, and that is the *wrong* lane for a
    # dedicated deployment -- its ceiling is its own single replica, not the account's token
    # budget. Pinned as the documented fallback, with `test_lane_of_overrides_the_prefix`
    # covering the override the roster actually uses.
    ("fireworks_ai/accounts/fireworks/models/qwen3-1p7b#accounts/<account>/deployments/x",
     "fireworks_ai"),
    ("", ""),                       # the single-model `evaluate()` path
    ("bare-model-name", "bare-model-name"),
])
def test_provider_of(model, provider):
    assert provider_of(model) == provider


# --- coverage: every task runs exactly once --------------------------------------------

def test_every_model_pair_side_and_run_is_called_exactly_once():
    fn, seen = _recorder()
    data = _entries(3)
    out = evaluate_models(data, MODELS, fn, num_runs=2, max_workers=4, progress=False)
    assert len(seen) == 3 * 2 * 2 * len(MODELS), "models x pairs x sides x runs"
    # Exactly once per (model, pair, side) *per run*, so each triple appears num_runs times.
    from collections import Counter
    assert set(Counter(seen).values()) == {2}
    assert set(out) == {label for label, _ in MODELS}
    assert all(len(out[label]) == 3 for label in out)


def test_results_are_per_model_and_independently_assembled():
    """One model's failures must not touch another's results."""
    def fn(label, entry, key):
        if label == "kimi-k3":
            raise RuntimeError("model is down")
        return _good(label, entry, key)

    out = evaluate_models(_entries(2), MODELS, fn, num_runs=1, retry_passes=0,
                          progress=False)
    assert len(out["gpt-oss-20b"]) == 2
    assert len(out["gemini-2.5-flash"]) == 2
    assert out["kimi-k3"] == [], "a dead model reports nothing, not zeros"
    assert all(r["runs"][0]["correct"] for r in out["gpt-oss-20b"])


# --- shuffling -------------------------------------------------------------------------

def test_same_seed_is_the_same_order():
    a, seen_a = _recorder()
    b, seen_b = _recorder()
    data = _entries(4)
    evaluate_models(data, MODELS, a, num_runs=1, max_workers=1, seed=7, progress=False)
    evaluate_models(data, MODELS, b, num_runs=1, max_workers=1, seed=7, progress=False)
    assert seen_a == seen_b, "a seeded run has to be re-creatable"


def test_a_different_seed_is_a_different_order():
    a, seen_a = _recorder()
    b, seen_b = _recorder()
    data = _entries(4)
    evaluate_models(data, MODELS, a, num_runs=1, max_workers=1, seed=1, progress=False)
    evaluate_models(data, MODELS, b, num_runs=1, max_workers=1, seed=2, progress=False)
    assert sorted(seen_a) == sorted(seen_b), "same work, different order"
    assert seen_a != seen_b


def test_order_interleaves_models_rather_than_finishing_one_at_a_time():
    """The property that keeps a scale-to-zero deployment awake.

    A serial roster let `qwen3-4b` and `qwen3-1p7b` sleep through the models ahead of them
    and score 0 of 36 entries. What prevents that is not the pool alone -- it is that any
    window of the task order contains every model, so no deployment goes 5 minutes without
    traffic. Asserted as "the longest same-model streak is short", which is the observable
    form of that claim, at 1 worker so the recorded order *is* the schedule.
    """
    fn, seen = _recorder()
    evaluate_models(_entries(20), MODELS, fn, num_runs=1, max_workers=1, seed=0,
                    progress=False)
    longest = best = 1
    for prev, cur in zip(seen, seen[1:]):
        best = best + 1 if cur[0] == prev[0] else 1
        longest = max(longest, best)
    assert longest <= 8, f"longest single-model streak was {longest} of {len(seen)} calls"
    # And the first slice of the run already touches every model, which is the warm-up claim.
    assert {c[0] for c in seen[:12]} == {label for label, _ in MODELS}


# --- provider admission control --------------------------------------------------------

def test_provider_cap_bounds_calls_in_flight_to_one_provider():
    """`provider_caps` is the guard rail the pool size cannot express.

    Gemini is one model among nine, so shuffling cannot spread its load: a burst that lands
    several workers on it has nowhere else to go.
    """
    live = {"gemini": 0}
    peak = {"gemini": 0, "fireworks_ai": 0}
    live_fw = {"n": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        provider = "gemini" if label == "gemini-2.5-flash" else "fireworks_ai"
        with lock:
            if provider == "gemini":
                live["gemini"] += 1
                peak["gemini"] = max(peak["gemini"], live["gemini"])
            else:
                live_fw["n"] += 1
                peak["fireworks_ai"] = max(peak["fireworks_ai"], live_fw["n"])
        import time
        time.sleep(0.02)
        with lock:
            if provider == "gemini":
                live["gemini"] -= 1
            else:
                live_fw["n"] -= 1
        return 0.5

    evaluate_models(_entries(8), MODELS, fn, num_runs=1, max_workers=8,
                    provider_caps={"gemini": 2}, progress=False)
    assert peak["gemini"] <= 2, f"gemini peaked at {peak['gemini']}, cap was 2"
    # Uncapped Fireworks must not have been throttled to the same number, or the cap is
    # leaking across providers and the whole pool is running at 2.
    assert peak["fireworks_ai"] > 2, f"fireworks peaked at only {peak['fireworks_ai']}"


def test_no_caps_means_only_the_pool_bounds_concurrency():
    peak = {"n": 0}
    live = {"n": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        import time
        with lock:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
        time.sleep(0.02)
        with lock:
            live["n"] -= 1
        return 0.5

    evaluate_models(_entries(10), MODELS, fn, num_runs=1, max_workers=6, progress=False)
    assert 1 < peak["n"] <= 6


# --- the circuit breaker ---------------------------------------------------------------

def test_a_model_that_never_answers_is_abandoned():
    """2,000 calls x 4 litellm retries x 3 retry passes is hours spent on nothing."""
    calls = {"dead": 0, "live": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        with lock:
            calls["dead" if label == "kimi-k3" else "live"] += 1
        if label == "kimi-k3":
            raise RuntimeError("Model not found, inaccessible, and/or not deployed")
        return _good(label, entry, key)

    out = evaluate_models(_entries(40), MODELS, fn, num_runs=1, retry_passes=1,
                          retry_delay_s=0, give_up_after=5, max_workers=1, progress=False)
    assert out["kimi-k3"] == []
    # 5 to trip, and the pool may have a few already in flight when it does. Far short of
    # the 80 the model would otherwise be asked for, plus retries.
    assert calls["dead"] < 20, calls["dead"]
    assert calls["live"] == 2 * 40 * 2, "the healthy models are untouched"
    assert len(out["gpt-oss-20b"]) == 40


def test_the_breaker_spares_a_model_that_is_merely_flaky():
    """Rate limits are the dominant failure and they are transient -- that is what the retry
    passes are for. A model with *any* success must never be abandoned."""
    n = {"i": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        if label != "kimi-k3":
            return _good(label, entry, key)
        with lock:
            n["i"] += 1
            i = n["i"]
        if i % 2 == 0:                       # every other call rate-limits
            raise RuntimeError("rate limit exceeded")
        return _good(label, entry, key)

    out = evaluate_models(_entries(6), MODELS, fn, num_runs=1, retry_passes=4,
                          retry_delay_s=0, give_up_after=3, max_workers=1, progress=False)
    assert len(out["kimi-k3"]) == 6, "a flaky model must be retried, not abandoned"


def test_give_up_after_zero_disables_the_breaker():
    calls = {"n": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        with lock:
            calls["n"] += 1
        raise RuntimeError("down")

    evaluate_models(_entries(5), [("only", "fw/x")], fn, num_runs=1, retry_passes=0,
                    give_up_after=0, max_workers=1, progress=False)
    assert calls["n"] == 10, "every call attempted: 5 pairs x 2 sides"


# --- Checkpoint, on its own ------------------------------------------------------------

def test_checkpoint_writes_one_line_per_call(tmp_path):
    ck = Checkpoint(tmp_path / "checkpoint.jsonl")
    evaluate_models(_entries(2), MODELS, _good, num_runs=1, checkpoint=ck, progress=False)
    lines = [json.loads(l) for l in ck.path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2 * 2 * len(MODELS)
    assert {l["label"] for l in lines} == {label for label, _ in MODELS}
    assert all(l["score"] is not None for l in lines)
    assert {l["key"] for l in lines} == {"response_1", "response_2"}


def test_checkpoint_is_absent_when_no_path_is_given(tmp_path):
    """The default has to stay "no file": every existing caller and test passes none."""
    ck = Checkpoint(None)
    evaluate_models(_entries(1), MODELS, _good, num_runs=1, checkpoint=ck, progress=False)
    assert list(tmp_path.iterdir()) == []
    assert ck.load() == ({}, [])


def test_checkpoint_tolerates_a_half_written_final_line(tmp_path):
    """The last line of a killed run is routinely a partial one, and refusing to resume
    because of it would defeat the whole feature."""
    p = tmp_path / "checkpoint.jsonl"
    p.write_text('{"label":"a","id":"pair-0","key":"response_1","run":0,"score":0.5}\n'
                 '{"label":"a","id":"pair-0","key":"resp')
    done, _ = Checkpoint(p).load()
    assert done == {("a", "pair-0", "response_1", 0): 0.5}


# --- resume ----------------------------------------------------------------------------

def test_resume_skips_completed_calls_and_reproduces_the_same_results(tmp_path):
    ck = Checkpoint(tmp_path / "checkpoint.jsonl")
    data = _entries(3)
    first = evaluate_models(data, MODELS, _good, num_runs=2, checkpoint=ck, progress=False)

    fn, seen = _recorder()
    second = evaluate_models(data, MODELS, fn, num_runs=2, checkpoint=ck, progress=False)
    assert seen == [], "a finished run must cost nothing to re-run"
    assert second == first


def test_resume_finishes_a_run_that_was_killed_partway(tmp_path):
    """The 11-hour case: stop mid-sweep, re-run, pay only for what is missing."""
    path = tmp_path / "checkpoint.jsonl"
    data = _entries(4)
    stop_after = 10

    calls = {"n": 0}
    lock = threading.Lock()

    def dies(label, entry, key):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n > stop_after:
            raise KeyboardInterrupt("user hit Ctrl-C")
        return _good(label, entry, key)

    with pytest.raises(KeyboardInterrupt):
        evaluate_models(data, MODELS, dies, num_runs=1, max_workers=1,
                        checkpoint=Checkpoint(path), progress=False)
    done, _ = Checkpoint(path).load()
    assert len(done) == stop_after, "everything finished before the kill is on disk"

    fn, seen = _recorder()
    out = evaluate_models(data, MODELS, fn, num_runs=1, max_workers=1,
                          checkpoint=Checkpoint(path), progress=False)
    total = 4 * 2 * len(MODELS)
    assert len(seen) == total - stop_after, "only the missing calls are paid for"
    assert all(len(out[label]) == 4 for label in out), "and the sweep is complete"


def test_resume_is_scoped_to_the_label(tmp_path):
    """A `--models X` run must not be able to satisfy Y's calls from X's checkpoint rows."""
    path = tmp_path / "checkpoint.jsonl"
    data = _entries(2)
    evaluate_models(data, [MODELS[0]], _good, num_runs=1,
                    checkpoint=Checkpoint(path), progress=False)

    fn, seen = _recorder()
    evaluate_models(data, MODELS, fn, num_runs=1, checkpoint=Checkpoint(path),
                    progress=False)
    assert {c[0] for c in seen} == {"gemini-2.5-flash", "kimi-k3"}
    assert len(seen) == 2 * 2 * 2


def test_raising_k_tops_up_rather_than_re_scoring(tmp_path):
    """`run` is part of the checkpoint key, so k=1 then k=3 buys runs 1 and 2 only.

    That is the recommended shape at n=1,000 -- k=1 everywhere, k=3 on a subset -- and it
    would be worthless if widening k re-paid for run 0.
    """
    path = tmp_path / "checkpoint.jsonl"
    data = _entries(2)
    evaluate_models(data, [MODELS[0]], _good, num_runs=1, checkpoint=Checkpoint(path),
                    progress=False)

    fn, seen = _recorder()
    out = evaluate_models(data, [MODELS[0]], fn, num_runs=3, checkpoint=Checkpoint(path),
                          progress=False)
    assert len(seen) == 2 * 2 * 2, "runs 1 and 2 only"
    assert len(out["gpt-oss-20b"][0]["runs"]) == 3
    assert all(r["run"] in (0, 1, 2) for r in out["gpt-oss-20b"][0]["runs"])


def test_a_failed_call_is_recorded_as_cost_but_not_as_progress(tmp_path):
    """It billed, so it must appear. It did not answer, so a resume must retry it.

    Dropping it would understate the run the way the lost invocation understated the
    collection run; treating it as done would leave the sweep permanently incomplete.
    """
    path = tmp_path / "checkpoint.jsonl"
    data = _entries(1)
    flaky = {"fail": True}

    def fn(label, entry, key):
        if key == "response_2" and flaky["fail"]:
            raise RuntimeError("rate limit exceeded")
        return _good(label, entry, key)

    out = evaluate_models(data, [MODELS[0]], fn, num_runs=1, retry_passes=0,
                          checkpoint=Checkpoint(path), progress=False)
    # The entry survives and is graded a miss -- see `_assemble`. What matters here is the
    # checkpoint row, which must record the failure as cost without marking it done.
    assert len(out["gpt-oss-20b"]) == 1
    assert out["gpt-oss-20b"][0]["runs"][0]["error"] is True
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert sum(1 for r in rows if r["score"] is None) == 1, "the failure is on the record"

    flaky["fail"] = False
    fn2, seen = _recorder()

    def fn3(label, entry, key):
        return fn2(label, entry, key)

    out2 = evaluate_models(data, [MODELS[0]], fn3, num_runs=1,
                           checkpoint=Checkpoint(path), progress=False)
    assert [c[2] for c in seen] == ["response_2"], "only the failed call is retried"
    assert len(out2["gpt-oss-20b"]) == 1


def test_no_resume_re_scores_but_still_appends(tmp_path):
    """`--no-resume` switches off *reading*. It must not erase what earlier runs paid."""
    path = tmp_path / "checkpoint.jsonl"
    data = _entries(2)
    evaluate_models(data, [MODELS[0]], _good, num_runs=1, checkpoint=Checkpoint(path),
                    progress=False)
    before = len(path.read_text().splitlines())

    fn, seen = _recorder()
    evaluate_models(data, [MODELS[0]], fn, num_runs=1, checkpoint=Checkpoint(path),
                    resume=False, progress=False)
    assert len(seen) == 4, "everything re-scored"
    assert len(path.read_text().splitlines()) == before + 4, "history kept, rows added"


# --- usage lands in the checkpoint -----------------------------------------------------

@patch("src.metaeval.scoring.litellm.completion")
def test_checkpoint_carries_what_each_call_cost(mock_completion, tmp_path):
    """The bug this nearly shipped with: `last_record()` read on the collector thread.

    It is thread-local, and the loop draining futures runs on the main thread, so reading it
    there returned None for every row and wrote a checkpoint with no cost in it at all.
    """
    from types import SimpleNamespace

    from src.metaeval.scoring import _score_model
    from src.metaeval.usage import UsageMeter

    msg = MagicMock()
    msg.content = json.dumps({"score": 0.7, "reasoning": "ok"})
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = SimpleNamespace(prompt_tokens=7000, completion_tokens=300,
                                 completion_tokens_details=None)
    mock_completion.return_value = resp

    meter = UsageMeter().start()
    path = tmp_path / "checkpoint.jsonl"
    evaluate_models(
        _entries(2), [MODELS[0]],
        lambda label, e, k: _score_model(e, k, "fw/x", meter=meter, label=label),
        num_runs=1, max_workers=4, checkpoint=Checkpoint(path),
        last_record=meter.last_record, progress=False)

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(rows) == 4
    assert all(r.get("usage") for r in rows), "every row carries its call's cost"
    assert {r["usage"]["prompt_tokens"] for r in rows} == {7000}
    assert {r["usage"]["label"] for r in rows} == {"gpt-oss-20b"}
    _done, records = Checkpoint(path).load()
    assert sum(r["prompt_tokens"] for r in records) == 4 * 7000


def test_usage_is_optional(tmp_path):
    """A caller with no meter passes no `last_record` and gets rows with no usage."""
    path = tmp_path / "checkpoint.jsonl"
    evaluate_models(_entries(1), [MODELS[0]], _good, num_runs=1,
                    checkpoint=Checkpoint(path), progress=False)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert all("usage" not in r for r in rows)
    assert all(r["score"] is not None for r in rows)


# --- the one-model entry point still behaves -------------------------------------------

def test_evaluate_is_a_one_model_pool():
    """`evaluate()` keeps its `(entry, key) -> float` contract and its shape."""
    results = evaluate(_entries(2), lambda e, k: _good("", e, k), num_runs=1)
    assert len(results) == 2
    assert all(r["runs"][0]["higher_score"] == 1 for r in results)


def test_evaluate_can_checkpoint_too(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    data = _entries(2)
    evaluate(data, lambda e, k: _good("", e, k), num_runs=1, checkpoint=Checkpoint(path))
    calls = {"n": 0}

    def fn(e, k):
        calls["n"] += 1
        return _good("", e, k)

    evaluate(data, fn, num_runs=1, checkpoint=Checkpoint(path))
    assert calls["n"] == 0


def test_default_pool_size_is_eight():
    """The number is a decision, not an accident: 8 across nine models is ~1 per model,
    where the old per-model 4 meant 4 concurrent calls to *one* model."""
    import inspect
    for fn in (evaluate, evaluate_models):
        assert inspect.signature(fn).parameters["max_workers"].default == 8


def test_task_is_hashable_and_compares_by_value():
    """It is the pool's dict key and `_assemble` reconstructs one to look scores up."""
    assert Task("m", 3, "response_1", 0) == Task("m", 3, "response_1", 0)
    assert len({Task("m", 3, "response_1", 0), Task("m", 3, "response_1", 0)}) == 1
    assert Task("m", 3, "response_1", 0) != Task("m", 3, "response_2", 0)


# --- lanes override the parsed prefix --------------------------------------------------

def test_lane_of_overrides_the_prefix_for_admission_control():
    """Two models on the same prefix, different lanes, one of them capped.

    This is the shape the real roster has twice over: a dedicated deployment shares the
    `fireworks_ai` prefix with serverless while sharing none of its limits, and every gateway
    model is addressed as `openai/` while sharing none of OpenAI's.
    """
    import time
    models = [("serverless", "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"),
              ("dedicated", "fireworks_ai/accounts/fireworks/models/qwen3-1p7b#dep/x")]
    lanes = {"serverless": "fw-serverless", "dedicated": "dep:x"}
    live, peak = {"dedicated": 0, "fw-serverless": 0}, {"dedicated": 0, "fw-serverless": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        k = "dedicated" if label == "dedicated" else "fw-serverless"
        with lock:
            live[k] += 1
            peak[k] = max(peak[k], live[k])
        time.sleep(0.02)
        with lock:
            live[k] -= 1
        return 0.5

    evaluate_models(_entries(8), models, fn, num_runs=1, max_workers=8,
                    provider_caps={"dep:x": 1}, lane_of=lambda l: lanes[l], progress=False)
    assert peak["dedicated"] == 1, f"the deployment lane peaked at {peak['dedicated']}"
    assert peak["fw-serverless"] > 1, "and the prefix it shares was NOT capped with it"


def test_lane_of_defaults_to_the_prefix():
    """Every existing caller passes no `lane_of`, so the default must not change behaviour."""
    peak = {"n": 0}
    live = {"n": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        import time
        with lock:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
        time.sleep(0.02)
        with lock:
            live["n"] -= 1
        return 0.5

    evaluate_models(_entries(6), [("g", "gemini/gemini-3.7-flash")], fn, num_runs=1,
                    max_workers=8, provider_caps={"gemini": 2}, progress=False)
    assert peak["n"] <= 2, "the cap still lands via the parsed prefix"


# --- rate limiting ---------------------------------------------------------------------
#
# The failure these exist for: the gateway stated its limit as **300 requests per minute per
# user**, and a concurrency cap of 8 tripped it 109 times in the first 2,100 calls of the full
# sweep. Every 429 in that run was on the gateway; no Fireworks or Gemini row hit one. A semaphore
# bounds calls in flight, which is only the same thing as a rate when latency is constant.

def test_the_rate_limiter_bounds_the_burst_not_just_the_average():
    """**The defect the second failing configuration exposed.** `n` is per minute, and counting
    over a full 60s window permits all of it in the first instant -- a 240/min limiter firing
    240 at once. Measured consequence: 56 req/min of real throughput against a stated 300/min
    limit still drew 18 rejections, because the gateway counts over something far shorter than
    a minute. Smoothing to n/60 per second leaves the average identical and makes the burst
    small.
    """
    import time as _t
    from src.metaeval.scoring import RateLimiter
    rl = RateLimiter(240)
    assert rl.n == 4 and rl.window_s == 1.0, "240/min smoothed to 4/s"
    t0 = _t.monotonic()
    for _ in range(12):
        rl.acquire()
    elapsed = _t.monotonic() - t0
    assert elapsed >= 1.9, f"12 starts at 4/s must take ~2s, took {elapsed:.2f}s"
    assert elapsed < 4.0, "and must not be slower than the average allows"


def test_the_per_minute_average_is_preserved():
    """Smoothing must not cost throughput: the point is the shape of the traffic, not less of
    it. 240/min is still 240/min."""
    from src.metaeval.scoring import RateLimiter
    for per_min in (30, 40, 50, 60, 72, 80, 90, 100, 240, 300, 600):
        rl = RateLimiter(per_min)
        assert rl.rate * 60 == pytest.approx(per_min, rel=1e-9), per_min


def test_a_tiny_budget_still_makes_progress():
    """`max(1, ...)`: 30/min rounds to 0.5 per second, and a limiter that admits nobody is a
    deadlock rather than a throttle."""
    import time as _t
    from src.metaeval.scoring import RateLimiter
    rl = RateLimiter(30)
    assert rl.n >= 1
    t0 = _t.monotonic()
    rl.acquire()
    assert _t.monotonic() - t0 < 1.0


def test_the_rate_limiter_lets_the_window_slide():
    import time as _t
    from src.metaeval.scoring import RateLimiter
    rl = RateLimiter(120, window_s=0.3)      # 2/s, burst 0.6 -> capacity 1 token
    rl.acquire()                             # spends the bucket
    _t.sleep(0.55)                           # 2/s refills a whole token in 0.5s
    t0 = _t.monotonic()
    rl.acquire()
    assert _t.monotonic() - t0 < 0.2, "an accrued token must be spendable immediately"


def test_the_rate_limiter_holds_across_threads():
    """One shared budget, sixteen workers. A per-thread limiter would not bound anything."""
    import time as _t
    from src.metaeval.scoring import RateLimiter
    rl = RateLimiter(240)                    # 4 per second
    started, lock = [], threading.Lock()

    def worker():
        rl.acquire()
        with lock:
            started.append(_t.monotonic())

    t0 = _t.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(started) == 8
    # The token bucket's guarantee: by time t, at most `burst + rate * t` starts have been
    # admitted. The deque formulation this replaced stepped instead -- 4 at once, a full
    # second of silence, 4 more -- so it passed a tighter bound here while presenting the
    # gateway with a burstier stream than the bucket does. Same 240/min average, smoother
    # shape, which is the property the limiter exists for.
    for s in started:
        dt = s - t0
        admitted = len([x for x in started if x <= s])
        assert admitted <= rl.n + rl.rate * dt + 1, (
            f"{admitted} starts by t={dt:.2f}s exceeds burst {rl.n} + {rl.rate}/s")
    assert started[-1] - t0 < 1.6, "8 starts at 4/s with a burst of 4 is ~1s"


def test_lane_rates_throttle_only_their_own_lane():
    """A rate on the gateway must not slow the eight Fireworks rows sharing the pool."""
    import time as _t
    models = [("gated", "openai/claude-haiku-4-5"),
              ("free", "fireworks_ai/accounts/fireworks/models/gpt-oss-20b")]
    lanes = {"gated": "gateway", "free": "fw-serverless"}
    seen, lock = [], threading.Lock()

    def fn(label, entry, key):
        with lock:
            seen.append((label, _t.monotonic()))
        return 0.5

    t0 = _t.monotonic()
    evaluate_models(_entries(4), models, fn, num_runs=1, max_workers=8,
                    lane_rates={"gateway": 60}, lane_of=lambda l: lanes[l],
                    progress=False)
    # the gateway is allowed 60/min = 1 per second, and there are 8 gated calls, so the gateway
    # lane cannot finish quickly -- but the ungated lane must have completed regardless.
    free = [t for lab, t in seen if lab == "free"]
    assert len(free) == 8, "every ungated call ran"
    assert max(free) - t0 < 5, "and none of them waited on the gateway's budget"


def test_a_lane_may_have_a_rate_a_cap_both_or_neither():
    """They compose: the cap bounds calls in flight, the rate bounds starts per minute."""
    import time as _t
    models = [("m", "openai/x")]
    live, peak, lock = {"n": 0}, {"n": 0}, threading.Lock()

    def fn(label, entry, key):
        with lock:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
        _t.sleep(0.02)
        with lock:
            live["n"] -= 1
        return 0.5

    evaluate_models(_entries(6), models, fn, num_runs=1, max_workers=8,
                    provider_caps={"openai": 2}, lane_rates={"openai": 1000},
                    progress=False)
    assert peak["n"] <= 2, "the cap still applies when a rate is also set"


# --- transient failures are retried; deterministic ones are not ------------------------

def test_a_no_score_reply_is_retried_once_and_then_left_alone():
    """**The most expensive thing in the sweep, before this.** 130 calls returned a reply that
    parsed fine and contained no score; each was attempted 3-6 times and never yielded one. At
    temperature 0 the model reproduces the reply, and the failure mode is a `<think>` block that
    runs to the 10,000-token cap -- so every attempt generates the maximum output the budget
    allows. Two further passes over those 130 was on track for five hours, after a main pass
    that had already scored 99.7% of the sweep, and nothing is written until the passes finish.

    Retried **once** rather than zero times: temperature 0 is not bitwise reproducible on a MoE,
    so a truncated trace can occasionally come back whole.
    """
    calls = {"n": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        with lock:
            calls["n"] += 1
        raise ValueError("No score in model response: '<think> ...'")

    evaluate_models(_entries(3), [("m", "fw/x")], fn, num_runs=1, retry_passes=3,
                    retry_delay_s=0, max_workers=1, give_up_after=0, progress=False)
    # 6 calls in the main pass, 6 in retry 1, then dropped -- not 6 more in passes 2 and 3.
    assert calls["n"] == 12, calls["n"]


def test_a_rate_limit_is_retried_in_every_pass():
    """The other half of the distinction: transient failures are what the passes exist for, and
    they must keep being retried."""
    calls = {"n": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        with lock:
            calls["n"] += 1
        raise RuntimeError("RateLimitError: 429 rate limit exceeded")

    evaluate_models(_entries(2), [("m", "fw/x")], fn, num_runs=1, retry_passes=3,
                    retry_delay_s=0, max_workers=1, give_up_after=0, progress=False)
    assert calls["n"] == 16, "4 calls x (1 main + 3 retry passes)"


def test_a_transient_no_score_still_recovers():
    """A reply that lacks a score once and has one next time must not be written off -- that is
    the whole reason the first retry pass still includes them."""
    n = {"i": 0}
    lock = threading.Lock()

    def fn(label, entry, key):
        with lock:
            n["i"] += 1
            i = n["i"]
        if i <= 2:
            raise ValueError("No score in model response: '<think> ...'")
        return 0.9 if key == "response_1" else 0.2

    out = evaluate_models(_entries(1), [("m", "fw/x")], fn, num_runs=1, retry_passes=3,
                          retry_delay_s=0, max_workers=1, progress=False)
    assert len(out["m"]) == 1, "recovered on the first retry pass"
    assert out["m"][0]["runs"][0]["correct"] is True


def test_mixed_failures_keep_their_own_retry_rules():
    """One model failing deterministically must not stop another's rate limits from retrying."""
    calls = collections.Counter()
    lock = threading.Lock()

    def fn(label, entry, key):
        with lock:
            calls[label] += 1
        if label == "stuck":
            raise ValueError("No score in model response: 'prose'")
        raise RuntimeError("429 rate limit exceeded")

    evaluate_models(_entries(1), [("stuck", "fw/a"), ("flaky", "fw/b")], fn, num_runs=1,
                    retry_passes=3, retry_delay_s=0, max_workers=1, give_up_after=0,
                    progress=False)
    assert calls["stuck"] == 4, "2 calls, main pass + one retry"
    assert calls["flaky"] == 8, "2 calls, main pass + three retries"


def test_the_rate_limiter_represents_rates_that_are_not_multiples_of_sixty():
    """The bug this replaced: an integer count per 1s window can only express n*60/min.

    `round(n_per_min * window_s / 60)` sent 100/min to 2 per second -- 120/min, i.e. 20% ABOVE
    a limit it was configured to respect -- and 72/min down to 1 per second, 60/min, giving
    away throughput nobody asked to give away. Both directions were silent.
    """
    from src.metaeval.scoring import RateLimiter
    for per_min, want_burst in ((72, 1.2), (100, 100 / 60), (40, 1.0)):
        rl = RateLimiter(per_min)
        assert rl.rate * 60 == pytest.approx(per_min, rel=1e-9)
        assert rl.n == pytest.approx(want_burst), "burst is rate*window, floored at one token"


def test_a_fractional_rate_paces_over_a_longer_run():
    """72/min is 1.2/s: eight starts must take ~5.8s, which an integer limiter cannot produce.

    Under the old formulation this took 7 whole seconds (1/s), and under a naive ceil it would
    have taken 3.5s (2/s). The point of the float is that neither number is what you get.
    """
    import time as _t
    from src.metaeval.scoring import RateLimiter
    rl = RateLimiter(72)                    # 1.2/s, burst 1.2 -> 1 free then 7 paced
    t0 = _t.monotonic()
    for _ in range(8):
        rl.acquire()
    elapsed = _t.monotonic() - t0
    assert 5.0 <= elapsed <= 6.6, f"7 paced starts at 1.2/s is ~5.8s, took {elapsed:.2f}s"
