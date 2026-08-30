"""Tests for `src/metaeval/usage.py`.

The arithmetic here ends up in the paper's cost and time table, so the cases that matter
most are the ones where a naive implementation would silently under-report: failed calls,
retry turns, and reasoning tokens hidden inside completion tokens.
"""

from types import SimpleNamespace

import pytest

from src.metaeval.usage import (
    CallRecord, UsageMeter, derive_reasoning_tokens, extract_content, extract_usage,
    merge, timed_call,
)


def _resp(prompt=100, completion=50, reasoning=None, content="x" * 40, trace=False):
    """A litellm-shaped response. `reasoning=None` omits the details object entirely,
    which is what Fireworks actually returns on every model measured."""
    details = (SimpleNamespace(reasoning_tokens=reasoning)
               if reasoning is not None else None)
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                            completion_tokens_details=details)
    # `trace` may be True (a short stand-in) or a string, when the test needs a realistic
    # trace length -- the derived share depends on it, so 5 chars would understate it.
    rc = ("think" if trace is True else trace) if trace else None
    msg = SimpleNamespace(content=content, reasoning_content=rc)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg)])


# --- extract_usage: every field is optional in practice ---

def test_extract_usage_reads_all_three():
    assert extract_usage(_resp(120, 60, 40)) == (120, 60, 40)


def test_extract_usage_no_usage_object():
    assert extract_usage(SimpleNamespace()) == (0, 0, None)


def test_unreported_reasoning_is_none_not_zero():
    """The bug this replaced. Fireworks returns `completion_tokens_details: null` on
    gpt-oss-20b and nemotron-lightning, both of which reason heavily. Reporting 0 asserted
    they did not -- on a call that spent 174 tokens to emit 12 characters."""
    assert extract_usage(_resp(10, 5))[2] is None


def test_extract_usage_none_fields_are_zero():
    """A provider reporting `None` is normal, and must not propagate into arithmetic."""
    usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=None))
    assert extract_usage(SimpleNamespace(usage=usage)) == (0, 0, None)


def test_reported_zero_reasoning_stays_zero():
    """A provider that reports 0 is making a measurement, and it must survive."""
    assert extract_usage(_resp(10, 5, reasoning=0))[2] == 0


def test_extract_usage_accepts_string_counts():
    usage = SimpleNamespace(prompt_tokens="100", completion_tokens="50",
                            completion_tokens_details=None)
    assert extract_usage(SimpleNamespace(usage=usage)) == (100, 50, None)


# --- derived reasoning tokens ---
#
# Fireworks reports no reasoning breakdown on any model measured, so the share is derived
# from the character split between the reasoning trace and the answer. These pin it against
# real observed calls.

def test_derive_matches_a_real_gpt_oss_call():
    """Observed: 82 completion tokens, 247 reasoning chars, 12 content chars.
    `{"score": 1}` is ~4-6 tokens, so 78/4 is the right split."""
    assert derive_reasoning_tokens(82, content_chars=12, reasoning_chars=247) == 78


def test_derive_matches_a_real_nemotron_call():
    """Observed: 281 completion tokens, 928 reasoning chars, 12 content chars."""
    assert derive_reasoning_tokens(281, content_chars=12, reasoning_chars=928) == 277


def test_derive_is_zero_without_a_trace():
    """No reasoning text is a real measurement of no reasoning, not missing data."""
    assert derive_reasoning_tokens(50, content_chars=200, reasoning_chars=0) == 0


def test_derive_never_exceeds_the_completion():
    assert derive_reasoning_tokens(100, content_chars=0, reasoning_chars=9999) == 100


def test_derive_handles_degenerate_inputs():
    assert derive_reasoning_tokens(0, 0, 0) == 0
    assert derive_reasoning_tokens(100, 0, 0) == 0


def test_derive_is_proportional():
    assert derive_reasoning_tokens(100, content_chars=50, reasoning_chars=50) == 50
    assert derive_reasoning_tokens(100, content_chars=25, reasoning_chars=75) == 75


def test_reasoning_share_is_the_headline():
    """The real gpt-oss-20b call: 174 output tokens, a 247-char trace, a 12-char answer."""
    m = UsageMeter().start()
    m.record(_resp(93, 174, content='{"score": 1}', trace="r" * 247),
             model="x", latency_s=1.0)
    s = m.summary()
    assert s["reasoning_share"] > 0.9, "nearly the whole completion was invisible"
    assert s["reasoning_source"] == "derived"
    assert s["visible_tokens"] == s["completion_tokens"] - s["reasoning_tokens"]


def test_provider_number_wins_when_reported():
    """If a provider ever does report it, we must not overwrite it with an estimate."""
    m = UsageMeter().start()
    m.record(_resp(100, 500, reasoning=450, content="x" * 10, trace=True),
             model="x", latency_s=1.0)
    s = m.summary()
    assert s["reasoning_tokens"] == 450
    assert s["reasoning_source"] == "reported"


def test_mixed_sources_are_labelled_mixed():
    m = UsageMeter().start()
    m.record(_resp(100, 100, reasoning=50), model="a", latency_s=1.0)
    m.record(_resp(100, 100, trace=True), model="b", latency_s=1.0)
    assert m.summary()["reasoning_source"] == "mixed"


def test_json_object_regression_would_have_been_caught():
    """472 -> 2,996 output tokens for an identical reply. The reasoning share is what
    separates "a chattier model" from "wasted output"."""
    before = UsageMeter().start()
    before.record(_resp(800, 472, content="j" * 300, trace=True), model="x", latency_s=1.0)
    after = UsageMeter().start()
    # Same answer, vastly longer trace. Added directly rather than recorded-then-overwritten,
    # which left a discarded response a reader had to rule out.
    after.add(CallRecord(
        model="x", completion_tokens=2996, content_chars=300, reasoning_chars=12_000,
        reasoning_tokens=derive_reasoning_tokens(2996, 300, 12_000),
        reasoning_source="derived"))
    assert after.summary()["reasoning_share"] > before.summary()["reasoning_share"]


# --- inline <think> tags, as the qwen3 family emits them ---

def test_inline_think_tag_counts_as_reasoning():
    resp = _resp(100, 200, content="<think>lots of deliberation here</think>{\"score\": 1}")
    chars, reason_chars = extract_content(resp)
    assert reason_chars == len("lots of deliberation here")
    assert chars == len('{"score": 1}'), "the tag must not count as visible answer"


def test_unclosed_think_tag_is_still_counted():
    """A reply truncated at max_tokens mid-thought has no closing tag -- and is exactly the
    runaway worth catching."""
    resp = _resp(100, 4000, content="<think>" + "a" * 500)
    chars, reason_chars = extract_content(resp)
    assert reason_chars == 500 and chars == 0


def test_separate_field_and_inline_tag_both_count():
    resp = _resp(100, 200, content="<think>inline</think>ok", trace=True)
    chars, reason_chars = extract_content(resp)
    assert reason_chars == len("think") + len("inline")
    assert chars == len("ok")


def test_content_extraction_survives_a_malformed_response():
    assert extract_content(SimpleNamespace(choices=[])) == (0, 0)
    assert extract_content(SimpleNamespace()) == (0, 0)


def test_none_content_with_a_trace():
    msg = SimpleNamespace(content=None, reasoning_content="thinking hard")
    assert extract_content(SimpleNamespace(choices=[SimpleNamespace(message=msg)])) == \
        (0, len("thinking hard"))


# --- totals ---

def test_totals_sum_across_calls():
    m = UsageMeter().start()
    m.record(_resp(100, 50, 20), model="x", latency_s=1.0)
    m.record(_resp(200, 60, 10), model="x", latency_s=2.0)
    s = m.summary()
    assert s["prompt_tokens"] == 300
    assert s["completion_tokens"] == 110
    assert s["reasoning_tokens"] == 30
    assert s["total_tokens"] == 410


def test_reasoning_is_inside_completion_not_added_to_it():
    """Reasoning tokens are a subset of completion tokens. Double-counting them would
    inflate the total by the size of the very thing that is hardest to notice."""
    m = UsageMeter().start()
    m.record(_resp(100, 500, 450), model="x", latency_s=1.0)
    s = m.summary()
    assert s["total_tokens"] == 600          # 100 + 500, NOT 100 + 500 + 450
    assert s["reasoning_share"] == pytest.approx(0.9)


def test_empty_meter_reports_zeros_not_crashes():
    s = UsageMeter().summary()
    assert s["n_calls"] == 0 and s["total_tokens"] == 0
    assert s["tokens_per_ok_call"] is None   # no calls, not "zero tokens per call"
    assert s["reasoning_share"] is None      # no completion tokens, so no share exists
    assert s["reasoning_tokens"] == 0


# --- failed calls: the case that made the most expensive model look cheapest ---

def test_errors_are_counted_not_dropped():
    m = UsageMeter().start()
    m.record(_resp(100, 50), model="x", latency_s=1.0)
    m.record_error(RuntimeError("rate limit exceeded"), model="x", latency_s=3.0)
    s = m.summary()
    assert s["n_calls"] == 2
    assert s["n_ok"] == 1
    assert s["n_error"] == 1


def test_error_latency_still_counts_toward_service_time():
    """A rate-limited call blocks a worker for real seconds. Excluding it would make a
    heavily throttled run look fast."""
    m = UsageMeter().start()
    m.record_error(RuntimeError("429"), model="x", latency_s=30.0)
    assert m.summary()["service_s"] == pytest.approx(30.0)


def test_tokens_per_call_divides_by_successes_only():
    """Averaging over failures (0 tokens) would report a throttled model as terse."""
    m = UsageMeter().start()
    m.record(_resp(100, 100), model="x", latency_s=1.0)
    m.record_error(RuntimeError("boom"), model="x", latency_s=1.0)
    assert m.summary()["tokens_per_ok_call"] == pytest.approx(200.0)


def test_error_message_is_captured_and_truncated():
    m = UsageMeter()
    rec = m.record_error(ValueError("x" * 500), model="x", latency_s=0.1)
    assert rec.error.startswith("ValueError: ")
    assert len(rec.error) < 250


def test_all_calls_failed_is_distinguishable_from_no_calls():
    """The distinction `run_evaluators` already learned the hard way for accuracy."""
    failed = UsageMeter().start()
    failed.record_error(RuntimeError("boom"), model="x", latency_s=1.0)
    assert failed.summary()["n_calls"] == 1
    assert UsageMeter().summary()["n_calls"] == 0


# --- timed_call ---

def test_timed_call_returns_the_response_and_records_it():
    m = UsageMeter().start()
    resp = _resp(10, 20)
    got = timed_call(lambda: resp, m, model="x", role="judge")
    assert got is resp
    assert m.summary()["n_calls"] == 1
    assert m.records[0].role == "judge"


def test_timed_call_reraises_and_still_records():
    m = UsageMeter().start()
    with pytest.raises(RuntimeError, match="boom"):
        timed_call(lambda: (_ for _ in ()).throw(RuntimeError("boom")), m, model="x")
    assert m.summary()["n_error"] == 1


def test_timed_call_with_no_meter_is_a_passthrough():
    """`meter=None` must stay free, so instrumentation never becomes mandatory."""
    resp = _resp()
    assert timed_call(lambda: resp, None, model="x") is resp


def test_timed_call_measures_positive_latency():
    m = UsageMeter().start()
    timed_call(lambda: _resp(), m, model="x")
    assert m.records[0].latency_s >= 0.0


# --- attempts: a retry is bigger than the call it retries ---

def test_retry_turns_are_separate_records():
    m = UsageMeter().start()
    m.record(_resp(500, 300), model="x", label="judge1", attempt=1, latency_s=1.0)
    m.record(_resp(830, 20), model="x", label="judge1", attempt=2, latency_s=1.0)
    s = m.summary()
    assert s["n_calls"] == 2
    assert s["prompt_tokens"] == 1330, "the retry's prompt must be counted, not replace"
    by = m.by("attempt")
    assert by["1"]["n_calls"] == 1 and by["2"]["n_calls"] == 1


def test_by_label_partitions_the_calls():
    m = UsageMeter().start()
    m.record(_resp(100, 10), model="a", label="judge1", latency_s=1.0)
    m.record(_resp(200, 20), model="b", label="judge2", latency_s=1.0)
    by = m.by("label")
    assert set(by) == {"judge1", "judge2"}
    assert by["judge1"]["total_tokens"] == 110
    assert by["judge2"]["total_tokens"] == 220


def test_by_group_totals_reconstruct_the_parent():
    m = UsageMeter().start()
    for i, lab in enumerate(["a", "b", "a", "c"]):
        m.record(_resp(10 * (i + 1), 5), model="m", label=lab, latency_s=1.0)
    by = m.by("label")
    assert sum(g["total_tokens"] for g in by.values()) == m.summary()["total_tokens"]


def test_by_groups_keep_the_parent_wall_clock():
    """Groups ran interleaved in one window; splitting elapsed time between them would
    invent per-group durations that no clock measured."""
    m = UsageMeter()
    m._t0, m._t1 = 100.0, 200.0
    m.add(CallRecord(model="m", label="a", latency_s=10.0))
    m.add(CallRecord(model="m", label="b", latency_s=10.0))
    by = m.by("label")
    assert by["a"]["wall_s"] == 100.0 and by["b"]["wall_s"] == 100.0


# --- service time vs wall clock ---

def test_concurrency_is_service_over_wall():
    m = UsageMeter()
    m._t0, m._t1 = 0.0, 10.0
    for _ in range(4):
        m.add(CallRecord(model="m", latency_s=10.0))
    s = m.summary()
    assert s["wall_s"] == 10.0
    assert s["service_s"] == 40.0
    assert s["concurrency"] == pytest.approx(4.0)


def test_serial_run_has_concurrency_one():
    m = UsageMeter()
    m._t0, m._t1 = 0.0, 30.0
    for _ in range(3):
        m.add(CallRecord(model="m", latency_s=10.0))
    assert m.summary()["concurrency"] == pytest.approx(1.0)


def test_out_tps_uses_elapsed_not_service_time():
    """Rate limits are provoked by tokens per *elapsed* second -- what the provider sees."""
    m = UsageMeter()
    m._t0, m._t1 = 0.0, 10.0
    for _ in range(2):
        m.add(CallRecord(model="m", completion_tokens=1000, latency_s=8.0))
    assert m.summary()["out_tps"] == pytest.approx(200.0)


def test_zero_wall_clock_does_not_divide_by_zero():
    m = UsageMeter()
    m._t0 = m._t1 = 5.0
    m.add(CallRecord(model="m", completion_tokens=10, latency_s=0.0))
    s = m.summary()
    assert s["concurrency"] is None and s["out_tps"] is None


def test_unstarted_meter_infers_a_window_from_the_first_call():
    """Forgetting `.start()` must not make wall clock 0 and concurrency undefined."""
    m = UsageMeter()
    m.record(_resp(), model="x", latency_s=2.0)
    assert m.wall_s() >= 2.0


# --- latency distribution ---

def test_latency_percentiles_are_observed_values():
    m = UsageMeter().start()
    for lat in (1.0, 2.0, 3.0, 100.0):
        m.add(CallRecord(model="m", latency_s=lat))
    s = m.summary()
    assert s["latency_p95_s"] == 100.0, "p95 must surface the outlier, not average it away"
    assert s["latency_mean_s"] == pytest.approx(26.5)


def test_single_call_percentiles_are_that_call():
    m = UsageMeter().start()
    m.add(CallRecord(model="m", latency_s=7.0))
    s = m.summary()
    assert s["latency_p50_s"] == 7.0 and s["latency_p95_s"] == 7.0


# --- thread safety ---

def test_concurrent_records_are_not_lost():
    from concurrent.futures import ThreadPoolExecutor
    m = UsageMeter().start()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: m.record(_resp(10, 5), model="x", latency_s=0.01),
                    range(400)))
    s = m.summary()
    assert s["n_calls"] == 400
    assert s["total_tokens"] == 400 * 15


# --- merge ---

def test_merge_carries_a_reasoning_share():
    a, b = UsageMeter().start(), UsageMeter().start()
    a.record(_resp(100, 100, reasoning=30), model="a", latency_s=1.0)
    b.record(_resp(100, 100, content="x" * 10, trace=True), model="b", latency_s=1.0)
    t = merge({"a": a, "b": b})["total"]
    assert t["reasoning_tokens"] > 30
    assert 0.0 < t["reasoning_share"] <= 1.0
    assert t["visible_tokens"] == t["completion_tokens"] - t["reasoning_tokens"]


def test_merge_sums_serial_models():
    a, b = UsageMeter(), UsageMeter()
    a._t0, a._t1 = 0.0, 10.0
    b._t0, b._t1 = 10.0, 40.0
    a.add(CallRecord(model="a", prompt_tokens=100, completion_tokens=50, latency_s=9.0))
    b.add(CallRecord(model="b", prompt_tokens=200, completion_tokens=60, latency_s=29.0))
    r = merge({"a": a, "b": b})
    assert r["total"]["total_tokens"] == 410
    assert r["total"]["wall_s"] == pytest.approx(40.0)   # serial: summed
    assert set(r["per_model"]) == {"a", "b"}


def test_merge_of_nothing_is_empty_not_an_error():
    r = merge({})
    assert r["per_model"] == {}
    assert r["total"]["total_tokens"] == 0
    assert r["total"]["out_tps"] is None


# --- serialisation ---

def test_to_json_is_json_serialisable_and_keeps_every_record():
    import json
    m = UsageMeter().start()
    m.record(_resp(10, 5, 2), model="x", label="judge1", item_id="pair-1", latency_s=1.0)
    m.record_error(RuntimeError("boom"), model="x", label="judge1", item_id="pair-2",
                   latency_s=1.0)
    blob = json.loads(json.dumps(m.to_json()))
    assert len(blob["records"]) == 2
    assert blob["summary"]["n_error"] == 1
    assert blob["records"][0]["item_id"] == "pair-1"


def test_record_total_tokens_property():
    assert CallRecord(model="m", prompt_tokens=7, completion_tokens=3).total_tokens == 10


# --- per-item and per-stage roll-ups ---
#
# What replaces per-stage wall clock in a pooled run. Once pair 7's instruction generation
# overlaps pair 2's judging there is no interval belonging to a stage, so these roll-ups
# have to be built from summed latency, which is additive, and must never be presented as
# elapsed time.

def _rec(item, role, lat, prompt=100, completion=10, attempt=1, ok=True):
    return CallRecord(model="m", role=role, item_id=item, latency_s=lat,
                      prompt_tokens=prompt, completion_tokens=completion,
                      attempt=attempt, ok=ok)


def test_item_stage_matrix_splits_a_pairs_chain_by_stage():
    """The shape the cost claim quotes: what one pair spent in each stage."""
    m = UsageMeter().start()
    m.add(_rec("pair-1", "instruction-gen", 3.0))
    m.add(_rec("pair-1", "trajectory", 20.0))
    m.add(_rec("pair-1", "trajectory", 21.0))
    m.add(_rec("pair-1", "judge", 5.0))
    row = m.item_stage_matrix()["pair-1"]
    assert row["stages"] == {"instruction-gen": 3.0, "judge": 5.0, "trajectory": 41.0}
    assert row["service_s"] == pytest.approx(49.0)
    assert row["n_calls"] == 4


def test_item_stage_matrix_separates_items():
    m = UsageMeter().start()
    m.add(_rec("pair-1", "judge", 5.0))
    m.add(_rec("pair-2", "judge", 9.0))
    mat = m.item_stage_matrix()
    assert set(mat) == {"pair-1", "pair-2"}
    assert mat["pair-2"]["service_s"] == pytest.approx(9.0)


def test_item_stage_matrix_distinguishes_slow_from_retried():
    """A pair that looks expensive only because it retried is a different problem from a
    pair that is genuinely slow, and the fix differs."""
    m = UsageMeter().start()
    m.add(_rec("slow", "trajectory", 60.0))
    m.add(_rec("retried", "trajectory", 20.0, attempt=1, ok=False))
    m.add(_rec("retried", "trajectory", 20.0, attempt=2))
    mat = m.item_stage_matrix()
    assert mat["slow"]["retries"] == 0
    assert mat["retried"]["retries"] == 1
    assert mat["retried"]["n_error"] == 1


def test_item_stage_matrix_carries_tokens_per_stage():
    """Tokens per stage per pair is how cost-per-kept-pair gets attributed."""
    m = UsageMeter().start()
    m.add(_rec("pair-1", "instruction-gen", 1.0, prompt=800, completion=200))
    m.add(_rec("pair-1", "judge", 1.0, prompt=3000, completion=100))
    row = m.item_stage_matrix()["pair-1"]
    assert row["stage_tokens"] == {"instruction-gen": 1000, "judge": 3100}
    assert row["total_tokens"] == 4100


def test_item_stage_matrix_is_empty_not_broken_with_no_records():
    assert UsageMeter().item_stage_matrix() == {}


def test_unattributed_calls_are_kept_under_an_empty_key():
    """An unattributed call is still spend. Dropping it would understate the total."""
    m = UsageMeter().start()
    m.add(_rec("", "judge", 4.0))
    mat = m.item_stage_matrix()
    assert "" in mat and mat[""]["service_s"] == pytest.approx(4.0)


def test_stage_service_is_additive_across_items():
    """The property per-stage wall clock lacks: these sum, elapsed times do not."""
    m = UsageMeter().start()
    m.add(_rec("pair-1", "trajectory", 10.0))
    m.add(_rec("pair-2", "trajectory", 15.0))
    m.add(_rec("pair-1", "judge", 5.0))
    assert m.stage_service() == {"judge": 5.0, "trajectory": 25.0}


def test_stage_service_totals_match_the_run_service_time():
    m = UsageMeter().start()
    for i, (role, lat) in enumerate([("a", 1.0), ("b", 2.0), ("a", 3.0)]):
        m.add(_rec(f"p{i}", role, lat))
    assert sum(m.stage_service().values()) == pytest.approx(m.summary()["service_s"])


def test_item_service_times_sum_to_the_run_service_time():
    """No pair's time is double-counted or lost."""
    m = UsageMeter().start()
    m.add(_rec("p1", "trajectory", 11.0))
    m.add(_rec("p2", "trajectory", 7.0))
    m.add(_rec("p2", "judge", 2.5))
    total = sum(v["service_s"] for v in m.item_stage_matrix().values())
    assert total == pytest.approx(m.summary()["service_s"])


def test_grouping_by_item_gives_a_full_summary_per_pair():
    m = UsageMeter().start()
    m.add(_rec("p1", "judge", 3.0, prompt=100, completion=50))
    per = m.by("item_id")
    assert per["p1"]["total_tokens"] == 150
    assert per["p1"]["n_calls"] == 1


def test_per_item_wall_clock_is_the_runs_not_the_items():
    """`by()` inherits the parent window by design, so a group's `wall_s` is the whole run's.
    That is why `item_stage_matrix` reports `service_s` and no per-item wall clock: the
    gaps between a pair's calls, queued behind other workers, are unmeasurable from records
    that carry a duration and no start timestamp."""
    m = UsageMeter()
    m._t0, m._t1 = 0.0, 100.0
    m.add(_rec("p1", "judge", 2.0))
    m.add(_rec("p2", "judge", 3.0))
    per = m.by("item_id")
    assert per["p1"]["wall_s"] == 100.0 and per["p2"]["wall_s"] == 100.0
    assert "wall_s" not in m.item_stage_matrix()["p1"]


# --- untimed calls: whether service time may be quoted as a total ---

def test_untimed_calls_counts_calls_that_spent_tokens_but_recorded_no_time():
    """The measured case this exists for: tau3's role flip copied cost/usage/raw_data off the
    timed response and dropped `generation_time_seconds`, so every user-simulator call was
    counted in tokens and not in seconds -- 417 of 1097 trajectory calls, 18% of simulation
    time unattributable."""
    m = UsageMeter()
    m.add(CallRecord(model="tau3:assistant", role="trajectory", label="ok",
                     prompt_tokens=1000, completion_tokens=100, latency_s=1.4))
    m.add(CallRecord(model="tau3:user", role="trajectory", label="ok",
                     prompt_tokens=800, completion_tokens=40, latency_s=0.0))
    assert m.untimed_calls() == {"trajectory": 1}


def test_a_fully_timed_stage_reports_nothing():
    """**The caveat has to disappear once it stops being true.** A report that hardcodes
    "the user simulator is untimed" keeps printing it after the patch makes it false, which
    is the same failure as omitting it while it is true."""
    m = UsageMeter()
    for _ in range(3):
        m.add(CallRecord(model="tau3:assistant", role="trajectory", label="ok",
                         prompt_tokens=500, completion_tokens=50, latency_s=0.8))
    assert m.untimed_calls() == {}


def test_a_call_that_spent_no_tokens_is_not_untimed():
    """Tool results and the scripted opener make no LLM call, so they have nothing to time.
    Counting them would make every run look like a floor forever."""
    m = UsageMeter()
    m.add(CallRecord(model="tau3:tool", role="trajectory", label="ok",
                     prompt_tokens=0, completion_tokens=0, latency_s=0.0))
    assert m.untimed_calls() == {}


def test_a_failed_call_is_not_counted_as_untimed():
    """A call that raised has no latency to record and is already counted by `n_error`;
    counting it again here would flag a floor caused by a failure rather than by a gap in
    instrumentation."""
    m = UsageMeter()
    m.record_error(RuntimeError("boom"), model="tau3:sim", latency_s=0.0,
                   role="trajectory", label="ok")
    assert m.untimed_calls() == {}


def test_untimed_calls_are_reported_per_stage():
    """Stage-level, because that is the granularity service time is quoted at -- a floor on
    trajectories says nothing about whether the judge figure is complete."""
    m = UsageMeter()
    m.add(CallRecord(model="tau3:user", role="trajectory", label="ok",
                     prompt_tokens=10, completion_tokens=1, latency_s=0.0))
    m.add(CallRecord(model="judge", role="judge", label="judge1",
                     prompt_tokens=10, completion_tokens=1, latency_s=2.0))
    assert m.untimed_calls() == {"trajectory": 1}
