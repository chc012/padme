import json
from unittest.mock import MagicMock, patch

import pytest

from src.metaeval.scoring import (
    _score_model,
    _pred_run,
    compute_metrics,
    evaluate,
    load_dataset,
)


# --- helpers ---

def _make_entry(entry_id="id-1", criterion="clarity", correct_response=1):
    return {
        "id": entry_id,
        "criterion_name": criterion,
        "criterion_description": "Is the response clear?",
        "positive_hint": "Write clearly.",
        "negative_hint": "Write confusingly.",
        "prompt": "Tell me something.",
        "response_1": "Clear response.",
        "response_2": "Confusing response.",
        "correct_response": correct_response,
    }


def _make_litellm_response(score: float, reasoning: str = "ok"):
    content = json.dumps({"score": score, "reasoning": reasoning})
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# --- load_dataset ---

def test_load_dataset(tmp_path):
    data = [_make_entry()]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data))
    assert load_dataset(str(p)) == data


# --- _score_model ---

@patch("src.metaeval.scoring.litellm.completion")
def test_score_model_returns_float(mock_completion):
    mock_completion.return_value = _make_litellm_response(0.75)
    score = _score_model(_make_entry(), "response_1", "openai/gpt-5.4")
    assert score == pytest.approx(0.75)


@patch("src.metaeval.scoring.litellm.completion")
def test_score_model_uses_correct_response_text(mock_completion):
    mock_completion.return_value = _make_litellm_response(0.5)
    _score_model(_make_entry(), "response_2", "openai/gpt-5.4")
    content = mock_completion.call_args[1]["messages"][0]["content"]
    assert "Confusing response." in content
    assert "Clear response." not in content


@patch("src.metaeval.scoring.litellm.completion")
def test_score_model_includes_criterion(mock_completion):
    mock_completion.return_value = _make_litellm_response(0.5)
    _score_model(_make_entry(), "response_1", "openai/gpt-5.4")
    content = mock_completion.call_args[1]["messages"][0]["content"]
    assert "clarity" in content
    assert "Is the response clear?" in content


# --- _score_endpoint ---

# --- evaluate ---

def _make_run(run_idx, s1, s2, correct_response):
    higher_score = None if s1 == s2 else (1 if s1 > s2 else 2)
    return {"run": run_idx, "score_1": s1, "score_2": s2,
            "higher_score": higher_score, "correct": higher_score == correct_response}


def _make_result(criterion="clarity", correct_response=1, runs=None):
    if runs is None:
        runs = [_make_run(0, 0.8, 0.4, correct_response)]
    import statistics as _stats
    s1s = [r["score_1"] for r in runs]
    s2s = [r["score_2"] for r in runs]
    k = len(runs)
    return {
        "criterion_name": criterion,
        "correct_response": correct_response,
        "runs": runs,
        "pass_rate": sum(r["correct"] for r in runs) / k,
        "score_1_mean": _stats.mean(s1s),
        "score_1_std": _stats.stdev(s1s) if k > 1 else 0.0,
        "score_2_mean": _stats.mean(s2s),
        "score_2_std": _stats.stdev(s2s) if k > 1 else 0.0,
    }


def test_evaluate_output_schema():
    score_fn = lambda e, k: 0.8 if k == "response_1" else 0.4
    results = evaluate([_make_entry(correct_response=1)], score_fn, num_runs=1)
    assert len(results) == 1
    r = results[0]
    assert len(r["runs"]) == 1
    assert r["runs"][0]["score_1"] == pytest.approx(0.8)
    assert r["runs"][0]["score_2"] == pytest.approx(0.4)
    assert r["runs"][0]["higher_score"] == 1
    assert r["correct_response"] == 1
    assert r["runs"][0]["correct"] is True
    assert r["pass_rate"] == pytest.approx(1.0)
    assert "score_1_mean" in r
    assert "score_1_std" in r


def test_evaluate_correct_when_higher_matches():
    score_fn = lambda e, k: 0.3 if k == "response_1" else 0.9
    results = evaluate([_make_entry(correct_response=2)], score_fn, num_runs=1)
    assert results[0]["runs"][0]["higher_score"] == 2
    assert results[0]["runs"][0]["correct"] is True


def test_evaluate_incorrect_when_higher_mismatches():
    score_fn = lambda e, k: 0.3 if k == "response_1" else 0.9
    results = evaluate([_make_entry(correct_response=1)], score_fn, num_runs=1)
    assert results[0]["runs"][0]["higher_score"] == 2
    assert results[0]["runs"][0]["correct"] is False


def test_a_permanently_failing_call_is_scored_wrong_not_dropped():
    """**Semantics changed deliberately**: this used to assert the entry was discarded.

    Dropping it grades a model on the subset it managed to answer, which flatters exactly the
    models least able to do the job -- and the failures are concentrated there: of 130
    unrecoverable calls in a 50,000-call sweep, 81 were `llama3.1-8b` and 47 `qwen3-1p7b`, the
    two lowest-ranked models. An evaluator that cannot emit a parseable score has not
    functionally finished the example, which is the same argument that makes a tie wrong.

    `retry_passes=0` because the default is 3 passes with a 20s pause each, and a test that
    sleeps 60s to prove a permanent failure stays permanent is a test nobody will run.
    """
    def score_fn(e, k):
        if k == "response_1":
            raise Exception("API error")
        return 0.5

    results = evaluate([_make_entry()], score_fn, num_runs=1, retry_passes=0)
    assert len(results) == 1, "the entry is kept so `n` stays the dataset size"
    run = results[0]["runs"][0]
    assert run["correct"] is False
    assert run["error"] is True, "and is distinguishable from a tie"
    assert run["score_1"] is None and run["score_2"] == pytest.approx(0.5)
    assert results[0]["score_1_mean"] is None, "no score, so no mean -- not 0.0"
    assert results[0]["score_2_mean"] == pytest.approx(0.5)


def test_a_model_that_answered_nothing_is_still_dropped():
    """The one case that must NOT become 0% accuracy. `qwen3-4b` and `qwen3-1p7b` once reported
    0.0% accuracy with 0.0% errors after their deployments slept through their turns, and that
    reads as a uniquely terrible evaluator rather than an absent one. Upstream renders `[]` as
    NO DATA."""
    def score_fn(e, k):
        raise Exception("deployment did not come up")

    assert evaluate([_make_entry(), _make_entry("id-2")], score_fn, num_runs=1,
                    retry_passes=0) == []


def test_a_tie_and_an_error_are_told_apart():
    """Both name no winner and both count wrong, but they are different failures and the tie
    rate must not absorb the errors."""
    def score_fn(e, k):
        if e["id"] == "id-2" and k == "response_2":
            raise Exception("API error")
        return 0.5                                   # id-1 ties

    results = evaluate([_make_entry("id-1"), _make_entry("id-2")], score_fn,
                       num_runs=1, retry_passes=0)
    by_id = {r["id"]: r["runs"][0] for r in results}
    assert by_id["id-1"]["higher_score"] is None and "error" not in by_id["id-1"]
    assert by_id["id-2"]["higher_score"] is None and by_id["id-2"]["error"] is True
    assert not by_id["id-1"]["correct"] and not by_id["id-2"]["correct"]


def test_evaluate_recovers_a_transient_failure():
    """The reason retries exist: one flaky call must not discard the whole entry.

    An entry needs all `2 * num_runs` calls to succeed, so at k=3 a single rate-limit blip
    out of six used to drop the pair. nemotron-lightning lost 7 of 36 entries that way, to
    errors that were not the model's fault, on a dataset of 36 hand-annotated pairs.
    """
    calls = {"n": 0}

    def score_fn(e, k):
        calls["n"] += 1
        if calls["n"] == 1:          # first call fails once, then the world is fine
            raise Exception("rate limit exceeded")
        return 0.8 if k == "response_1" else 0.3

    results = evaluate([_make_entry(correct_response=1)], score_fn, num_runs=1,
                       retry_passes=2, retry_delay_s=0)
    assert len(results) == 1, "a transient failure must not cost the entry"
    assert results[0]["runs"][0]["correct"] is True


def test_evaluate_retry_gives_up_after_the_configured_passes():
    """Retries are bounded: a permanent failure must not loop forever."""
    attempts = {"n": 0}

    def score_fn(e, k):
        attempts["n"] += 1
        raise Exception("permanent")

    assert evaluate([_make_entry()], score_fn, num_runs=1,
                    retry_passes=2, retry_delay_s=0) == []
    # 2 calls in the first pass, then 2 retry passes over both failures.
    assert attempts["n"] == 6, attempts["n"]


def test_evaluate_multiple_entries():
    entries = [_make_entry(f"id-{i}", correct_response=1) for i in range(3)]
    score_fn = lambda e, k: 0.8 if k == "response_1" else 0.3
    results = evaluate(entries, score_fn, num_runs=1)
    assert len(results) == 3
    assert all(r["runs"][0]["correct"] for r in results)


def test_evaluate_tie_higher_score_is_none():
    score_fn = lambda e, k: 0.5
    results = evaluate([_make_entry(correct_response=1)], score_fn, num_runs=1)
    assert len(results) == 1
    assert results[0]["runs"][0]["higher_score"] is None


def test_evaluate_tie_correct_is_false():
    score_fn = lambda e, k: 0.5
    results = evaluate([_make_entry(correct_response=1)], score_fn, num_runs=1)
    assert results[0]["runs"][0]["correct"] is False


def test_evaluate_tie_not_dropped():
    score_fn = lambda e, k: 0.5
    results = evaluate([_make_entry()], score_fn, num_runs=1)
    assert len(results) == 1


def test_evaluate_multi_run_stores_all_runs():
    call_counts = {}
    def score_fn(e, k):
        call_counts[k] = call_counts.get(k, 0) + 1
        return 0.7 if k == "response_1" else 0.3
    results = evaluate([_make_entry()], score_fn, num_runs=3)
    assert len(results[0]["runs"]) == 3
    assert call_counts["response_1"] == 3
    assert call_counts["response_2"] == 3


# --- compute_metrics ---

def test_compute_metrics_all_correct():
    results = [
        _make_result("clarity", correct_response=1, runs=[_make_run(0, 0.8, 0.3, 1)]),
        _make_result("clarity", correct_response=2, runs=[_make_run(0, 0.3, 0.8, 2)]),
    ]
    m = compute_metrics(results)
    assert m["overall"]["mean_accuracy"] == pytest.approx(1.0)
    assert m["by_criterion"]["clarity"]["mean_accuracy"] == pytest.approx(1.0)


def test_compute_metrics_all_wrong():
    results = [
        _make_result("clarity", correct_response=1, runs=[_make_run(0, 0.3, 0.8, 1)]),
        _make_result("clarity", correct_response=2, runs=[_make_run(0, 0.8, 0.3, 2)]),
    ]
    assert compute_metrics(results)["overall"]["mean_accuracy"] == pytest.approx(0.0)


def test_compute_metrics_per_criterion():
    results = [
        _make_result("clarity",    correct_response=1, runs=[_make_run(0, 0.8, 0.3, 1)]),
        _make_result("factuality", correct_response=1, runs=[_make_run(0, 0.3, 0.8, 1)]),
    ]
    m = compute_metrics(results)
    assert m["by_criterion"]["clarity"]["mean_accuracy"] == pytest.approx(1.0)
    assert m["by_criterion"]["factuality"]["mean_accuracy"] == pytest.approx(0.0)
    assert m["overall"]["n"] == 2


def test_compute_metrics_tie_counts_as_wrong():
    results = [
        _make_result("clarity", correct_response=1, runs=[_make_run(0, 0.5, 0.5, 1)]),
        _make_result("clarity", correct_response=2, runs=[_make_run(0, 0.3, 0.8, 2)]),
    ]
    m = compute_metrics(results)
    assert m["overall"]["mean_accuracy"] == pytest.approx(0.5)
    assert m["overall"]["n"] == 2


def test_compute_metrics_std_across_runs():
    runs = [_make_run(0, 0.9, 0.3, 1), _make_run(1, 0.5, 0.3, 1), _make_run(2, 0.8, 0.3, 1)]
    results = [_make_result("clarity", correct_response=1, runs=runs)]
    m = compute_metrics(results)
    assert m["overall"]["std_accuracy"] == pytest.approx(0.0)  # all runs correct


def test_compute_metrics_mean_score():
    runs = [_make_run(0, 0.8, 0.4, 1)]
    results = [_make_result("clarity", correct_response=1, runs=runs)]
    m = compute_metrics(results)
    assert m["overall"]["avg_raw_score"] == pytest.approx(0.6)  # (0.8 + 0.4) / 2


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m["overall"]["n"] == 0
    assert m["by_criterion"] == {}

# --- _extract_score ---
#
# gemini-2.5-flash ignored the JSON instruction on roughly a third of calls and wrote prose
# ending "Score: 1.0". Because an entry needs all 2*num_runs calls to succeed, that 33%
# per-call rate left 3 of 36 entries: a per-call annoyance amplified into a 92% data loss.
# Every string here is either a real observed reply or a near-miss built to break the regex.

from src.metaeval.scoring import _extract_score  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ('{"reasoning": "x", "score": 0.4}', 0.4),
    ('```json\n{"score": 0.75}\n```', 0.75),
    ('{"score": "0.5"}', 0.5),
    ("The agent guides the customer well.\n\nScore: 1.0", 1.0),     # observed
    ("...devoid of empathy.\n\nscore: 0.2", 0.2),                   # observed
    ("**Score:** 0.85", 0.85),
    ("score = 0.6", 0.6),
    ("Score:0.45", 0.45),
    ("score: 0", 0.0),
    ("score: 1", 1.0),
])
def test_extract_score_recovers(raw, expected):
    assert _extract_score(raw) == pytest.approx(expected)


# Observed on `llama3.1-8b`, which produced 8 of the 13 unparsed replies in a 2,100-call run --
# every one of them stating a usable score in a form the first two patterns missed. A per-call
# annoyance again, and again amplified by the all-calls-must-succeed guard.
@pytest.mark.parametrize("raw,expected", [
    ("The agent's score on task_resolution is 0.8.\n\nThe agent identified the issue.", 0.8),
    ("The agent's friendliness score is 0.8.", 0.8),
    ("The agent's conduct on the criterion task_resolution is 0.0. Not resolved.", 0.0),
    ("Score is 1.0", 1.0),
    ("the criterion communication_clarity is 0.35", 0.35),
])
def test_extract_score_recovers_the_is_phrasing(raw, expected):
    assert _extract_score(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [
    # The guarantee the anchor exists for: a number in prose is not a verdict.
    "The agent made 1 tool call and 2 KB searches.",
    "The agent is helpful. It is clear. There is nothing wrong.",
    "In turn 1 the agent is polite.",
    # `is` must stay near its anchor, or an unrelated later clause starts matching.
    "The score, which the rubric defined at length in a much earlier section, is 3",
])
def test_extract_score_is_phrasing_invents_nothing(raw):
    assert _extract_score(raw) is None


def test_the_new_patterns_cannot_change_an_already_parsed_reply():
    """Ordered least to most permissive, and the first in-range hit wins -- so a reply the old
    two patterns handled must come out identically. That is what makes this safe to add
    mid-measurement, with 1,927 calls already scored and checkpointed."""
    assert _extract_score('{"reasoning": "the score is 0.9", "score": 0.1}') == \
        pytest.approx(0.1)
    assert _extract_score("The score is 0.9 at first glance. Final score: 0.3") == \
        pytest.approx(0.3)


@pytest.mark.parametrize("raw", [
    "In turn 5, they acknowledge the request. In turn 7, the agent explains.",
    "The agent scored well overall on this conversation.",
    "",
    None,
    "I cannot evaluate this conversation.",
])
def test_extract_score_invents_nothing(raw):
    """A number in the prose is not a score, and no score means no score."""
    assert _extract_score(raw) is None


@pytest.mark.parametrize("raw", ['{"score": 7}', '{"score": -1}', "score: 42",
                                 '{"score": 100}'])
def test_extract_score_rejects_out_of_range(raw):
    """Out of range is rejected, never clamped.

    A reply scoring 7 on a 0-1 scale is not answering the question asked; reading it as 1.0
    would invent an opinion the model never expressed.
    """
    assert _extract_score(raw) is None


def test_extract_score_last_verdict_wins():
    """Reasoning may name a candidate score before settling on one."""
    assert _extract_score("The score might be 0.9 at first. Final score: 0.3") == \
        pytest.approx(0.3)


def test_extract_score_json_beats_trailing_prose():
    assert _extract_score('{"reasoning": "not a score: 0.9", "score": 0.1}') == \
        pytest.approx(0.1)


@patch("src.metaeval.scoring.litellm.completion")
def test_score_model_accepts_prose_score(mock_completion):
    """End to end: the prose shape that cost 33 of 36 entries now scores."""
    msg = MagicMock()
    msg.content = "The agent was warm throughout.\n\nScore: 0.8"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    mock_completion.return_value = resp
    assert _score_model(_make_entry(), "response_1", "gemini/gemini-2.5-flash") == \
        pytest.approx(0.8)


@patch("src.metaeval.scoring.litellm.completion")
def test_score_model_still_raises_without_a_score(mock_completion):
    msg = MagicMock()
    msg.content = "I am unable to assess this conversation."
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    mock_completion.return_value = resp
    with pytest.raises(ValueError, match="No score"):
        _score_model(_make_entry(), "response_1", "gemini/gemini-2.5-flash")


# --- usage accounting ---
#
# The scoring path used to read `resp.choices[0].message.content` and drop `resp.usage`, so
# no token count reached any saved result. These pin the wiring rather than the arithmetic
# (that is `tests/test_usage.py`).

from src.metaeval.usage import UsageMeter  # noqa: E402


def _usage_resp(score=0.5, prompt=400, completion=120, reasoning=80):
    from types import SimpleNamespace
    msg = MagicMock()
    msg.content = json.dumps({"score": score, "reasoning": "ok"})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning))
    return resp


@patch("src.metaeval.scoring.litellm.completion")
def test_score_model_records_usage(mock_completion):
    mock_completion.return_value = _usage_resp()
    meter = UsageMeter().start()
    _score_model(_make_entry(), "response_1", "fw/x", meter=meter, label="gpt-oss-20b")
    s = meter.summary()
    assert s["n_calls"] == 1
    assert s["prompt_tokens"] == 400 and s["completion_tokens"] == 120
    assert s["reasoning_tokens"] == 80
    assert meter.records[0].role == "evaluator"
    assert meter.records[0].label == "gpt-oss-20b"
    assert meter.records[0].item_id == "id-1"


@patch("src.metaeval.scoring.litellm.completion")
def test_score_model_records_a_failed_call(mock_completion):
    """The nemotron case: 20 rate-limit errors that cost real time and left no trace."""
    mock_completion.side_effect = RuntimeError("rate limit exceeded")
    meter = UsageMeter().start()
    with pytest.raises(RuntimeError):
        _score_model(_make_entry(), "response_1", "fw/x", meter=meter)
    s = meter.summary()
    assert s["n_calls"] == 1 and s["n_error"] == 1


@patch("src.metaeval.scoring.litellm.completion")
def test_no_score_reply_is_recorded_as_a_successful_call(mock_completion):
    """It billed. Counting it as an error would hide the tokens of the most verbose
    failure mode in the roster."""
    msg = MagicMock()
    msg.content = "I cannot assess this."
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    from types import SimpleNamespace
    resp.usage = SimpleNamespace(prompt_tokens=500, completion_tokens=900,
                                 completion_tokens_details=None)
    mock_completion.return_value = resp
    meter = UsageMeter().start()
    with pytest.raises(ValueError, match="No score"):
        _score_model(_make_entry(), "response_1", "fw/x", meter=meter)
    s = meter.summary()
    assert s["n_ok"] == 1 and s["n_error"] == 0
    assert s["total_tokens"] == 1400, "a verbose non-answer still costs 1,400 tokens"


@patch("src.metaeval.scoring.litellm.completion")
def test_scoring_without_a_meter_is_unchanged(mock_completion):
    """`meter` is optional, so the `(entry, key) -> float` contract is untouched."""
    mock_completion.return_value = _usage_resp(score=0.9)
    assert _score_model(_make_entry(), "response_1", "fw/x") == pytest.approx(0.9)


@patch("src.metaeval.scoring.litellm.completion")
def test_evaluate_accumulates_usage_over_every_call(mock_completion):
    """One meter across a whole `evaluate()` sweep: 2 responses x k runs."""
    mock_completion.return_value = _usage_resp(prompt=100, completion=10, reasoning=0)
    meter = UsageMeter().start()
    evaluate([_make_entry()], lambda e, k: _score_model(e, k, "fw/x", meter=meter),
             num_runs=3)
    assert meter.summary()["n_calls"] == 6
    assert meter.summary()["total_tokens"] == 6 * 110


@patch("src.metaeval.scoring.litellm.completion")
def test_usage_counts_retry_passes_too(mock_completion):
    """A retried call is a second billable call, and must appear as one."""
    mock_completion.side_effect = [
        RuntimeError("rate limit"),                      # first attempt fails
        _usage_resp(prompt=100, completion=10, reasoning=0),
        _usage_resp(prompt=100, completion=10, reasoning=0),
    ]
    meter = UsageMeter().start()
    evaluate([_make_entry()], lambda e, k: _score_model(e, k, "fw/x", meter=meter),
             num_runs=1, retry_passes=2, retry_delay_s=0)
    s = meter.summary()
    assert s["n_calls"] == 3, "2 calls for the entry plus 1 retry"
    assert s["n_error"] == 1
