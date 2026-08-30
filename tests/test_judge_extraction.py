"""Tests for `_extract_answer` in `src/metaeval/judge.py`.

Every string here is either taken from a real failing judge response in the first 36-call
run or is a near-miss designed to break the salvage patterns. The point of the function is
to recover verdicts from malformed wrappers **without ever inventing one**, so the negative
cases matter more than the positive ones.
"""

from src.metaeval.judge import _extract_answer


# --- shapes seen in real output ---

def test_clean_json():
    assert _extract_answer('{"reasoning": "x", "answer": "A"}') == "A"


def test_json_with_literal_newlines_inside_a_string():
    # Illegal JSON -- a real newline inside a string value -- but the verdict is intact.
    # This was 2 of 5 failures in the first run.
    raw = '{\n  "reasoning": "Response B does this.\nResponse A does not.",\n  "answer": "B"\n}'
    assert _extract_answer(raw) == "B"


def test_markdown_answer_heading():
    # The model ignored the JSON instruction entirely. 2 of 5 failures.
    assert _extract_answer("**Reasoning:**\nA is clearer.\n\n**Answer:** A") == "A"


def test_code_fenced_json():
    assert _extract_answer('```json\n{"reasoning": "x", "answer": "B"}\n```') == "B"


def test_truncated_json_with_answer_present():
    assert _extract_answer('{"reasoning": "cut off mid-sen') is None


def test_lowercase_answer_key_and_value():
    assert _extract_answer('{"answer": "b"}') == "B"


def test_answer_without_quotes():
    assert _extract_answer("answer: A") == "A"


def test_answer_with_equals():
    assert _extract_answer("Answer = B") == "B"


# --- must never invent a verdict ---

def test_ramble_with_no_verdict_is_none():
    # The 41,006-character failure: ran to the token cap without ever answering.
    # Recovering something here would be fabricating data.
    assert _extract_answer("I need to weigh several things. " * 500) is None


def test_prose_naming_a_side_is_not_a_verdict():
    # "Response A fails the criterion" must not be read as choosing A.
    assert _extract_answer("Response A fails the criterion. Response B is better.") is None


def test_bare_claim_without_answer_marker_is_none():
    assert _extract_answer("Response A is clearly better throughout.") is None


def test_empty_and_none_are_safe():
    assert _extract_answer("") is None
    assert _extract_answer(None) is None


def test_reasoning_mentioning_answer_loses_to_the_real_verdict():
    # The last match wins: reasoning may use the word "answer" on the way to the verdict.
    raw = 'The answer might be A at first glance.\n\n**Answer:** B'
    assert _extract_answer(raw) == "B"


def test_json_answer_beats_stray_text():
    assert _extract_answer('{"reasoning":"answer A looks tempting","answer":"B"}') == "B"


def test_neither_a_nor_b_is_none():
    assert _extract_answer('{"answer": "C"}') is None
    assert _extract_answer('{"answer": "tie"}') is None


def test_multi_letter_token_is_not_matched():
    # "ANSWER: ABSTAIN" must not be read as "A".
    assert _extract_answer("ANSWER: ABSTAIN") is None


# --- usage accounting on the judge path ---
#
# `judge_side` dropped `resp.usage` too, and its retry loop makes the omission worse: the
# correction turn re-sends the prompt *plus* the assistant's reply, so it is the single
# most expensive call in a run and used to be entirely unaccounted for.

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from src.metaeval.judge import judge_side  # noqa: E402
from src.metaeval.usage import UsageMeter  # noqa: E402


def _entry(entry_id="pair-1"):
    return {"id": entry_id, "criterion_name": "clarity",
            "criterion_description": "Is it clear?",
            "prompt": "ctx", "response_1": "r1", "response_2": "r2",
            "correct_response": 1}


def _judge_resp(content, prompt=800, completion=60, reasoning=0):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning))
    return resp


@patch("src.metaeval.judge.litellm.completion")
def test_judge_side_records_usage(mock_completion):
    mock_completion.return_value = _judge_resp(
        '{"reasoning": "clearer", "answer": "A"}', prompt=900, completion=40, reasoning=25)
    meter = UsageMeter().start()
    judge_side(_entry(), "fw/x", "judge1", meter=meter)
    s = meter.summary()
    assert s["n_calls"] == 1
    assert s["prompt_tokens"] == 900 and s["reasoning_tokens"] == 25
    assert meter.records[0].role == "judge"
    assert meter.records[0].label == "judge1"
    assert meter.records[0].item_id == "pair-1"
    assert meter.records[0].attempt == 1


@patch("src.metaeval.judge.litellm.completion")
def test_retry_turn_is_recorded_with_its_larger_prompt(mock_completion):
    """The correction turn carries the original prompt plus the failed reply, so its
    prompt tokens exceed attempt 1's. Recording only the last attempt would have
    *under*-counted the run while reporting the bigger number."""
    mock_completion.side_effect = [
        _judge_resp("I would rather not choose.", prompt=800, completion=300),
        _judge_resp('{"answer": "B"}', prompt=1150, completion=15),
    ]
    meter = UsageMeter().start()
    out = judge_side(_entry(), "fw/x", "judge1", meter=meter)
    s = meter.summary()
    assert out["attempts"] == 2
    assert s["n_calls"] == 2, "both turns are billable calls"
    assert s["prompt_tokens"] == 1950, "attempt 2's prompt adds to, not replaces, attempt 1"
    by = meter.by("attempt")
    assert by["2"]["prompt_tokens"] > by["1"]["prompt_tokens"]


@patch("src.metaeval.judge.litellm.completion")
def test_exhausted_retries_still_leave_a_usage_record(mock_completion):
    """`judge_side` raises when no verdict ever arrives. The calls still happened."""
    mock_completion.return_value = _judge_resp("No comment.", prompt=800, completion=200)
    meter = UsageMeter().start()
    with pytest.raises(ValueError, match="no answer"):
        judge_side(_entry(), "fw/x", "judge1", meter=meter, max_retries=2)
    s = meter.summary()
    assert s["n_calls"] == 3, "1 initial attempt + 2 retries, all billed"
    assert s["n_ok"] == 3, "the provider answered every time; it just never committed"
    assert s["total_tokens"] == 3000


@patch("src.metaeval.judge.litellm.completion")
def test_judge_transport_failure_is_recorded_as_an_error(mock_completion):
    mock_completion.side_effect = RuntimeError("rate limit exceeded")
    meter = UsageMeter().start()
    with pytest.raises(RuntimeError):
        judge_side(_entry(), "fw/x", "judge1", meter=meter)
    assert meter.summary()["n_error"] == 1


@patch("src.metaeval.judge.litellm.completion")
def test_judge_side_without_a_meter_is_unchanged(mock_completion):
    mock_completion.return_value = _judge_resp('{"answer": "A"}')
    out = judge_side(_entry(), "fw/x", "judge1")
    assert out["side"] in (1, 2) and out["answer"] == "A"


@patch("src.metaeval.judge.litellm.completion")
def test_two_judges_share_one_meter_and_stay_separable(mock_completion):
    """The pipeline uses a single meter across the whole cascade, so per-judge spend has to
    be recoverable from it -- the two judges differ by ~10x in output tokens."""
    mock_completion.return_value = _judge_resp('{"answer": "A"}', prompt=800, completion=50)
    meter = UsageMeter().start()
    judge_side(_entry("p1"), "fw/a", "judge1", meter=meter)
    judge_side(_entry("p1"), "fw/b", "judge2", meter=meter)
    by = meter.by("label")
    assert set(by) == {"judge1", "judge2"}
    assert meter.summary()["total_tokens"] == 1700
