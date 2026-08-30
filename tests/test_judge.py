"""`judge_side` -- the one call the filter cascade makes.

**What is pinned here is the A/B randomization and the mapping back.** The judge is shown two
responses as A and B in an order derived from `(pair id, judge name)`, and its answer has to be
mapped back to slot 1 or slot 2. Both halves are easy to get wrong in a way no output would
reveal: a judge that always answers "A" would look accurate if the order were fixed, and an
inverted mapping would report every verdict backwards while still looking like a plausible
filter. So the seed, the order and the mapping are asserted directly.

The no-verdict retry is here too, because it is the one path that costs a second call. Answer
extraction from malformed replies lives in `test_judge_extraction.py`.
"""

import json
import random
from unittest.mock import MagicMock, patch

import pytest

from src.metaeval.judge import judge_side


# --- helpers ---

def _make_entry(entry_id="abc-123"):
    return {
        "id": entry_id,
        "criterion_name": "clarity",
        "criterion_description": "Is the response clear?",
        "prompt": "Tell me something.",
        "response_1": "Clear response.",
        "response_2": "Confusing response.",
    }


def _resp(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    r = MagicMock()
    r.choices = [choice]
    return r


def _verdict(answer: str, reasoning: str = "some reasoning"):
    return _resp(json.dumps({"answer": answer, "reasoning": reasoning}))


def _a_is_response_1(entry_id: str, judge: str) -> bool:
    """The same draw `judge_side` makes, so a test can know which slot A holds."""
    return random.Random(f"{entry_id}:{judge}").random() > 0.5


# --- the A/B mapping ---

@patch("src.metaeval.judge.litellm.completion")
def test_answer_a_maps_to_whichever_slot_was_shown_as_a(mock_completion):
    entry = _make_entry()
    mock_completion.return_value = _verdict("A")
    out = judge_side(entry, "openai/gpt-5.4", "judge1")
    expected = 1 if _a_is_response_1(entry["id"], "judge1") else 2
    assert out["side"] == expected
    assert out["answer"] == "A"
    assert out["a_was"] == expected


@patch("src.metaeval.judge.litellm.completion")
def test_answer_b_maps_to_the_other_slot(mock_completion):
    entry = _make_entry()
    mock_completion.return_value = _verdict("B")
    out = judge_side(entry, "openai/gpt-5.4", "judge1")
    expected = 2 if _a_is_response_1(entry["id"], "judge1") else 1
    assert out["side"] == expected


@patch("src.metaeval.judge.litellm.completion")
def test_the_side_is_returned_rather_than_agreement_with_a_label(mock_completion):
    """The entry carries no label at all, and the call still succeeds.

    That is the point of returning a side: the judge is never shown `correct_response`, and
    agreement is computed by the caller afterwards. A `judge_side` that needed the label could
    not be run on the blind dataset.
    """
    entry = _make_entry()
    assert "correct_response" not in entry
    mock_completion.return_value = _verdict("A")
    out = judge_side(entry, "openai/gpt-5.4", "judge1")
    assert out["side"] in (1, 2)


# --- the randomization ---

@patch("src.metaeval.judge.litellm.completion")
def test_the_ab_order_is_reproducible_for_one_pair_and_judge(mock_completion):
    """Same pair, same judge, same prompt -- so a re-run is comparable rather than a new draw."""
    entry = _make_entry()
    mock_completion.return_value = _verdict("A")
    judge_side(entry, "openai/gpt-5.4", "judge1")
    first = mock_completion.call_args[1]["messages"][0]["content"]
    judge_side(entry, "openai/gpt-5.4", "judge1")
    second = mock_completion.call_args[1]["messages"][0]["content"]
    assert first == second


def test_the_two_judges_of_the_cascade_draw_independently():
    """Both judges seeing the identical A/B order would make the cascade's second vote
    partly a re-run of the first's position bias rather than a second opinion."""
    assert (random.Random("fixed-id:judge1").random()
            != random.Random("fixed-id:judge2").random())


@patch("src.metaeval.judge.litellm.completion")
def test_both_responses_reach_the_prompt(mock_completion):
    entry = _make_entry()
    mock_completion.return_value = _verdict("A")
    judge_side(entry, "openai/gpt-5.4", "judge1")
    sent = mock_completion.call_args[1]["messages"][0]["content"]
    assert entry["response_1"] in sent
    assert entry["response_2"] in sent
    assert entry["criterion_description"] in sent


# --- the no-verdict retry ---

@patch("src.metaeval.judge.litellm.completion")
def test_a_reply_with_no_verdict_is_asked_again_and_the_second_answer_counts(mock_completion):
    """Not covered by litellm's `num_retries`: this reply arrived as a healthy HTTP 200."""
    entry = _make_entry()
    mock_completion.side_effect = [_resp("I cannot decide."), _verdict("A")]
    out = judge_side(entry, "openai/gpt-5.4", "judge1")
    assert out["attempts"] == 2, "the extra call is recorded, so a prompted vote is visible"
    assert out["side"] == (1 if _a_is_response_1(entry["id"], "judge1") else 2)


@patch("src.metaeval.judge.litellm.completion")
def test_the_retry_shows_the_model_its_own_reply(mock_completion):
    """At temperature 0 a blind re-ask returns the same text, so the retry has to add
    something. It adds the assistant's own non-answer plus a request to commit -- and says
    nothing about which side to pick, so it cannot bias the answer, only its existence."""
    entry = _make_entry()
    mock_completion.side_effect = [_resp("Hmm."), _verdict("B")]
    judge_side(entry, "openai/gpt-5.4", "judge1")
    messages = mock_completion.call_args[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "Hmm."
    assert "A" in messages[2]["content"] and "B" in messages[2]["content"]


@patch("src.metaeval.judge.litellm.completion")
def test_an_exhausted_retry_raises_rather_than_guessing(mock_completion):
    """A judge that will not commit is signal, not noise: the pair that first provoked this
    was one where both judges chose the side the generated label calls wrong. Guessing a side
    here would convert that signal into a vote."""
    entry = _make_entry()
    mock_completion.return_value = _resp("still no answer")
    with pytest.raises(ValueError, match="no answer"):
        judge_side(entry, "openai/gpt-5.4", "judge1", max_retries=2)
    assert mock_completion.call_count == 3, "one first attempt plus max_retries"
