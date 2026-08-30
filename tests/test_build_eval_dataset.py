"""Selecting the evaluation set: the winning draw of every kept cell, and nothing else.

The two failures this guards against both cost real money at 1,000 pairs, and neither shows
up as an error -- only as a plausible-looking total:

- scoring **every draw** rather than the winning one, which is 1,673 pairs where 1,000 were
  kept, a 67% overspend on pairs the filter rejected;
- taking `records[0]` from a pair file instead of matching `pair_id`, which silently scores a
  *rejected* draw and reports it as benchmark accuracy.

The third thing guarded here costs nothing and invalidates everything: **the blind/answers
split.** `levels` is slot-ordered, so `["bad", "good"]` says response_1 is the worse side in
plain text. An evaluator prompt built from an unsplit record would be handing over the answer,
and every score would look excellent.
"""

import json

import pytest

from tools.build_eval_dataset import (
    ANSWER,
    BLIND,
    KEEP_CONTEXT,
    _pair_file,
    agent_of,
    answer,
    blind,
    build,
)


def _pair(pair_id, marker="winner"):
    return {
        "id": pair_id,
        "criterion_name": "clarity",
        "criterion_description": "Is it clear?",
        "positive_hint": "be clear",
        "negative_hint": "be vague",
        "prompt": f"policy and tools ({marker})",
        # Deliberately not "good"/"bad": the blinding test scans the serialised record for
        # those two words, and a fixture that plants them in free text would mask a real leak.
        "response_1": f"polished {marker}",
        "response_2": f"curt {marker}",
        "correct_response": 1,
        "schema_version": "3",
        "group_id": "g1",
        "levels": ["bad", "good"],
        "context": {"domain": "airline", "task_id": "0", "task_purpose": "p",
                    "user_persona": "u", "policy": "x" * 8000,
                    "agent_tools": "y" * 11000, "knowledge_base": {}},
        "trajectory_1": {"turns": ["z" * 40000],
                         "provenance": {"agent_model": "fw/models/gpt-oss-120b"}},
        "trajectory_2": {"turns": ["z" * 30000],
                         "provenance": {"agent_model": "fw/models/gpt-oss-120b"}},
    }


@pytest.fixture
def run(tmp_path):
    """A run directory with one kept cell on attempt 1, one kept on attempt 3, one rejected."""
    (tmp_path / "pairs").mkdir()
    index = {
        "airline.clarity.t0": {"state": "kept", "pair_id": "p-a", "attempt": 1},
        "airline.clarity.t1": {"state": "kept", "pair_id": "p-c", "attempt": 3},
        "airline.clarity.t2": {"state": "rejected", "pair_id": "p-r", "attempt": 1},
    }
    (tmp_path / "pair_index.json").write_text(json.dumps(index))
    p = tmp_path / "pairs"
    (p / "airline.clarity.t0.pairs.json").write_text(json.dumps([_pair("p-a")]))
    # The rejected earlier draws live beside the winner and must not be picked up.
    (p / "airline.clarity.t1.pairs.json").write_text(json.dumps([_pair("p-b", "loser1")]))
    (p / "airline.clarity.t1.a2.pairs.json").write_text(json.dumps([_pair("p-x", "loser2")]))
    (p / "airline.clarity.t1.a3.pairs.json").write_text(json.dumps([_pair("p-c")]))
    (p / "airline.clarity.t2.pairs.json").write_text(json.dumps([_pair("p-r", "rejected")]))
    return tmp_path


# --- selection -------------------------------------------------------------------------

def test_kept_pool_takes_the_winning_draw_of_every_kept_cell(run):
    entries, _answers, problems = build(run)
    assert problems == []
    assert [e["id"] for e in entries] == ["p-a", "p-c"]
    assert "rejected" not in json.dumps(entries), "a rejected cell must not be scored"
    assert "loser" not in json.dumps(entries), "an earlier draw of a kept cell must not be"


def test_all_pool_adds_the_rejected_cells(run):
    entries, _answers, problems = build(run, pool="all")
    assert problems == []
    assert {e["id"] for e in entries} == {"p-a", "p-c", "p-r"}


def test_the_winner_is_matched_by_id_not_by_position(tmp_path):
    """A pair file can hold several records; the index names which one won."""
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pair_index.json").write_text(json.dumps(
        {"d.c.t0": {"state": "kept", "pair_id": "second", "attempt": 1}}))
    (tmp_path / "pairs" / "d.c.t0.pairs.json").write_text(json.dumps(
        [_pair("first", "loser"), _pair("second", "winner")]))
    entries, _answers, problems = build(tmp_path)
    assert problems == []
    assert [e["id"] for e in entries] == ["second"]
    assert "loser" not in json.dumps(entries)


def test_output_order_is_stable(run):
    """`run_evaluators --limit N` takes a prefix, and a prefix of an unstable order is not a
    reproducible smoke test."""
    assert [e["id"] for e in build(run)[0]] == [e["id"] for e in build(run)[0]]


@pytest.mark.parametrize("attempt,name", [
    (1, "d.c.t0.pairs.json"), (2, "d.c.t0.a2.pairs.json"), (3, "d.c.t0.a3.pairs.json")])
def test_pair_file_naming_matches_the_pipeline(tmp_path, attempt, name):
    assert _pair_file(tmp_path, "d.c.t0", attempt).name == name


# --- the blind/answers split -----------------------------------------------------------

def test_every_field_the_evaluator_reads_survives_the_blinding():
    out = blind(_pair("p"))
    for k in ("id", "criterion_name", "criterion_description", "prompt",
              "response_1", "response_2"):
        assert out[k] == _pair("p")[k], k


def test_the_blind_record_cannot_reveal_which_side_is_better():
    """The one assertion that protects every number in the paper.

    Three fields give the answer away, each in a different way: `correct_response` states it,
    `levels` is slot-ordered so `["bad", "good"]` states it in words, and the two hints are the
    steering text with its own direction written into it. Any of the three in a prompt turns
    the whole sweep into a reading-comprehension test.
    """
    out = blind(_pair("p"))
    for k in ("correct_response", "levels", "positive_hint", "negative_hint"):
        assert k not in out, f"{k} reveals the label"
    text = json.dumps(out)
    for word in ("bad", "good", "correct_response"):
        assert word not in text, f"the word {word!r} survives into the blind record"


def test_the_two_halves_partition_the_label_bearing_fields():
    """Neither half may quietly grow into the other's territory: a field added to `BLIND`
    that is also in `ANSWER` would be a label shipped inside the prompt."""
    assert set(BLIND) & set(ANSWER) == {"id"}, "only the join key may appear in both"


def test_the_answers_carry_the_label_and_the_provenance():
    out = answer(_pair("p"))
    assert out["id"] == "p"
    assert out["correct_response"] == 1
    assert out["levels"] == ["bad", "good"]
    assert out["group_id"] == "g1"
    assert out["agent"] == "gpt-oss-120b", "lifted before the trajectory is dropped"


def test_the_agent_is_recoverable_only_because_it_is_lifted_early():
    """`agent_of` reads the trajectory, and `blind` drops the trajectory. Self-preference --
    three roster evaluators also authored these conversations -- is answerable only if the
    name is copied across before it is thrown away."""
    p = _pair("p")
    assert agent_of(p) == "gpt-oss-120b"
    assert "trajectory_1" not in blind(p)
    assert "provenance" not in json.dumps(blind(p))


def test_a_pair_with_no_provenance_reports_an_unknown_agent_rather_than_guessing():
    p = _pair("p")
    del p["trajectory_1"]["provenance"]
    del p["trajectory_2"]["provenance"]
    assert agent_of(p) == "", "an empty string a report can print as unknown"


def test_trajectories_and_the_bulk_of_context_are_dropped():
    """The evaluator never sees a trajectory -- `prompt` carries the rendered policy and tool
    list. `context` is 20KB of which 19KB is that same policy and tool list again."""
    out = blind(_pair("p"))
    assert "trajectory_1" not in out and "trajectory_2" not in out
    assert set(out["context"]) <= set(KEEP_CONTEXT)
    assert "policy" not in out["context"] and "agent_tools" not in out["context"]
    assert len(json.dumps(out)) < len(json.dumps(_pair("p"))) / 10


def test_the_fields_a_per_domain_breakdown_partitions_on_survive():
    """A per-domain or per-gap table is computed after the sweep, from the two files joined.
    `context.domain` has to stay on the blind side (it is not a label) and `levels` on the
    answers side (it is) -- dropping either turns a table into a KeyError once the money is
    already spent."""
    assert blind(_pair("p"))["context"]["domain"] == "airline"
    assert answer(_pair("p"))["levels"] == ["bad", "good"]


def test_a_pair_without_context_does_not_crash():
    p = _pair("p")
    del p["context"]
    assert blind(p)["context"] == {}


# --- problems are reported, not swallowed ----------------------------------------------

def test_a_missing_pair_file_is_reported_and_the_rest_still_build(run):
    (run / "pairs" / "airline.clarity.t0.pairs.json").unlink()
    entries, _answers, problems = build(run)
    assert [e["id"] for e in entries] == ["p-c"]
    assert len(problems) == 1 and "missing" in problems[0]


def test_an_unreadable_pair_file_is_reported(run):
    (run / "pairs" / "airline.clarity.t0.pairs.json").write_text("{not json")
    entries, _answers, problems = build(run)
    assert [e["id"] for e in entries] == ["p-c"]
    assert "unreadable" in problems[0]


def test_a_pair_id_absent_from_its_file_is_reported(run):
    (run / "pairs" / "airline.clarity.t0.pairs.json").write_text(
        json.dumps([_pair("someone-else")]))
    entries, _answers, problems = build(run)
    assert [e["id"] for e in entries] == ["p-c"]
    assert "not in" in problems[0]


def test_a_kept_cell_with_no_pair_id_is_reported(tmp_path):
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pair_index.json").write_text(json.dumps(
        {"d.c.t0": {"state": "kept", "attempt": 1}}))
    entries, _answers, problems = build(tmp_path)
    assert entries == []
    assert "no pair_id" in problems[0]


# --- the handshake with the sweep ------------------------------------------------------

def test_the_output_is_what_run_evaluators_loads(run, tmp_path):
    """Both files together, through the loader the sweep actually uses.

    Written as one test on purpose: the blind file alone scores nothing (no `correct_response`
    to be right about) and the answers file alone has nothing to score, so the handshake is
    the unit -- and a filename convention is half of it.
    """
    from src.metaeval.scoring import evaluate_models, load_pairs
    entries, answers, _ = build(run)
    (tmp_path / "eval_dataset.json").write_text(json.dumps(entries))
    (tmp_path / "eval_answers.json").write_text(json.dumps(answers))

    data = load_pairs(str(tmp_path / "eval_dataset.json"))
    scored = evaluate_models(data, [("m", "fw/x")],
                             lambda l, e, k: 0.9 if k == "response_1" else 0.1,
                             num_runs=1, progress=False)
    assert len(scored["m"]) == 2
    assert all(r["runs"][0]["correct"] for r in scored["m"]), (
        "response_1 is the better side in this fixture, and the join has to say so")


def test_scoring_the_blind_file_without_its_answers_fails_loudly(run, tmp_path):
    """The failure mode this prevents is not a crash but a *number*: with no labels joined,
    a sweep would run 50,000 paid calls and then report accuracy against nothing."""
    entries, _answers, _ = build(run)
    (tmp_path / "eval_dataset.json").write_text(json.dumps(entries))
    from src.metaeval.scoring import load_pairs
    with pytest.raises(SystemExit, match="no answers file"):
        load_pairs(str(tmp_path / "eval_dataset.json"))
