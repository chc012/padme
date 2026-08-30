"""Fidelity tests for the canonical schema.

The central one is `test_roundtrip_is_exact`: convert a real tau3 simulation and
reconstruct the source messages from the converted record. It runs over every
transcript in `data/calibration/`, so it exercises all 4 domains including retrieval,
user-side tool calls, and a run that terminated by transfer.

Why it can fail rather than being a tautology: modelled fields are read out of
the source and written back from the model, so mis-modelling one (reading
`tc["name"]` and writing `tc["function"]["name"]`, say) breaks equality. Fields
the schema does not model are preserved in `extra`, so a tau3 upgrade that adds
a field passes -- which is the intent. What must not pass silently is losing a
field we claim to carry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.metaeval.schema import (
    LEVEL_ORDER,
    JudgeVote,
    PairwiseEntry,
    Provenance,
    SteeringRecord,
    TaskContext,
    Trajectory,
    Turn,
    pairs_from_group,
    render_context,
    render_task_metadata,
    render_trajectory,
)
from src.metaeval.schema.convert import (
    compute_stats,
    normalise_source_message,
    parse_retrieved_docs,
    roundtrip_is_exact,
    trajectory_from_simulation,
)

# The four raw tau3 simulation transcripts from the domain calibration runs, one per
# domain. Absent from a fresh clone; see the fixture note below.
CALIBRATION = Path("data/calibration")

# `data/` is gitignored, so the calibration transcripts are absent from a fresh clone.
# Parametrising only over them would collect zero cases and report a pass -- the
# same silent-zero failure this schema goes to some trouble to avoid elsewhere.
# This fixture is committed, so the round-trip is always exercised; the calibration
# files add breadth when present.
FIXTURE = Path("tests/fixtures/tau2_simulation_sample.json")
FIXTURE_DOMAIN = "banking_knowledge"


def _transcripts() -> list[Path]:
    extra = (
        sorted(p for p in CALIBRATION.glob("*.json") if p.name != "summary.json")
        if CALIBRATION.is_dir()
        else []
    )
    return [FIXTURE, *extra]


def _domain(path: Path) -> str:
    return FIXTURE_DOMAIN if path == FIXTURE else path.name.split(".")[0]


# --------------------------------------------------------------------------- #
# The fidelity test
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", _transcripts(), ids=lambda p: p.stem)
def test_roundtrip_is_exact(path: Path):
    """Every source message is reconstructible from the converted trajectory."""
    data = json.loads(path.read_text())
    ctx = TaskContext(domain=_domain(path), task_id=str(data["task_id"]), policy="")
    traj = trajectory_from_simulation(data, ctx)

    # Uses `convert.roundtrip_is_exact` rather than restating the comparison, so the test
    # and any other consumer of a converted trajectory cannot drift apart on what "exact"
    # means.
    assert roundtrip_is_exact(traj, data["messages"])


@pytest.mark.parametrize("path", _transcripts(), ids=lambda p: p.stem)
def test_conversion_keeps_every_message(path: Path):
    data = json.loads(path.read_text())
    ctx = TaskContext(domain=_domain(path), task_id=str(data["task_id"]), policy="")
    traj = trajectory_from_simulation(data, ctx)
    assert len(traj.turns) == len(data["messages"])
    assert [t.index for t in traj.turns] == list(range(len(data["messages"])))


def test_dropped_field_fails_the_roundtrip():
    """The fidelity test is load-bearing: removing a modelled field breaks it.

    Guards against the test degenerating into a tautology if `extra` were ever
    widened to swallow modelled keys.
    """
    source = {"role": "assistant", "content": "hi", "turn_idx": 3,
              "tool_calls": None, "cost": 0.01}
    turn = Turn(index=0, role="assistant", content="hi", turn_idx=3,
                extra={"cost": 0.01})
    assert turn.to_tau2_dict() == source

    lossy = Turn(index=0, role="assistant", content="hi", turn_idx=3)  # extra dropped
    assert lossy.to_tau2_dict() != source


# --------------------------------------------------------------------------- #
# Retrieval parsing
# --------------------------------------------------------------------------- #

RETRIEVAL_RESULT = """\
1. Gold Rewards Card: Overview
   ID: doc_credit_cards_gold_rewards_card_001
   Score: 20.2385
   Content: ## Eligibility
- Minimum credit score: 735

2. Silver Rewards Card: Overview
   ID: doc_credit_cards_silver_rewards_card_001
   Score: 16.2514
   Content: ## Eligibility
- Minimum credit score: 680
"""


def test_parse_retrieved_docs():
    docs, failed = parse_retrieved_docs(RETRIEVAL_RESULT)
    assert not failed
    assert [d.doc_id for d in docs] == [
        "doc_credit_cards_gold_rewards_card_001",
        "doc_credit_cards_silver_rewards_card_001",
    ]
    assert docs[0].rank == 1
    assert docs[0].score == pytest.approx(20.2385)
    assert docs[0].title == "Gold Rewards Card: Overview"
    assert "Minimum credit score: 735" in docs[0].content


def test_numbered_lines_inside_a_document_are_not_documents():
    """Document bodies contain numbered lists. Those must not become documents."""
    text = """\
1. Card Overview
   ID: doc_a
   Score: 5.0
   Content: Steps to apply:
1. Fill the form
2. Wait for a decision
"""
    docs, failed = parse_retrieved_docs(text)
    assert not failed
    assert [d.doc_id for d in docs] == ["doc_a"]


def test_empty_retrieval_is_not_a_parse_failure():
    docs, failed = parse_retrieved_docs("No matching documents found.")
    assert docs == [] and not failed


def test_unparseable_retrieval_is_flagged_not_silently_empty():
    """A result carrying ids we could not read must not look like zero hits."""
    docs, failed = parse_retrieved_docs('{"hits": [{"ID: doc_a": 1}]}' + "x" * 300)
    assert docs == [] and failed


def test_real_retrieval_result_parses():
    path = CALIBRATION / "banking_knowledge.json"
    if not path.is_file():
        pytest.skip("banking transcript not present")
    data = json.loads(path.read_text())
    ctx = TaskContext(domain="banking_knowledge", task_id=str(data["task_id"]), policy="")
    traj = trajectory_from_simulation(data, ctx)
    results = [r for t in traj.turns for r in t.tool_results if r.retrieved is not None]
    assert results, "no retrieval result was recognised in the banking transcript"
    assert all(not r.retrieval_parse_failed for r in results)
    assert traj.stats.retrieved_doc_ids


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def test_spoken_turns_exclude_tool_traffic():
    """The distinction the calibration runs measured: raw messages run ~3x spoken turns."""
    turns = [
        Turn(index=0, role="assistant", content="Hello!"),
        Turn(index=1, role="user", content="I need help."),
        Turn(index=2, role="assistant", content=None,
             tool_calls=[{"id": "c1", "name": "lookup", "arguments": {"x": 1}}]),
        Turn(index=3, role="tool",
             tool_results=[{"id": "c1", "content": "{}", "requestor": "assistant"}]),
        Turn(index=4, role="assistant", content="Found it."),
    ]
    stats = compute_stats([Turn.model_validate(t) for t in turns])
    assert stats.n_messages == 5
    assert stats.spoken_turns == 3
    assert stats.spoken_words == 1 + 3 + 2
    assert stats.agent_tool_calls == ["lookup"]
    assert stats.user_tool_calls == []


def test_user_tool_calls_are_attributed_to_the_user():
    turns = [Turn(index=0, role="user", content="Applying now.",
                  tool_calls=[{"id": "c1", "name": "apply_for_credit_card",
                               "arguments": {}, "requestor": "user"}])]
    stats = compute_stats(turns)
    assert stats.user_tool_calls == ["apply_for_credit_card"]
    assert stats.agent_tool_calls == []


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #

def _traj(level: str, ctx: TaskContext, text: str) -> Trajectory:
    return Trajectory(
        trajectory_id=f"t-{level}",
        context=ctx,
        provenance=Provenance(
            domain=ctx.domain, task_id=ctx.task_id,
            steering=SteeringRecord(criterion_name="friendliness", level=level,
                                    instruction=f"be {level}"),
        ),
        turns=[Turn(index=0, role="assistant", content=text)],
    )


def _group() -> dict[str, Trajectory]:
    ctx = TaskContext(domain="airline", task_id="task_0", policy="Be helpful.")
    return {lvl: _traj(lvl, ctx, f"{lvl} answer") for lvl in LEVEL_ORDER}


def test_group_yields_three_pairs_with_both_gap_sizes():
    pairs = pairs_from_group(_group(), criterion_name="friendliness",
                             criterion_description="warmth and empathy")
    assert len(pairs) == 3
    assert sorted(p.gap for p in pairs) == ["adjacent", "adjacent", "wide"]
    assert len({p.group_id for p in pairs}) == 1


def test_correct_response_points_at_the_better_level():
    for p in pairs_from_group(_group(), criterion_name="friendliness",
                              criterion_description="warmth"):
        better_side = p.levels[p.correct_response - 1]
        worse_side = p.levels[2 - p.correct_response]
        assert LEVEL_ORDER[better_side] > LEVEL_ORDER[worse_side]


def test_better_side_is_not_always_response_1():
    """Position must not be a free signal for judges or humans."""
    slots = set()
    for i in range(40):
        ctx = TaskContext(domain="airline", task_id=f"task_{i}", policy="p")
        group = {lvl: _traj(lvl, ctx, lvl) for lvl in LEVEL_ORDER}
        slots.update(p.correct_response
                     for p in pairs_from_group(group, criterion_name="c",
                                               criterion_description="d"))
    assert slots == {1, 2}


def test_incomplete_group_is_rejected():
    group = _group()
    del group["good"]
    with pytest.raises(ValueError, match="incomplete"):
        pairs_from_group(group, criterion_name="c", criterion_description="d")


def test_a_group_that_was_not_held_fixed_is_refused():
    """The pair stores one `context` for both sides, so building it under a mismatch would
    record one side's policy as if both had run under it -- and nothing downstream reads
    the trajectories' own contexts to notice."""
    group = _group()
    other = TaskContext(domain="airline", task_id="task_0", policy="A DIFFERENT POLICY")
    group["good"] = _traj("good", other, "good answer")
    with pytest.raises(ValueError, match="not held fixed"):
        pairs_from_group(group, criterion_name="c", criterion_description="d")


def test_a_pair_read_back_from_disk_can_still_be_checked():
    """`context_consistent` survives the build-time refusal for a reason: a pair loaded
    from a file was not built by this process."""
    pair = pairs_from_group(_group(), criterion_name="c", criterion_description="d")[0]
    assert pair.context_consistent
    tampered = pair.model_copy(deep=True)
    tampered.trajectory_2.context = TaskContext(
        domain="airline", task_id="task_0", policy="A DIFFERENT POLICY")
    assert not tampered.context_consistent


def test_flat_view_is_populated_and_compatible():
    """The judge module and `build_eval_dataset.py` read exactly these keys."""
    pair = pairs_from_group(_group(), criterion_name="friendliness",
                            criterion_description="warmth")[0]
    for key in ("id", "criterion_name", "criterion_description", "positive_hint",
                "negative_hint", "prompt", "response_1", "response_2",
                "correct_response"):
        value = getattr(pair, key)
        assert value or value == 0, f"{key} is empty"
    assert pair.correct_response in (1, 2)


# --------------------------------------------------------------------------- #
# Votes are recorded, not applied
# --------------------------------------------------------------------------- #

def test_parse_failure_is_not_disagreement():
    """The distinction `judge.py`'s `_parse_json` cannot make today."""
    assert JudgeVote(judge_name="j", model="m", chose=1).agrees_with(1) is True
    assert JudgeVote(judge_name="j", model="m", chose=2).agrees_with(1) is False
    unparseable = JudgeVote(judge_name="j", model="m", parse_failed=True,
                            raw_output="I think response A is...")
    assert unparseable.agrees_with(1) is None


def test_entry_stores_votes_without_filtering():
    pair = pairs_from_group(_group(), criterion_name="c",
                            criterion_description="d")[0]
    pair.judge_votes = [
        JudgeVote(judge_name="judge1", model="llama", chose=1),
        JudgeVote(judge_name="judge2", model="gemma", parse_failed=True),
    ]
    assert len(pair.judge_votes) == 2
    # No field on the entry decides the label. That is deliberate.
    assert not hasattr(pair, "label")


# --------------------------------------------------------------------------- #
# Serialisation and rendering
# --------------------------------------------------------------------------- #

def test_entry_survives_a_json_round_trip():
    pair = pairs_from_group(_group(), criterion_name="c",
                            criterion_description="d")[0]
    pair.judge_votes = [JudgeVote(judge_name="j1", model="m", chose=2)]
    again = PairwiseEntry.model_validate(json.loads(pair.model_dump_json()))
    assert again.trajectory_1.to_tau2_messages() == pair.trajectory_1.to_tau2_messages()
    assert again.correct_response == pair.correct_response
    assert again.gap == pair.gap
    assert again.judge_votes == pair.judge_votes


def test_render_marks_truncation_rather_than_hiding_it():
    ctx = TaskContext(domain="airline", task_id="t", policy="p")
    traj = Trajectory(
        trajectory_id="t", context=ctx, provenance=Provenance(),
        turns=[
            Turn(index=0, role="assistant", content=None,
                 tool_calls=[{"id": "c1", "name": "lookup", "arguments": {}}]),
            Turn(index=1, role="tool",
                 tool_results=[{"id": "c1", "content": "y" * 900,
                                "requestor": "assistant"}]),
        ],
    )
    text = render_trajectory(traj, tool_result_chars=100)
    assert "truncated, 800 more characters" in text
    assert "lookup()" in text


def test_render_shows_available_tools_even_when_never_called():
    """Judging a call the agent should have made needs the tool to be visible,
    even though it appears nowhere in the transcript."""
    ctx = TaskContext(
        domain="airline", task_id="t", policy="Follow the rules.",
        agent_tools=[{"name": "cancel_reservation", "description": "Cancel a booking."}],
    )
    text = render_context(ctx)
    assert "cancel_reservation" in text
    assert "Follow the rules." in text


def test_render_flags_an_unparseable_retrieval_result():
    ctx = TaskContext(domain="banking_knowledge", task_id="t", policy="p")
    traj = Trajectory(
        trajectory_id="t", context=ctx, provenance=Provenance(),
        turns=[Turn(index=0, role="tool", tool_results=[{
            "id": "c1", "content": "garbage", "requestor": "assistant",
            "retrieval_parse_failed": True}])],
    )
    assert "could not be parsed" in render_trajectory(traj)


# --------------------------------------------------------------------------- #
# Two subtleties the real transcripts surfaced
# --------------------------------------------------------------------------- #

def test_protocol_markers_are_not_spoken_turns():
    """`###TRANSFER###` ends a conversation; it is not something a customer says.

    On airline's 3-turn trajectory, counting it inflated length by 33%.
    """
    turns = [
        Turn(index=0, role="assistant", content="Hi! How can I help you today?"),
        Turn(index=1, role="user", content="Cancel my booking please."),
        Turn(index=2, role="assistant", content="YOU ARE BEING TRANSFERRED."),
        Turn(index=3, role="user", content="###TRANSFER###"),
    ]
    stats = compute_stats(turns)
    assert stats.n_messages == 4
    assert stats.spoken_turns == 3
    assert not turns[3].is_spoken


def test_airline_length_matches_the_calibration_measurement():
    path = CALIBRATION / "airline.json"
    if not path.is_file():
        pytest.skip("airline transcript not present")
    data = json.loads(path.read_text())
    ctx = TaskContext(domain="airline", task_id=str(data["task_id"]), policy="")
    traj = trajectory_from_simulation(data, ctx)
    assert (traj.stats.spoken_turns, traj.stats.spoken_words) == (3, 94)


def test_unscored_basis_exposes_a_partial_reward():
    """Retail declares NL_ASSERTION but the patched substrate scores only DB."""
    from src.metaeval.schema import RewardRecord

    partial = RewardRecord(reward=1.0, reward_basis=["DB", "NL_ASSERTION"],
                           reward_breakdown={"DB": 1.0})
    assert partial.unscored_basis == ["NL_ASSERTION"]
    assert not partial.nl_assertion_present  # no LLM judge ran, which is correct

    full = RewardRecord(reward=1.0, reward_basis=["DB"], reward_breakdown={"DB": 1.0})
    assert full.unscored_basis == []


def test_render_omits_the_task_purpose_entirely():
    """Ground truth reaches no grader -- see test_no_ground_truth_reaches_a_grader."""
    real = TaskContext(domain="d", task_id="task_001", policy="p",
                       task_purpose="Customer wants to cancel a flight.")
    assert "Customer wants to cancel a flight." not in render_context(real)


def test_the_committed_fixture_exists():
    """Without it, the round-trip tests above collect nothing on a fresh clone."""
    assert FIXTURE.is_file(), f"missing {FIXTURE}"


def test_fixture_covers_a_multi_tool_message():
    """One assistant turn calling several tools -- absent from all 8 calibration runs,
    so only this fixture exercises the MultiToolMessage branch."""
    data = json.loads(FIXTURE.read_text())
    ctx = TaskContext(domain=FIXTURE_DOMAIN, task_id=str(data["task_id"]), policy="")
    traj = trajectory_from_simulation(data, ctx)

    multi = [t for t in traj.turns if t.multi]
    assert len(multi) == 1
    assert len(multi[0].tool_results) == 2
    # Retrieval is recognised inside a multi-tool message, and the non-retrieval
    # sibling is left alone.
    by_id = {r.id: r for r in multi[0].tool_results}
    assert by_id["call_multi_b"].retrieved is not None
    assert by_id["call_multi_a"].retrieved is None
    assert roundtrip_is_exact(traj, data["messages"])


def test_tool_descriptions_truncate_at_a_word_boundary():
    """Cutting mid-word read as a typo, not an elision."""
    ctx = TaskContext(domain="d", task_id="t", policy="p", agent_tools=[{
        "name": "log_verification",
        "description": "Log a verification record after successfully verifying a "
                       "user's identity. Call this tool after you have verified a "
                       "user by confirming 2 out of 4 identity fields (date of "
                       "birth, email, phone number, address) and then storing the "
                       "outcome in the audit log for compliance review.",
    }])
    line = next(ln for ln in render_context(ctx).splitlines()
                if ln.startswith("- log_verification"))
    assert line.endswith(" ...")
    assert not line.rstrip(" .").endswith(",")
    # The cut lands between words, never inside one.
    body = line.split(": ", 1)[1].removesuffix(" ...")
    assert ctx.agent_tools[0].description.startswith(body)
    assert ctx.agent_tools[0].description[len(body)] in " ,"


# --------------------------------------------------------------------------- #
# What the grader sees, and what it does not
# --------------------------------------------------------------------------- #

def _ctx_with_everything() -> TaskContext:
    return TaskContext(
        domain="banking_knowledge", task_id="task_001", policy="Be helpful.",
        task_purpose="Testing that agent refuses a cancellation that is not allowed.",
        user_persona="impatient",
        user_instructions="You are playing the role of a customer. ONLY MENTION the "
                          "free subscription if you are asked about it.",
        required_documents=["doc_gold_001", "doc_silver_001"],
        agent_tools=[{"name": "KB_search", "description": "Search the KB."}],
    )


def test_no_ground_truth_reaches_a_grader():
    """Reversed on 2026-08-11, for external validity.

    A judge deployed in the wild has no answer key, so giving ours one would
    measure a scoring task nobody runs. Both graders now work from the same
    material the agent had -- policy, tools, conversation.
    """
    text = render_context(_ctx_with_everything())
    assert "refuses a cancellation that is not allowed" not in text
    assert "doc_gold_001" not in text and "doc_silver_001" not in text
    assert "REFERENCE" not in text
    # The policy and the tools are still there: withholding those would make the
    # criteria unjudgeable rather than merely harder.
    assert "Be helpful." in text and "KB_search" in text


def test_grader_does_not_see_the_user_simulator_prompt():
    """It reports on the simulator's instruction-following, not the agent's work,
    and it includes instructions to withhold information."""
    text = render_context(_ctx_with_everything())
    assert "playing the role of a customer" not in text
    assert "ONLY MENTION" not in text
    assert "impatient" not in text


def test_withheld_fields_are_still_stored_and_inspectable():
    ctx = _ctx_with_everything()
    meta = render_task_metadata(ctx)
    assert "playing the role of a customer" in meta
    assert "impatient" in meta
    assert "NOT SHOWN TO EVALUATOR OR ANNOTATOR" in meta
    # And the structured record keeps everything regardless of rendering.
    assert ctx.user_instructions and ctx.user_persona


def test_ground_truth_stays_stored_and_inspectable():
    """Removed from the render, not from the record."""
    ctx = _ctx_with_everything()
    meta = render_task_metadata(ctx)
    assert "refuses a cancellation that is not allowed" in meta
    assert "doc_gold_001" in meta
    assert ctx.task_purpose and ctx.required_documents


def test_the_render_cannot_vary_by_criterion():
    """Both graders read one string, and it is the same string for every criterion.

    Withholding something for friendliness but not for tool use would hide exactly
    the cross-criterion distortions the benchmark exists to detect. The render takes
    no criterion argument, so this cannot drift.
    """
    import inspect

    params = set(inspect.signature(render_context).parameters)
    assert "criterion" not in params and "criterion_name" not in params

    ctx = _ctx_with_everything()
    assert render_context(ctx) == render_context(ctx)  # one view, no variants


def test_task_id_and_domain_are_not_rendered():
    """The telecom task id named the answer.

    `[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]` states the root
    cause the agent is supposed to diagnose, so it was ground truth hiding in a field
    nobody had labelled as such -- the same leak the reference block was removed for.
    Airline's `0` and banking's `task_001` carry nothing, and the domain is inferable
    from the policy, so both fields simply go.
    """
    ctx = TaskContext(
        domain="telecom",
        task_id="[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]",
        policy="Be helpful.",
        agent_tools=[{"name": "reboot_device", "description": "Reboot it."}],
    )
    text = render_context(ctx)
    assert "roaming_enabled" not in text
    assert "TASK ID" not in text and "DOMAIN" not in text
    assert "telecom" not in text
    # Still stored, and still inspectable by us.
    assert "roaming_enabled" in render_task_metadata(ctx)
    assert "telecom" in render_task_metadata(ctx)
    # What a grader does need is untouched.
    assert "reboot_device" in text and "Be helpful." in text


def test_render_does_not_open_with_blank_lines():
    """Each section prepends a separator; the first one must not survive."""
    text = render_context(TaskContext(domain="d", task_id="t", policy="p"))
    assert not text.startswith("\n")
    assert text.startswith("POLICY THE AGENT MUST FOLLOW")


def test_pairs_from_group_can_build_a_single_gap():
    """The one-gap-per-group design: only the assigned pair, from only two trajectories.

    The gap is assigned at generation time so a group simulates 2 levels rather than 3 --
    72 simulations for 36 human-annotated pairs instead of 108 for the same 36.
    """
    ctx = TaskContext(domain="airline", task_id="t", policy="p")

    def traj(level: str) -> Trajectory:
        return Trajectory(
            trajectory_id=f"tr-{level}", context=ctx,
            provenance=Provenance(steering={"criterion_name": "friendliness",
                                            "level": level, "instruction": level}),
            turns=[Turn(index=0, role="assistant", content=f"hello from {level}")],
        )

    two = {"bad": traj("bad"), "good": traj("good")}
    pairs = pairs_from_group(two, criterion_name="friendliness",
                             criterion_description="warmth",
                             pair_levels=[("bad", "good")])
    assert len(pairs) == 1
    assert set(pairs[0].levels) == {"bad", "good"}


def test_a_narrowed_group_only_requires_the_levels_its_pairs_reference():
    """`ok` missing is fine when no requested pair mentions it, and an error when one
    does -- otherwise a short group would silently yield fewer pairs than planned."""
    ctx = TaskContext(domain="airline", task_id="t", policy="p")

    def traj(level: str) -> Trajectory:
        return Trajectory(
            trajectory_id=f"tr-{level}", context=ctx, provenance=Provenance(),
            turns=[Turn(index=0, role="assistant", content=level)],
        )

    two = {"bad": traj("bad"), "good": traj("good")}
    # no pair mentions `ok`, so its absence is not an error
    pairs_from_group(two, criterion_name="c", criterion_description="d",
                     pair_levels=[("bad", "good")])
    # a pair that does mention it must fail loudly
    with pytest.raises(ValueError, match="missing level"):
        pairs_from_group(two, criterion_name="c", criterion_description="d",
                         pair_levels=[("bad", "ok")])
    # and the default still demands all three
    with pytest.raises(ValueError, match="missing level"):
        pairs_from_group(two, criterion_name="c", criterion_description="d")
