"""Tests for the steering layer.

These check the *contract* -- that the baseline instructions are distinct, that they
reach the agent's system prompt after the policy, and that the invariants the
instructions are supposed to hold are actually written into them. Whether a 4B
model obeys is not testable here; that is what step 3's eyeball run is for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.metaeval.steering import (
    CRITERIA,
    LEVELS,
    agent_name,
    get_criterion,
    register_steered_agents,
)
from src.metaeval.steering.criteria import BASELINE_INSTRUCTIONS, STEERING_WRAPPER
from tests._tau3 import needs_tau3

# Criteria carrying the legacy hand-written instructions. `task_resolution`, added
# 2026-08-11, deliberately has none -- its steering text comes from the generator per
# data point -- so contract tests about instruction *text* run over these, while tests
# about descriptions run over every criterion.
BASELINE = sorted(set(CRITERIA) & set(BASELINE_INSTRUCTIONS))


def _retired_tool_use_criterion():
    """`tool_use_relevancy`, kept as a meta-prompt fixture after it left the roster.

    Replaced by `task_resolution` on 2026-08-11, but still the sharpest test of the
    meta-prompt: it is where "aim at the opposite" is least obvious, and where the
    retired rules did the most damage.
    """
    from src.metaeval.steering.criteria import TOOL_USE_RELEVANCY

    return TOOL_USE_RELEVANCY


def _flat(text: str) -> str:
    """Collapse whitespace before matching.

    The instructions are hard-wrapped in the source, so a phrase can straddle a
    newline and a naive substring check fails on text that is present.
    """
    return " ".join(text.split())


def test_three_criteria_loaded_from_config():
    assert set(CRITERIA) == {"friendliness", "communication_clarity",
                             "task_resolution"}
    for name in BASELINE:
        assert set(CRITERIA[name].instructions) == set(LEVELS)


def test_baseline_instructions_are_all_distinct():
    texts = [get_criterion(c).instruction(lvl) for c in BASELINE for lvl in LEVELS]
    assert len(texts) == len(BASELINE) * 3
    assert len(set(texts)) == len(texts)


def test_a_criterion_without_a_baseline_says_so_instead_of_failing_obscurely():
    """`task_resolution` has no hand-written instructions on purpose, so the error has
    to point at the generator or the next reader takes it for a bug."""
    with pytest.raises(ValueError, match="no hand-written baseline"):
        get_criterion("task_resolution").instruction("bad")


def test_unknown_criterion_and_level_are_rejected():
    with pytest.raises(ValueError, match="unknown criterion"):
        get_criterion("politeness")
    with pytest.raises(ValueError, match="no instruction for level"):
        get_criterion("friendliness").instruction("terrible")


@pytest.mark.parametrize("criterion", BASELINE)
@pytest.mark.parametrize("level", LEVELS)
def test_every_instruction_forbids_meta_commentary(criterion, level):
    """A trajectory that mentions its own steering is a giveaway to the grader."""
    text = _flat(get_criterion(criterion).instruction(level))
    assert "Do not mention, quote or allude to this objective" in text


@pytest.mark.parametrize("criterion", BASELINE)
@pytest.mark.parametrize("level", LEVELS)
def test_every_instruction_licenses_underperformance(criterion, level):
    """Models resist being asked to be worse; the objective has to override."""
    text = _flat(get_criterion(criterion).instruction(level))
    assert "takes strict precedence over your default behaviour" in text


def test_criterion_descriptions_do_not_leak_the_steering():
    """A grader must be able to apply the description without knowing an
    instruction existed -- it defines the axis, it does not describe the intervention."""
    for c in CRITERIA.values():
        low = c.description.lower()
        for word in ("instruct", "steer", "objective", "told to", "prompt"):
            assert word not in low, f"{c.name} description leaks: {word!r}"


def test_descriptions_scope_each_criterion_away_from_the_others():
    """Three axes judged on the same trajectory need explicit boundaries, or a
    grader scores overall quality three times."""
    assert "not whether the request was resolved" in _flat(
        CRITERIA["friendliness"].description)
    assert "not the warmth" in _flat(CRITERIA["communication_clarity"].description)
    resolution = _flat(CRITERIA["task_resolution"].description)
    assert "not the manner" in resolution
    # A correct refusal is resolution -- several tau3 tasks test exactly that, and a
    # grader scoring refusals as failures would invert the criterion.
    assert "refusal counts as resolution" in resolution


# --------------------------------------------------------------------------- #
# Registration and the resulting prompt
# --------------------------------------------------------------------------- #

@needs_tau3
def test_registration_is_idempotent():
    """Called from a CLI, a test and possibly a notebook in one process.
    tau3's register_agent_factory raises on a duplicate name."""
    first = register_steered_agents()
    second = register_steered_agents()
    assert len(first) == len(BASELINE) * 3 and first == second


def test_agent_names_are_unique_per_criterion_and_level():
    names = {agent_name(c, lvl) for c in BASELINE for lvl in LEVELS}
    assert len(names) == len(BASELINE) * 3


@needs_tau3
def test_instruction_is_appended_after_tau3s_own_prompt():
    """Appended, not substituted -- so tau3's own instructions and policy survive."""
    from tau2.runner.build import build_agent, build_environment

    register_steered_agents()
    env = build_environment("airline")
    agent = build_agent("steered:friendliness:bad", env, llm="dummy", llm_args={})

    prompt = agent.system_prompt
    policy = env.get_policy().strip()
    assert policy in prompt
    assert "Be cold and transactional" in prompt
    assert prompt.index(policy) < prompt.index("Be cold and transactional")


def test_wrapper_keeps_only_metric_agnostic_invariants():
    """The wrapper no longer promises task success or policy compliance.

    Both were retired because each breaks when it IS the metric: "still resolve
    the request" is incoherent if the metric is task success. What survives is the
    licence to underperform and the ban on meta-commentary, which hold for any
    metric.
    """
    text = _flat(get_criterion("friendliness").instruction("bad"))
    assert "takes strict precedence over your default behaviour" in text
    assert "Do not mention, quote or allude to this objective" in text
    assert "less successful" in text, "underperformance must cover failing the task"
    assert "still get the customer's request resolved" not in text
    assert "policy> above in full" not in text


@needs_tau3
def test_each_registered_agent_carries_only_its_own_instruction():
    """Nine closures, not one module-level variable: concurrent simulations on
    different criteria must not read each other's steering."""
    from tau2.runner.build import build_agent, build_environment

    register_steered_agents()
    env = build_environment("airline")
    prompts = {}
    for criterion in BASELINE:
        for level in LEVELS:
            a = build_agent(agent_name(criterion, level), env,
                            llm="dummy", llm_args={})
            prompts[(criterion, level)] = a.system_prompt

    assert len(set(prompts.values())) == len(BASELINE) * 3
    for (criterion, level), prompt in prompts.items():
        flat = _flat(prompt)
        assert _flat(get_criterion(criterion).instructions[level]) in flat
        # and no other level's instruction leaked in
        for other in LEVELS:
            if other != level:
                assert _flat(get_criterion(criterion).instructions[other]) not in flat


@needs_tau3
def test_steered_agent_is_otherwise_tau3s_own_agent():
    """Only the system prompt differs, or steered and unsteered runs are not
    comparable."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.runner.build import build_agent, build_environment

    register_steered_agents()
    env = build_environment("airline")
    steered = build_agent("steered:friendliness:good", env, llm="dummy", llm_args={})
    plain = build_agent("llm_agent", env, llm="dummy", llm_args={})

    assert isinstance(steered, LLMAgent)
    # `_abc_impl` is added by ABCMeta on any subclass, not by us.
    overridden = {n for n in vars(type(steered))
                  if not n.startswith("__") and n != "_abc_impl"}
    assert overridden == {"system_prompt", "steering_instruction"}
    # The steered prompt is the plain one plus the instruction.
    assert steered.system_prompt.startswith(plain.system_prompt)


# --------------------------------------------------------------------------- #
# Group configuration
# --------------------------------------------------------------------------- #

def test_agent_is_random_across_groups_and_fixed_within_one():
    from src.metaeval.sources.tau2_source import AGENTS, pick_agent

    once = pick_agent("airline", "task_0", "friendliness")
    assert once == pick_agent("airline", "task_0", "friendliness")  # deterministic
    assert once in AGENTS

    chosen = {pick_agent(d, f"task_{i}", c)
              for d in ("airline", "telecom", "retail", "banking_knowledge")
              for i in range(6) for c in CRITERIA}
    assert chosen == set(AGENTS), "some agent is never selected"


def test_per_domain_step_caps_survived_the_move_into_the_library():
    """A capped run is never graded, so these values matter. Measured on the calibration runs."""
    from src.metaeval.sources.tau2_source import DOMAINS

    assert DOMAINS["telecom"]["max_steps"] == 120
    assert DOMAINS["airline"]["max_steps"] == 30
    assert DOMAINS["banking_knowledge"]["retrieval"] == "bm25_grep"
    assert DOMAINS["retail"]["retrieval"] is None


def test_the_runner_reads_the_roster_rather_than_restating_it():
    """One copy of the roster, and the runner imports it.

    Checked on the source text rather than by running the tool, because the failure this
    guards against is a *second definition* -- and a second definition that happens to agree
    today passes any behavioural test while drifting the moment either copy is edited. A
    normaliser bug of exactly that shape is why the roster was centralised.
    """
    src = Path("tools/run_pipeline.py").read_text()
    assert "from src.metaeval.sources.tau2_source import" in src
    for name in ("DOMAINS", "AGENTS", "JUDGES"):
        assert f"{name} = {{" not in src, (
            f"{name} looks like it is redefined in run_pipeline.py; it belongs to "
            f"src/metaeval/sources/tau2_source.py alone")


@pytest.mark.parametrize("level", ["bad", "ok"])
def test_degrading_tool_use_instructions_are_bounded(level):
    """Rule 5. Unbounded, "be inefficient" ran qwen-a3b to 40 KB_search calls into
    the step cap on banking -- ungraded, and `bad` indistinguishable from `ok`.
    An instruction that destroys its own separation is worse than no instruction."""
    text = _flat(BASELINE_INSTRUCTIONS["tool_use_relevancy"][level])
    assert "once you have what" in text.lower(), "no stopping condition"


# --------------------------------------------------------------------------- #
# The generator's interface: plain data only
# --------------------------------------------------------------------------- #

def test_meta_prompt_takes_only_plain_data():
    """No framework objects cross into the generator.

    The signature is the guarantee: a metric, a description string, and tool names.
    Anything requiring a tau3 Task or Environment would make this tau3-only.
    """
    import inspect

    from src.metaeval.steering.generate import build_meta_prompt

    params = list(inspect.signature(build_meta_prompt).parameters)
    assert params == ["criterion", "task_description", "tool_names"]


def test_generator_is_never_shown_ground_truth():
    """Leakage is structural, not policed.

    The generator wrote "conclude after stating the refusal" when it had been told
    the outcome. Nothing in its prompt may carry a resolution or a gold document.
    """
    from src.metaeval.schema import TaskContext
    from src.metaeval.steering.generate import build_meta_prompt

    ctx = TaskContext(
        domain="banking_knowledge", task_id="task_001", policy="SECRET POLICY TEXT",
        task_purpose="Testing that the agent refuses a cancellation that is not allowed",
        required_documents=["doc_gold_001"],
        user_instructions="I want the highest cash back card with no annual fee.",
    )
    prompt = build_meta_prompt(_retired_tool_use_criterion(),
                              ctx.user_instructions, ["KB_search", "grep"])
    assert "doc_gold_001" not in prompt
    assert "refuses a cancellation" not in prompt
    assert "SECRET POLICY TEXT" not in prompt
    assert "highest cash back card" in prompt  # the task description does reach it


def test_no_validation_layer_exists():
    """Removed on request after measuring it: 2 of 12 generations flagged, both on
    the same trivial pattern, and the LLM reviewer never caught a real defect."""
    import importlib

    from src.metaeval.steering import generate

    assert not hasattr(generate, "REVISION_PROMPT")
    for name in ("check_instructions", "review_instructions", "run_checks"):
        assert name not in generate.__dict__, name
    try:
        importlib.import_module("src.metaeval.steering.validate")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("steering.validate should have been deleted")


def test_bad_level_must_aim_at_the_opposite_of_the_metric():
    """A merely-lukewarm BAD was the observed failure: instructions withheld good
    behaviour instead of pursuing the opposite."""
    from src.metaeval.steering.generate import build_meta_prompt

    prompt = _flat(build_meta_prompt(get_criterion("friendliness"),
                                     "Cancel my flight.", ["cancel_reservation"]))
    assert "AIM AT THE OPPOSITE OF THE METRIC" in prompt
    assert "do not merely withhold good behaviour" in prompt
    assert "Name the opposite explicitly" in prompt


def test_meta_prompt_forbids_copying_task_particulars():
    """Naming the customer and their booking id made GOOD far more specific than
    BAD, so the levels differed by detail carried rather than by the metric."""
    from src.metaeval.steering.generate import build_meta_prompt

    prompt = _flat(build_meta_prompt(get_criterion("friendliness"),
                                     "Cancel my flight.", ["cancel_reservation"]))
    assert "NOT TO ITS PARTICULARS" in prompt
    assert "no names" in prompt and "identifiers" in prompt
    assert "learns those from the conversation itself" in prompt


def test_meta_prompt_permits_omission_and_failure():
    """A bad level may leave work out or fail the task."""
    from src.metaeval.steering.generate import build_meta_prompt

    prompt = _flat(build_meta_prompt(_retired_tool_use_criterion(),
                                     "Cancel my flight.", ["cancel_reservation"]))
    assert "Nothing is off limits in how badly the BAD level may perform" in prompt
    assert "failing the task outright" in prompt
    # and it no longer demands task success
    assert "still resolve the request at every level" not in prompt



def test_telecom_samples_the_base_split():
    """The full telecom set is 2,285 machine-generated variants of 114 scenarios.

    Sampling 36 tasks from `full` would look diverse and not be. The plan has said
    `base` from the start, but the call omitted the split until 2026-08-11, so every
    telecom run before then used `full`.
    """
    from src.metaeval.sources.tau2_source import DOMAINS

    assert DOMAINS["telecom"]["split"] == "base"
    # the other three have a single set, so no split to choose
    assert {d: c["split"] for d, c in DOMAINS.items() if d != "telecom"} == {
        "airline": None, "banking_knowledge": None, "retail": None}


def test_every_domain_declares_a_split():
    """A domain added without one would silently fall back to tau3's default set."""
    from src.metaeval.sources.tau2_source import DOMAINS

    for domain, cfg in DOMAINS.items():
        assert "split" in cfg, domain


def test_no_qwen_agent():
    """Both Qwen agents called telecom's *customer-side* tools and were dropped.

    Measured wrong-side calls on telecom: qwen3-4b 13/25, qwen3-30b-a3b 19/70, against
    0 for gpt-oss-20b (0/23), gpt-oss-120b (0/43) and nemotron-lightning (0/41). It
    tracks the family, not model size. It also ignores the steer -- wrong-side calls
    went bad 5 / ok 6 / good 8 -- so it swamps tool-use relevancy instead of varying
    along it.
    """
    from src.metaeval.sources.tau2_source import AGENTS

    assert not any("qwen" in m.lower() for m, _ in AGENTS.values())


def test_every_agent_is_serverless():
    """No `#` means no dedicated deployment: nothing to create, nothing to warm.

    tau3 does not retry a cold start -- `generate()` raises on the first error -- so a
    scaling deployment kills a whole simulation. Every earlier run lost minutes to it.
    """
    from src.metaeval.sources.tau2_source import AGENTS

    for key, (model, _) in AGENTS.items():
        assert "#" not in model, f"{key} needs a dedicated deployment"


def test_gpt_oss_agents_pin_reasoning_effort():
    """Unset or 'medium' spends 87% of the token budget on hidden reasoning; 'high'
    returns an empty string."""
    from src.metaeval.sources.tau2_source import AGENTS

    for key, (model, args) in AGENTS.items():
        if "gpt-oss" in model:
            assert args.get("reasoning_effort") == "low", key


def test_the_user_simulator_is_not_also_an_agent():
    """No self-play: previously a third of groups ran one model against itself."""
    from src.metaeval.sources.tau2_source import AGENTS, USER

    assert all(m != USER for m, _ in AGENTS.values())


def test_max_errors_is_the_tau3_default():
    """Cut to 4, reverted to 10 on 2026-08-11.

    At 4, telecom's median fell from 8 spoken turns to 4 -- too thin to judge -- and it
    contaminated measurement: comparing 20b against 120b on tool-use steering, 2 of the
    20b's 3 runs were truncated at the cap while all 3 of the 120b's finished, because
    the 120b barely errs. The apparent difference disappeared at 10.
    """
    import inspect

    from src.metaeval.sources import tau2_source

    src = inspect.getsource(tau2_source.run_steered)
    assert "max_errors=10" in src


def test_judge_panel_matches_the_roster_doc():
    """The panel is code now, not just a markdown table.

    It once lived only in a markdown table while the filter module defaulted to two
    entirely different models -- ones this project has no working key for -- so the
    documented panel and the runnable default disagreed and nothing failed.
    """
    from src.metaeval.sources.tau2_source import JUDGES

    assert set(JUDGES) == {"judge1", "judge2"}
    assert "nemotron-lightning" in JUDGES["judge1"]
    # judge2 was gemma-4-26b-a4b-it until 2026-08-11. Its deployment sat initializing for
    # 30+ minutes and it is not available serverless, so it was unreachable by any route.
    assert "gpt-oss-120b" in JUDGES["judge2"]


def test_a_judge_may_also_be_an_agent():
    """Deliberate, and safe only because of the fixed-agent property below.

    `nemotron-lightning` is both judge 1 and an agent. That is not the
    self-preference trap it resembles: the agent is fixed within a group, so both
    sides of every pair come from the same model and a judge never compares its own
    output against another model's -- which is the comparison self-preference needs.

    The earlier rule ("no model does double duty across the generate/judge boundary")
    was a convenience that fell out of judges happening to be unable to call tools. It
    was never the thing protecting us.
    """
    from src.metaeval.sources.tau2_source import AGENTS, JUDGES

    agent_models = {m for m, _ in AGENTS.values()}
    assert JUDGES["judge1"] in agent_models


def test_the_agent_is_fixed_within_a_group():
    """The invariant that makes a judge-as-agent safe. Load-bearing.

    Randomising the agent *within* a group would put two families on opposite sides of
    a pair and silently reintroduce self-preference. `run_group` must resolve the agent
    once, before the per-level loop.
    """
    import inspect

    from src.metaeval.sources import tau2_source

    src = inspect.getsource(tau2_source.run_group)
    before, _, after = src.partition("for level in LEVELS:")
    assert after, "run_group no longer loops over levels; re-check this test"
    # the agent is chosen above the loop and only passed in below it
    assert "pick_agent(" in before
    assert "pick_agent(" not in after


def test_the_pool_also_resolves_the_agent_once_per_cell():
    """The same invariant on the path that will generate the paper's data.

    The source-text guard above inspects `run_group`, which `tools/run_pipeline.py` never
    calls -- it drives `run_steered_resilient` per level directly. So the assertion `a81d323`
    installed *because* this property is "inherited, not intrinsic" had quietly stopped
    covering the production path. `Cell.agent` is resolved once in `build_cells` and read by
    the trajectory stage for every level, which is what makes it structural here rather than
    merely true.
    """
    import inspect

    import tools.run_pipeline as rp

    src = inspect.getsource(rp.build_cells)
    assert "pick_agent(" in src, "build_cells must be where the agent is resolved"

    stage_src = inspect.getsource(rp.real_stages)
    _, _, after = stage_src.partition("def trajectory(")
    assert after, "real_stages no longer defines a trajectory stage; re-check this test"
    assert "cell.agent" in after, "the per-level stage must read the cell's fixed agent"
    assert "pick_agent(" not in after, "and must never re-pick it per level"


def test_a_cell_carries_one_agent_for_every_level():
    """Checked on the object rather than its source: one agent, both levels."""
    from tools.run_pipeline import Cell

    cell = Cell(domain="airline", criterion="friendliness", task_index=0, task_id="t",
                gap=("bad", "good"), agent="gpt-oss-120b")
    assert len({cell.agent for _ in cell.levels}) == 1


def test_stored_pairs_never_mix_two_agents():
    """The same invariant, checked against real data rather than source text.

    **Globs every generated directory, not one.** This looked only in `data/step3/runs`, the
    retired three-level directory, and `pytest.skip`ped when it was empty -- so it stopped
    covering `runs_final` (the 36 pairs the paper uses) and would never have covered the
    pool's output either. A guard that skips is a guard that is gone.
    """
    import glob
    import json

    files = [f for pattern in ("data/step*/**/*.pairs.json", "data/step*/*.pairs.json")
             for f in glob.glob(pattern, recursive=True)]
    if not files:
        pytest.skip("no generated pairs on disk")
    mixed = 0
    for f in files:
        for e in json.load(open(f)):
            if (e["trajectory_1"]["provenance"]["agent_model"]
                    != e["trajectory_2"]["provenance"]["agent_model"]):
                mixed += 1
    assert mixed == 0


def test_no_judge_is_the_user_simulator():
    """Kept for a narrower reason than before.

    The fixed-agent argument would cover the simulator too -- it writes the customer
    turns on both sides of a pair. But the simulator is the one participant whose own
    system prompt is withheld from graders, so letting it grade would hand one panel
    member context the others were denied, breaking the parity the study rests on.
    """
    from src.metaeval.sources.tau2_source import JUDGES, USER

    assert USER not in JUDGES.values()


def test_spare_judges_are_deployed_and_unassigned():
    """gemma-3-4b-it was the listed spare but had no deployment, so it was never
    reachable. A spare that cannot be called is not a spare."""
    from src.metaeval.sources.tau2_source import JUDGES, SPARE_JUDGES

    assert len(SPARE_JUDGES) == 3
    assert any("qwen3-4b" in m for m in SPARE_JUDGES)
    assert any("qwen3-1p7b" in m for m in SPARE_JUDGES)
    # Demoted from judge2, kept last so the swap stays traceable.
    assert any("gemma-4-26b" in m for m in SPARE_JUDGES)
    for m in SPARE_JUDGES:
        assert "gemma-3-4b" not in m
        assert "#" in m, "a dedicated model must carry its deployment ref"
        assert m not in JUDGES.values(), "a spare cannot already be seated"


def test_the_seated_panel_needs_no_deployment():
    """Both judges must be serverless.

    This is the property that was violated and cost 30+ minutes: judge2's scale-to-zero
    deployment never came up, and a panel is unusable if seating it means waiting on a
    contended GPU. `#` marks a dedicated deployment ref, so its absence is the test.
    """
    from src.metaeval.sources.tau2_source import JUDGES

    for name, model in JUDGES.items():
        assert "#" not in model, (
            f"{name} points at a dedicated deployment ({model}); the panel must be "
            "serverless so a batch never waits on a cold replica")


def test_steering_agent_names_are_unique_per_task():
    """The registry is keyed by agent name and has no removal.

    The generator writes a *different* instruction per task, so a tag built only from
    (domain, criterion, level) collides across the three tasks of one (domain,
    criterion). `register_instruction` then refuses to overwrite -- correctly, since
    silently running the previous task's instruction would be worse -- which failed 2 of
    every 3 groups in the first 108-simulation attempt.
    """
    from src.metaeval.steering import agent_name

    names = {agent_name("friendliness", level,
                        f"airline-friendliness-t{task}-{level}")
             for task in range(3) for level in LEVELS}
    assert len(names) == 9, names


def test_the_runner_puts_the_task_in_the_agent_tag():
    """Guards the specific line that broke, since the failure is silent-ish: it looks
    like a per-cell error rather than a systematic collision.

    Asserted on the source line rather than by running the runner, because the collision is
    a property of the *tag* the runner builds, and reaching that tag through `run_task` means
    standing up tau3, a registry and two judges.
    """
    import pathlib

    src = pathlib.Path("tools/run_pipeline.py").read_text()
    assert ('instruction_tag=f"{cell.domain}-{cell.criterion}-t{cell.task_index}-{level}"'
            in src), "the agent tag no longer carries the task, so levels will collide"


def test_every_llm_seat_has_a_timeout_and_retries():
    """Nothing else in the stack sets either.

    tau3's `generate()` passes llm_args straight through and configures no timeout, so a
    hung socket blocks forever -- a 108-simulation run stalled exactly that way: alive,
    0% CPU, no output for seven minutes, with a healthy API and nothing to catch.

    Retries are separate and just as necessary: `generate()` does
    `logger.error(e); raise e` and the run config's `max_retries` is batch-level, so one
    transient 500 kills a whole simulation. Three telecom groups were lost to
    `ServiceUnavailableError` that way.
    """
    from src.metaeval.sources.tau2_source import AGENTS, JUDGE_ARGS, USER_ARGS

    seats = {k: a for k, (_, a) in AGENTS.items()}
    seats["user_simulator"] = USER_ARGS
    seats["judges"] = JUDGE_ARGS
    for name, args in seats.items():
        assert args.get("timeout"), f"{name} has no timeout"
        assert args.get("num_retries", 0) >= 1, f"{name} has no retries"


# --------------------------------------------------------------------------- #
# Registration under concurrency
#
# tau3's registry is global process state with no removal, and
# `register_agent_factory` raises on a duplicate name. Both registration paths do a
# check-then-register, which is a race the moment a worker pool runs simulations
# concurrently: two threads see the name absent, both register, the loser raises and its
# whole simulation dies -- ~40k tokens for one lost trajectory. These tests drive the race
# hard enough to fail reliably without the lock.
# --------------------------------------------------------------------------- #

@needs_tau3
def test_concurrent_registration_of_one_name_never_raises(monkeypatch):
    """The pool's shape: many workers, same instruction, same name.

    **The delay is what makes this a real test.** Without the lock and without it, 64
    threads produced 0 errors across 5 trials -- building the factory is a few
    microseconds, so the check-then-register window almost never loses. Slowing that build
    to 2ms widens the window to what real work looks like, and the same run then produced
    15 duplicate-name failures. A concurrency test that cannot fail is worse than none.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from src.metaeval.steering import agent as agent_mod
    from src.metaeval.steering.agent import register_instruction

    real = agent_mod._make_instruction_factory

    def slow(instruction):
        time.sleep(0.002)
        return real(instruction)

    monkeypatch.setattr(agent_mod, "_make_instruction_factory", slow)

    name = "steered:friendliness:bad:concurrency-probe-1"
    text = "Be warm and specific."

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: register_instruction(text, name), range(64)))

    assert set(results) == {name}, "every caller gets the name back, nobody raises"


@needs_tau3
def test_concurrent_registration_of_distinct_names_registers_all(monkeypatch):
    """Distinct tags is the normal case -- one per (domain, criterion, task, level).

    Same widened window as above, for the same reason.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from tau2.registry import registry

    from src.metaeval.steering import agent as agent_mod
    from src.metaeval.steering.agent import register_instruction

    real = agent_mod._make_instruction_factory

    def slow(instruction):
        time.sleep(0.002)
        return real(instruction)

    monkeypatch.setattr(agent_mod, "_make_instruction_factory", slow)

    names = [f"steered:friendliness:ok:concurrency-probe-2-{i}" for i in range(48)]

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda n: register_instruction(f"instruction for {n}", n), names))

    for n in names:
        factory = registry.get_agent_factory(n)
        assert factory is not None, n
        assert factory.steering_instruction == f"instruction for {n}"


@needs_tau3
def test_re_registering_the_same_name_is_what_makes_a_retry_safe():
    """`run_steered_resilient` re-runs a failed simulation, which re-registers the same
    tag with the same text. That must be a no-op, not tau3's duplicate-name error."""
    from src.metaeval.steering.agent import register_instruction

    name = "steered:task_resolution:good:retry-probe"
    text = "Resolve the request fully."
    assert register_instruction(text, name) == name
    assert register_instruction(text, name) == name      # the retry
    assert register_instruction(text, name) == name      # and again


@needs_tau3
def test_a_different_instruction_under_a_taken_name_still_raises():
    """The lock must not turn a real collision into a silent overwrite: running the old
    instruction under a name the caller thinks is theirs would corrupt the steering."""
    from src.metaeval.steering.agent import register_instruction

    name = "steered:friendliness:good:collision-probe"
    register_instruction("first text", name)
    with pytest.raises(ValueError, match="different"):
        register_instruction("second, different text", name)


@needs_tau3
def test_concurrent_baseline_registration_is_safe():
    """`run_steered` calls this on the baseline path while another worker may be in
    `register_instruction` on the generated path, so both share one lock."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: register_steered_agents(), range(32)))

    assert all(r == results[0] for r in results)
    assert len(results[0]) == len(BASELINE) * 3


@needs_tau3
def test_both_registration_paths_concurrently():
    """The realistic mix: baseline and generated registrations interleaved."""
    from concurrent.futures import ThreadPoolExecutor

    from src.metaeval.steering.agent import register_instruction

    def work(i):
        if i % 2:
            return register_steered_agents() and "baseline"
        return register_instruction(f"text {i}", f"steered:friendliness:bad:mixed-{i}")

    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(work, range(48)))

    assert len(results) == 48 and all(results), "no call returned empty or raised"


# --------------------------------------------------------------------------- #
# Simulation-level retry
#
# `CALL_CONTROL` retries one HTTP request twice. When those are exhausted the exception
# leaves `run_single_task` and takes the whole simulation with it -- every turn already
# generated discarded, ~97k tokens for a two-level group. A DEPLOYMENT_SCALING_UP cold
# start outlasts two litellm retries comfortably, which is how three telecom levels were
# lost. These drive `run_steered_resilient` against a stubbed `run_steered`, so no
# simulation runs and no token is spent.
# --------------------------------------------------------------------------- #

def test_resilient_returns_the_trajectory_and_one_attempt_on_success(monkeypatch):
    from src.metaeval.sources import tau2_source as ts

    monkeypatch.setattr(ts, "run_steered", lambda *a, **k: "TRAJ")
    traj, attempts = ts.run_steered_resilient("airline", None, "friendliness", "bad", "x",
                                              backoff_s=0)
    assert traj == "TRAJ" and attempts == 1


def test_resilient_recovers_a_transient_failure(monkeypatch):
    """The cold-start case: first attempt dies, the replica comes up, the second works."""
    from src.metaeval.sources import tau2_source as ts

    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("DEPLOYMENT_SCALING_UP")
        return "TRAJ"

    monkeypatch.setattr(ts, "run_steered", flaky)
    traj, attempts = ts.run_steered_resilient("airline", None, "friendliness", "bad", "x",
                                              backoff_s=0)
    assert traj == "TRAJ"
    assert attempts == 2, "the attempt count is reported, not hidden"


def test_resilient_gives_up_after_the_configured_attempts(monkeypatch):
    """Bounded: a permanently dead deployment must not loop forever."""
    from src.metaeval.sources import tau2_source as ts

    calls = {"n": 0}

    def dead(*a, **k):
        calls["n"] += 1
        raise RuntimeError("permanently down")

    monkeypatch.setattr(ts, "run_steered", dead)
    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        ts.run_steered_resilient("airline", None, "friendliness", "bad", "x",
                                 attempts=3, backoff_s=0)
    assert calls["n"] == 3


def test_resilient_preserves_the_original_error(monkeypatch):
    """The final message must name the real cause, or a dead deployment is
    indistinguishable from a bug in our own code."""
    from src.metaeval.sources import tau2_source as ts

    monkeypatch.setattr(ts, "run_steered",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad task id")))
    with pytest.raises(RuntimeError, match="ValueError: bad task id"):
        ts.run_steered_resilient("airline", None, "friendliness", "bad", "x",
                                 attempts=2, backoff_s=0)


def test_resilient_passes_every_argument_through(monkeypatch):
    """It wraps `run_steered` and must not swallow or reorder its arguments -- the
    instruction and the tag in particular, since a dropped tag collides in tau3's
    registry and a dropped instruction silently runs the baseline."""
    from src.metaeval.sources import tau2_source as ts

    seen = {}

    def capture(*a, **k):
        seen["args"], seen["kwargs"] = a, k
        return "TRAJ"

    monkeypatch.setattr(ts, "run_steered", capture)
    ts.run_steered_resilient("airline", "TASK", "friendliness", "bad", "gpt-oss",
                             seed=7, instruction="be warm", instruction_tag="a-b-t0-bad",
                             backoff_s=0)
    assert seen["args"] == ("airline", "TASK", "friendliness", "bad", "gpt-oss")
    assert seen["kwargs"] == {"seed": 7, "instruction": "be warm",
                              "instruction_tag": "a-b-t0-bad"}
    assert "backoff_s" not in seen["kwargs"], "wrapper args must not leak to run_steered"


def test_resilient_reports_each_retry(monkeypatch):
    """The pool needs to log retries: silent ones make a slow run look mysterious."""
    from src.metaeval.sources import tau2_source as ts

    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("saturated")
        return "TRAJ"

    seen = []
    monkeypatch.setattr(ts, "run_steered", flaky)
    ts.run_steered_resilient("airline", None, "friendliness", "bad", "x", attempts=3,
                             backoff_s=0, on_retry=lambda n, e, secs: seen.append(n))
    assert seen == [1, 2], "one callback per failed attempt, not per call"


def test_resilient_reports_the_final_failure_too(monkeypatch):
    """It fires on every failed attempt including the last. The last one is a lost simulation
    whose time nothing else records -- the caller needs it to account for the waste."""
    from src.metaeval.sources import tau2_source as ts

    monkeypatch.setattr(ts.time, "sleep", lambda s: None)
    monkeypatch.setattr(ts, "run_steered",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    seen = []
    with pytest.raises(RuntimeError):
        ts.run_steered_resilient("a", None, "b", "c", "d", attempts=3, backoff_s=0,
                                 on_retry=lambda n, e, secs: seen.append((n, secs >= 0)))
    assert seen == [(1, True), (2, True), (3, True)], "including the attempt that gave up"


def test_resilient_passes_the_failed_attempts_elapsed_time(monkeypatch):
    """Tokens of a lost simulation are unknowable; its seconds are not."""
    from src.metaeval.sources import tau2_source as ts

    def slow_fail(*a, **k):
        import time as _t
        _t.sleep(0.02)
        raise RuntimeError("down")

    monkeypatch.setattr(ts, "run_steered", slow_fail)
    seen = []
    with pytest.raises(RuntimeError):
        ts.run_steered_resilient("a", None, "b", "c", "d", attempts=1, backoff_s=0,
                                 on_retry=lambda n, e, secs: seen.append(secs))
    assert seen and seen[0] >= 0.02


def test_resilient_waits_between_attempts(monkeypatch):
    """Retrying faster than a cold start just burns attempts on the same cause."""
    from src.metaeval.sources import tau2_source as ts

    slept = []
    monkeypatch.setattr(ts.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(ts, "run_steered",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        ts.run_steered_resilient("airline", None, "friendliness", "bad", "x",
                                 attempts=3, backoff_s=30)
    assert slept == [30, 30], "waits before each retry, and not after the last failure"


def test_resilient_with_attempts_one_is_a_plain_call(monkeypatch):
    from src.metaeval.sources import tau2_source as ts

    calls = {"n": 0}

    def dead(*a, **k):
        calls["n"] += 1
        raise RuntimeError("x")

    monkeypatch.setattr(ts, "run_steered", dead)
    with pytest.raises(RuntimeError):
        ts.run_steered_resilient("a", None, "b", "c", "d", attempts=1, backoff_s=0)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# generate_instructions -- the P1 stage
#
# Entirely untested until now, which mattered because it is the only stage whose output is
# *text the agent then obeys*: a silently empty body simulates a group with no steering, and
# the pair that comes out looks structurally fine while measuring nothing. `litellm` is
# stubbed here, so no call is made.
# --------------------------------------------------------------------------- #

def _generator_reply(payload: str):
    from unittest.mock import MagicMock

    msg = MagicMock()
    msg.content = payload
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = type("U", (), {"prompt_tokens": 900, "completion_tokens": 300,
                                "completion_tokens_details": None})()
    return resp


def _gen(monkeypatch, payload, **kw):
    import litellm

    from src.metaeval.steering.generate import generate_instructions

    monkeypatch.setattr(litellm, "completion",
                        lambda **k: _generator_reply(payload))
    return generate_instructions(CRITERIA["friendliness"], "book a flight", ["search"],
                                 **kw)


def test_generate_instructions_returns_all_three_levels(monkeypatch):
    g = _gen(monkeypatch, '{"bad": "be cold", "ok": "be neutral", "good": "be warm"}')
    assert set(g.instructions) == set(LEVELS)
    assert g.instructions["good"] == "be warm"


def test_generate_instructions_returns_empty_bodies_rather_than_inventing_text(monkeypatch):
    """A generator that answers nothing useful must not be papered over: the pool treats an
    empty body as an incomplete phase and regenerates, which is only possible if the
    emptiness survives to it."""
    g = _gen(monkeypatch, "I would rather not.")
    assert all(g.instructions[lvl] == "" for lvl in LEVELS)


def test_generate_instructions_records_usage_under_the_given_item_id(monkeypatch):
    """The accounting key. Measured before this existed: the default
    `"{domain}:{task_id}"` is not unique per datapoint -- one task carries all three
    criteria -- so two P1 calls collapsed into one row and the per-pair chain could not be
    assembled."""
    from src.metaeval.usage import UsageMeter

    m = UsageMeter().start()
    _gen(monkeypatch, '{"bad": "a", "ok": "b", "good": "c"}', meter=m,
         domain="airline", task_id="0", item_id="airline.friendliness.t0")
    r = m.records[0]
    assert r.item_id == "airline.friendliness.t0"
    assert r.role == "instruction-gen"
    assert r.label == "friendliness"
    assert r.prompt_tokens == 900


def test_generate_instructions_falls_back_to_the_derived_item_id(monkeypatch):
    """Existing callers pass no `item_id` and must keep their current behaviour."""
    from src.metaeval.usage import UsageMeter

    m = UsageMeter().start()
    _gen(monkeypatch, '{"bad": "a", "ok": "b", "good": "c"}', meter=m,
         domain="airline", task_id="7")
    assert m.records[0].item_id == "airline:7"


def test_generate_instructions_works_without_a_meter(monkeypatch):
    g = _gen(monkeypatch, '{"bad": "a", "ok": "b", "good": "c"}')
    assert g.instructions["bad"] == "a"


def test_generate_instructions_recovers_json_from_a_code_fence(monkeypatch):
    """The generator wraps its JSON in markdown often enough that this is the common path,
    not an edge case."""
    g = _gen(monkeypatch,
             '```json\n{"bad": "x", "ok": "y", "good": "z"}\n```')
    assert g.instructions["ok"] == "y"


def test_generate_instructions_strips_surrounding_whitespace(monkeypatch):
    """The body is interpolated into the steering wrapper, so stray newlines land in the
    agent's system prompt."""
    g = _gen(monkeypatch, '{"bad": "  padded  ", "ok": "b", "good": "c"}')
    assert g.instructions["bad"] == "padded"
