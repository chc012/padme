"""Run tau3 to produce a group of three steered trajectories.

The generation path. One `(domain, task, criterion)` in, three trajectories out
-- one steered `bad`, one `ok`, one `good` -- with the agent model, user
simulator, task and seed held fixed across all three. tau3 types stop at the
`Trajectory` boundary; the caller gets our schema.

Also the canonical home for the model roster and the per-domain step caps. Both
were measured on the domain calibration runs and both have exactly one correct value, so
they live in one place: `tools/run_pipeline.py` imports them from here rather than
keeping a second copy.
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from typing import Any, Optional

from ..schema import Provenance, SteeringRecord, Trajectory
from ..steering import agent_name, get_criterion, register_steered_agents
from ..steering.agent import register_instruction

FW = "fireworks_ai/accounts/fireworks/models"

# Address suffix for a **dedicated** Fireworks deployment, which is spelled
# `<model>#accounts/<account>/deployments/<id>`. The account id is the reader's, so it comes
# from `FIREWORKS_ACCOUNT_ID` and nothing here hardcodes one. Deployment ids are named
# `meta-eval-<model>` by convention; `run_evaluators.py` prints the `firectl` recipe that
# creates one.
#
# **Only the user simulator and the spare judges need one.** Every agent and both judges
# on the panel are serverless, so a run with `FIREWORKS_ACCOUNT_ID` unset gets as far as
# the user seat and then fails with a 404 -- which is a loud failure on turn one rather
# than a silent substitution, and is the intended behaviour.
FW_ACCOUNT = os.environ.get("FIREWORKS_ACCOUNT_ID", "<account>")
DEP = f"#accounts/{FW_ACCOUNT}/deployments/meta-eval"

# Agents, settled 2026-08-11 after live probes on telecom -- the only domain where an
# agent has broken. All three are **serverless**: no deployment to create, no cold
# start, and nothing to warm before a run. Every earlier generation run lost minutes
# to `DEPLOYMENT_SCALING_UP`, which tau3 does not retry.
#
# **Both Qwen agents were dropped for calling the customer's tools.** Telecom keeps its
# diagnostics on the customer's side; the agent is meant to ask them to run one.
# Measured wrong-side calls: qwen3-4b 13 of 25, qwen3-30b-a3b 19 of 70, gpt-oss-20b
# 0 of 23, gpt-oss-120b 0 of 43, nemotron-lightning 0 of 41. It tracks the Qwen family,
# not model size -- a 3B-active NVIDIA model makes none. And it ignores the steer
# (wrong-side calls went bad 5 / ok 6 / good 8), so it would swamp tool-use relevancy
# on telecom rather than vary along it.
#
# Neither Qwen is bad everywhere: both ran airline, retail and banking with zero tool
# errors. They are dropped because telecom is in the design, not because they are weak.
# Applied to every LLM seat. **Neither tau3 nor litellm sets a timeout here** -- tau3's
# `generate()` passes llm_args straight through and configures none -- so a hung request
# blocks forever. A 108-simulation run stalled on exactly that: alive, 0% CPU, no log
# output for six minutes, with no error to catch and nothing to time out.
#
# 120s is generous: a whole simulation takes 7-40s, so a single call is a few seconds.
# `num_retries` matters because tau3 never retries -- `generate()` does
# `logger.error(e); raise e` and the run config's `max_retries` is batch-level, so one
# transient 500 kills a whole simulation. Two retries at the litellm layer make transient
# failures self-heal without touching tau3.
#
# **These two retries are not the whole retry story, and they are not the half that saves
# data.** They retry one HTTP request; when both are spent the exception still leaves
# `run_single_task` and takes the whole simulation with it. `run_steered_resilient` below is
# the simulation-level retry that covers that, and a cold start outlasting two litellm
# retries is exactly the case it exists for.
CALL_CONTROL = {"timeout": 120, "num_retries": 2}

AGENTS: dict[str, tuple[str, dict]] = {
    # reasoning_effort MUST be pinned: 'high' returns an empty string,
    # unset/'medium' spends 87% of the budget on hidden reasoning.
    "gpt-oss": (f"{FW}/gpt-oss-20b",
                {"temperature": 0.0, "max_tokens": 10_000,
                 "reasoning_effort": "low", **CALL_CONTROL}),
    # 120B total but ~5B active, so it fits an active-parameter budget and blows a
    # total-parameter one. Best tool competence measured: 1 error in 43 telecom calls,
    # and the fastest (7-38s a simulation against qwen3-4b's 50-87s).
    "gpt-oss-120b": (f"{FW}/gpt-oss-120b",
                     {"temperature": 0.0, "max_tokens": 10_000,
                      "reasoning_effort": "low", **CALL_CONTROL}),
    # 30B total / 3B active, hybrid Mamba-Transformer. Errors more than either gpt-oss
    # (34% of 41 telecom calls) but they are *business-logic* errors -- "Line must be
    # suspended to resume" -- i.e. real tools with wrong preconditions, which is what
    # tool-use relevancy is meant to catch, unlike "Tool not found".
    "nemotron-lightning": (f"{FW}/nemotron-lightning-3p5-30b-a3b",
                           {"temperature": 0.0, "max_tokens": 10_000, **CALL_CONTROL}),
}

# The user simulator keeps `qwen3-30b-a3b`. **Only the simulator can end a
# conversation** -- it emits ###STOP### / ###TRANSFER### and the agent cannot -- so
# termination authority and model capability belong in the same seat. It plays the role
# well (a nervous, non-technical telecom customer, convincingly) and its weakness is
# agent-side tool discipline, which this seat never exercises.
#
# Must be tool-capable: telecom passes 30 user tools and 80 of 97 banking tasks pass
# some. That rules out llama-v3p2-3b and both gemma models, none of which tool-call.
#
# With both Qwens out of the agent pool there is **no self-play left** -- previously a
# third of groups had the same model on both sides of the conversation.
USER = f"{FW}/qwen3-30b-a3b-instruct-2507{DEP}-qwen3-30b-a3b-instruct-2507"
USER_ARGS = {"temperature": 0.0, "max_tokens": 10_000, **CALL_CONTROL}

# Judges. Two LLM votes; the third is the generator's steering intent, so unanimity is
# 3/3 and majority 2/3. Recorded here as code rather than only in a markdown table so
# there is one place the panel can be read from: an earlier version carried its own
# defaults in the judging module, naming two models this project has no working key for,
# and a caller that omitted the models got them instead of the panel.
#
# **`nemotron-lightning` is judge 1 *and* an agent, deliberately.** That looks like the
# self-preference trap and is not one, for a structural reason: **the agent is fixed
# within a group**, so both sides of every pair come from the same model. A judge
# therefore never compares its own output against another model's, which is the
# comparison self-preference needs. Classic judge-prefers-its-own-family bias cannot act
# here. `pair_index.json` records the agent per cell, so the property is checkable on any
# run's output and not merely asserted.
#
# **This safety is inherited, not intrinsic.** It holds only while `run_group` assigns
# one agent per group. Randomising the agent *within* a group would silently reintroduce
# the bias, so `tests/test_steering.py` asserts the fixed-agent property alongside the
# panel itself.
#
# The residual risk is real but different in kind, and measurable rather than fatal:
# familiarity could shift a judge's *calibration* on its own family's trajectories -- a
# per-agent difference in judge/human agreement, not a within-pair thumb on the scale.
# Check judge-vs-intent agreement on nemotron-authored pairs against the others.
#
# **The whole panel is serverless as of 2026-08-11**, which is not cosmetic: it is the
# difference between a 60-second batch and an unbounded wait.
#
# `gemma-4-26b-a4b-it` was judge2 and had to go. Its scale-to-zero deployment sat at
# `Initializing Replica Count: 1` for **30+ minutes** -- past `warm()`'s 15-minute budget
# -- and a direct call failed in 0.2s with `DEPLOYMENT_SCALING_UP`, so the zero-replica
# status was real and not a stale report. It is also **not available serverless**: called
# without the deployment suffix it returns "Model not found, inaccessible, and/or not
# deployed", so that one contended H100 was the only route to it. Measured 5/5 at
# 0.3-0.4s, `gpt-oss-120b` needs no deployment at all.
#
# Practical upside: nemotron is serverless and the fastest model measured (489.5 tok/s).
#
# **Both judges are also agent models, and that is safe for the same structural reason**
# spelled out above -- the agent is fixed within a group, so no judge ever compares its
# own output against another model's. The residual calibration risk is worth checking on
# any run's output: group the kept pairs by authoring agent and compare each judge's
# agreement-with-intent on its own family against the rest.
#
# Both seats were assigned by availability and structure, **not by measured skill**. Note
# what that costs: the filter's kept set is the dataset, so a systematic blind spot shared
# by these two models is a systematic blind spot in the benchmark, and no evaluator score
# computed from it would show it.
JUDGES: dict[str, str] = {
    "judge1": f"{FW}/nemotron-lightning-3p5-30b-a3b",
    "judge2": f"{FW}/gpt-oss-120b",
}
JUDGE_ARGS = {"temperature": 0.0, "max_tokens": 10_000, **CALL_CONTROL}

# Spares, in order of preference. Every one of these needs a scale-to-zero deployment
# woken before use, which is exactly why none of them is on the panel.
#
# `gemma-4-26b-a4b-it` is listed last and **was unreachable when dropped** -- kept only so
# the swap is traceable. `gemma-3-4b-it` was a listed spare for a while but has no
# deployment at all, so it was never reachable either.
SPARE_JUDGES: tuple[str, ...] = (
    f"{FW}/qwen3-4b-instruct-2507{DEP}-qwen3-4b-instruct-2507",
    f"{FW}/qwen3-1p7b{DEP}-qwen3-1p7b",
    f"{FW}/gemma-4-26b-a4b-it{DEP}-gemma-4-26b-a4b-it",
)

# `max_steps` is per domain because a run that hits the cap is NEVER GRADED --
# reward_breakdown comes back None and reward 0.0, which reads as a failed task.
# Measured 2026-08-09: airline 10 messages, banking 12, retail 18, telecom 22-52.
#
# banking raised 40 -> 80 on 2026-08-10. `qwen-a3b` searches the knowledge base
# far more than `gpt-oss` did on the calibration runs -- at 40 it hit the cap on 3 of 4 runs and
# produced ungraded trajectories; at 80 the same configuration finishes at ~34
# messages with `user_stop`. The cap has to clear the most search-happy agent in
# the pool, not the one that happened to be measured first.
#
# `split` is passed to tau3's `get_tasks(task_set, task_split_name)`. **telecom must ask
# for `base` explicitly.** Its default is the `full` set -- 2,285 machine-generated
# variants of 114 base scenarios -- so sampling 36 tasks from it would look diverse and
# not be. `base` was the intended split from the start, but this call omitted it until
# 2026-08-11, which means every telecom run before that used `full`.
#
# Task counts on these splits: airline 50, telecom 114, banking 97, retail 114.
DOMAINS: dict[str, dict] = {
    "airline":           {"retrieval": None,        "max_steps": 30,  "split": None},
    "telecom":           {"retrieval": None,        "max_steps": 120, "split": "base"},
    "banking_knowledge": {"retrieval": "bm25_grep", "max_steps": 80,  "split": None},
    "retail":            {"retrieval": None,        "max_steps": 60,  "split": None},
}


def ensure_env() -> None:
    """litellm reads FIREWORKS_AI_API_KEY; .env carries FIREWORKS_API_KEY."""
    from dotenv import load_dotenv

    load_dotenv()
    if "FIREWORKS_API_KEY" in os.environ:
        os.environ.setdefault("FIREWORKS_AI_API_KEY", os.environ["FIREWORKS_API_KEY"])


def warm(models: list[str], budget_s: int = 900, verbose: bool = True) -> list[str]:
    """Wake deployments before any simulation starts. Returns those that did not.

    tau3's `generate()` raises without retrying, and the run config's
    `max_retries` is batch-level, so it never reaches `run_single_task`. One
    `DEPLOYMENT_SCALING_UP` therefore kills an entire simulation on turn one.

    Cold starts can run to ~10 minutes, hence the budget. For a long generation run, pin
    `--min-replica-count 1` instead: scale-to-zero releases the GPU and a contended pool may
    not give it back.

    Readiness is inferred from the call rather than read from the deployment, so a deployment
    that will never start is waited out for the full budget. Reading `firectl get deployment`
    would settle it immediately, but that is a subprocess dependency this module does not have.
    """
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True  # required for parity with tau3's own call path

    failed = []
    for m in dict.fromkeys(models):  # de-duplicated, order preserved
        label = m.split("/")[-1].split("#")[0]
        deadline = time.time() + budget_s
        attempt = 0
        while True:
            attempt += 1
            try:
                litellm.completion(model=m, messages=[{"role": "user", "content": "ok"}],
                                   max_tokens=2_000, temperature=0.0)
                if verbose:
                    extra = f" (after {attempt} attempts)" if attempt > 1 else ""
                    print(f"    {label}: warm{extra}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                txt = str(e)
                scaling = ("DEPLOYMENT_SCALING_UP" in txt or "scaling up" in txt
                           or "ServiceUnavailable" in txt)
                # **A dedicated deployment that is still `CREATING` 404s.** Fireworks does
                # not route the `#deployments/<id>` address until a replica registers, so a
                # deployment provisioning normally returns "Model not found, inaccessible,
                # and/or not deployed" -- indistinguishable, by text, from a typo in the model
                # name. Treated as permanent, which cost a real row: a freshly created
                # `meta-eval-gemma-4-26b-a4b-it` was skipped as NOT READY while
                # `Replica Stats: Initializing Replica Count: 1` said it was three minutes
                # into loading weights, and a 26B on one H100 takes longer than that.
                #
                # Retried only when the address names a deployment (`#`). For a serverless
                # model the same 404 really is permanent -- `deepseek-v4-flash` returns it
                # because the model was removed -- and waiting out the budget on a typo is
                # the failure this branch is careful not to reintroduce.
                if "#" in m and "not found" in txt.lower():
                    scaling = True
                if scaling and time.time() < deadline:
                    if attempt == 1 and verbose:
                        print(f"    {label}: cold, waiting...", flush=True)
                    time.sleep(30)
                    continue
                if verbose:
                    print(f"    {label}: NOT READY -- {type(e).__name__}: {txt[:120]}")
                failed.append(m)
                break
    return failed


def pick_agent(domain: str, task_id: str, criterion: str, seed: int = 42) -> str:
    """Choose the agent for a group: random across groups, fixed within one.

    Derived from the group's identity rather than drawn from a stream, so it is
    reproducible from the stored record and does not depend on how many groups
    ran before. Randomising *within* a group would confound model identity with
    the level being steered, which is the one thing the design isolates.
    """
    rng = random.Random(f"{domain}:{task_id}:{criterion}:{seed}")
    return rng.choice(sorted(AGENTS))


def task_description(task: Any) -> str:
    """tau3's task, as a plain description string for the generator.

    Handed over **verbatim**. An earlier version stripped the user simulator's
    role-play scaffolding ("You are playing the role of a customer...") with a list
    of known phrasings, which is exactly the tau3-specific brittleness the plain-data
    interface exists to avoid: it would do nothing for another framework and would
    silently stop working if tau3 reworded a prompt. The scaffolding is harmless
    noise to the generator -- it never reaches the agent, and it is not ground truth.

    This function is the whole tau3-to-generator boundary. Another framework writes
    its own three lines.
    """
    scenario = getattr(task, "user_scenario", None)
    instructions = getattr(scenario, "instructions", None)
    return "" if instructions is None else str(instructions)


# Parsed task catalogues, keyed by (domain, split). tau3's `get_tasks` re-reads and
# re-validates the whole file on every call, and for telecom the `base` split means validating
# all 2,285 `full` tasks and discarding 2,171 of them. Measured: telecom costs 4.07s on the
# first call and **293ms on every call after**, against 2.7ms for airline.
#
# The pool calls `get_task` up to four times per data pair (once to plan the cell, once for
# the instructions, once per simulated level), so on telecom cells that was ~1.2s of GIL-held
# re-parsing per pair -- taken out of exactly the concurrency the pool exists to buy. At 1000
# cells it is minutes.
_TASK_CACHE: dict[tuple[str, Optional[str]], list] = {}
_TASK_CACHE_LOCK = threading.Lock()


def domain_tasks(domain: str) -> list:
    """Every task in a domain, on the split we sample from (see `DOMAINS`). Cached.

    Returns the **shared** catalogue, so counting and iterating are fine but the `Task`
    objects must not be handed to anything that might mutate them. `get_task` deep-copies for
    exactly that reason.

    A probe tool once sliced this list and handed the shared objects straight to
    `run_single_task`. It was serial, so there was no concurrency hazard, but it is the reason
    this warning sits here rather than in a commit message -- prefer `get_task` in anything
    that runs a simulation.
    """
    split = DOMAINS[domain]["split"]
    key = (domain, split)
    with _TASK_CACHE_LOCK:
        if key not in _TASK_CACHE:
            # This is the first line of the pipeline that needs the fork -- `--dry-run`
            # reaches it too, because planning cells means enumerating tasks. A bare
            # ModuleNotFoundError here is the most likely first experience anyone has of this
            # repo, and it names a package they have never heard of, so it is translated into
            # the command that fixes it.
            try:
                from tau2.run import get_tasks
            except ModuleNotFoundError as e:
                if e.name != "tau2":
                    raise
                raise SystemExit(
                    "tau3-bench is not installed, and the generation pipeline needs it to "
                    "enumerate tasks (--dry-run included).\n"
                    "  git clone https://github.com/sierra-research/tau2-bench.git "
                    "../tau2-bench\n"
                    "  git -C ../tau2-bench checkout "
                    "668d3bcd135c02aa3438f987ef45735b7c163ee3\n"
                    "  bash scripts/setup_tau3.sh    # applies patches/tau3.patch, installs\n"
                    "The evaluator sweep does not need it: see docs/REPRODUCE.md section 3."
                ) from e

            _TASK_CACHE[key] = get_tasks(domain, split)
        return _TASK_CACHE[key]


def get_task(domain: str, task_id: Optional[str] = None, index: int = 0) -> Any:
    """One task. By `task_id` when given, else positional `index`.

    An unknown `task_id` raises rather than falling back to `index`, because a silent
    fallback would generate trajectories for a *different* task than the caller named and
    the pair would look fine.

    **Returns a deep copy**, because the catalogue behind it is now cached and shared. Handing
    the same `Task` object to two concurrent simulations would be a new behaviour whose safety
    depends on tau3 treating it as read-only; a copy keeps the isolation callers had when every
    call re-parsed. It is nearly free next to what it replaces: 0.06ms against a 293ms
    re-parse on telecom.
    """
    tasks = domain_tasks(domain)
    if task_id is None:
        return tasks[index].model_copy(deep=True)
    match = next((t for t in tasks if str(t.id) == str(task_id)), None)
    if match is None:
        raise ValueError(f"task {task_id!r} not in {domain} ({len(tasks)} tasks)")
    return match.model_copy(deep=True)


def run_steered(
    domain: str,
    task: Any,
    criterion: str,
    level: str,
    agent_key: str,
    *,
    seed: int = 42,
    max_steps: Optional[int] = None,
    instruction: Optional[str] = None,
    instruction_tag: str = "",
) -> Trajectory:
    """One steered simulation, converted to our schema.

    `instruction` is the wrapped text to steer with -- normally
    `GeneratedInstructions.wrapped(level)` from `steering.generate`. Omitted, the static hand-written instruction for
    (criterion, level) is used, which is the baseline the generator is measured
    against.
    """
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_single_task

    from ..schema.convert import build_context, trajectory_from_simulation

    cfg = DOMAINS[domain]
    steps = max_steps or cfg["max_steps"]
    agent_model, agent_args = AGENTS[agent_key]

    if instruction is None:
        register_steered_agents()
        instruction = get_criterion(criterion).instruction(level)
        name = agent_name(criterion, level)
    else:
        name = register_instruction(
            instruction, agent_name(criterion, level, instruction_tag or "gen")
        )

    run_cfg = TextRunConfig(
        domain=domain,
        agent=name,
        llm_agent=agent_model,
        llm_args_agent=dict(agent_args),
        user="user_simulator",
        llm_user=USER,
        llm_args_user=dict(USER_ARGS),
        max_steps=steps,
        num_trials=1,
        seed=seed,
        retrieval_config=cfg["retrieval"],
        # Reverted 4 -> 10 (tau3's default) on 2026-08-11. Cutting it to 4 stopped
        # telecom runs burning calls on user-side tools, but the cost was worse than
        # the disease: telecom's median fell from 8 spoken turns to 4, too thin to
        # judge friendliness or clarity on. It also *contaminated measurement* --
        # comparing gpt-oss-20b against gpt-oss-120b on tool-use steering, 2 of the
        # 20b's 3 runs were truncated at the cap while all 3 of the 120b's completed,
        # because the 120b barely errs. Any per-agent comparison at a low cap
        # measures the cap as much as the agent.
        max_errors=10,
    )
    sim = run_single_task(run_cfg, task, seed=seed)

    context = build_context(domain, task, retrieval_config=cfg["retrieval"])
    return trajectory_from_simulation(
        sim,
        context,
        provenance=Provenance(
            domain=domain,
            task_id=str(task.id),
            seed=seed,
            agent_model=agent_model,
            agent_args=dict(agent_args),
            user_model=USER,
            user_args=dict(USER_ARGS),
            max_steps=steps,
            retrieval_config=cfg["retrieval"],
            steering=SteeringRecord(
                criterion_name=criterion, level=level, instruction=instruction
            ),
        ),
        trajectory_id=f"{domain}.{task.id}.{criterion}.{level}."
                      f"{agent_key}.{uuid.uuid4().hex[:8]}",
    )


def run_steered_resilient(
    *args: Any,
    attempts: int = 3,
    backoff_s: float = 30.0,
    on_retry: Any = None,
    **kwargs: Any,
) -> tuple[Trajectory, int]:
    """`run_steered`, but a failed simulation is re-run instead of lost.

    Returns `(trajectory, attempts_used)`.

    **This is the pipeline's last data-loss path.** `CALL_CONTROL` already gives litellm
    `num_retries=2` on every seat, but that retries one HTTP request. When those are
    exhausted the exception leaves `run_single_task` and takes the *whole simulation* with
    it -- every turn already generated is discarded, ~97k tokens for a two-level group. A
    `DEPLOYMENT_SCALING_UP` cold start outlasts two retries comfortably, which is how three
    telecom levels were lost.

    **A retry re-runs the entire conversation; it cannot resume mid-way.** tau3 exposes no
    way to continue a partial simulation, so this trades tokens for a trajectory that
    exists. Worth it at these odds, and the reason the user simulator's replica count
    matters more than this wrapper does: a warm replica stops the error happening at all.

    Re-running re-enters `register_instruction` with the same tag and text, which is a
    no-op by design (`steering/agent.py`) and thread-safe. Without that, the retry would
    fail on tau3's raise-on-duplicate before it reached the simulator.

    `backoff_s` defaults to 30s because that is the observed scale of a Fireworks cold
    start, and retrying inside it just burns another attempt on the same cause. Retrying
    faster than the thing you are waiting for is not a retry.

    `on_retry(attempt, exc, elapsed_s)` fires for **every** failed attempt including the
    last, so a caller can account for the time a lost simulation cost. Without it the most
    expensive stage in the pipeline reported its failures nowhere: the surviving trajectory
    was folded in as though it were the first and only attempt.
    """
    last: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        started = time.time()
        try:
            return run_steered(*args, **kwargs), attempt
        except Exception as exc:  # noqa: BLE001 -- any simulation failure is retryable
            last = exc
            # `on_retry` gets the failed attempt's elapsed time, because that time is the
            # only thing recoverable about a lost simulation. The conversation itself is
            # gone -- the exception took it -- so its tokens are unknowable, but the minutes
            # it burned are real and were previously recorded nowhere at all.
            if on_retry is not None:
                on_retry(attempt, exc, time.time() - started)
            if attempt >= attempts:
                break
            time.sleep(backoff_s)
    raise RuntimeError(
        f"simulation failed after {attempts} attempt(s): "
        f"{type(last).__name__}: {str(last)[:200]}"
    ) from last


def run_group(
    domain: str,
    criterion: str,
    *,
    task_id: Optional[str] = None,
    task_index: int = 0,
    agent_key: Optional[str] = None,
    seed: int = 42,
    max_steps: Optional[int] = None,
    do_warm: bool = True,
    on_progress: Any = None,
) -> dict[str, Trajectory]:
    """Three trajectories for one (domain, task, criterion): bad, ok, good.

    Everything we control is held fixed across the three -- task, agent model,
    user simulator, seed. Only the steering instruction differs. That does not
    make the runs identical apart from the steer (the calibration runs measured 22-52 messages
    from identical inputs); it makes the steered axis the largest *systematic*
    difference rather than the only one.

    A level that raises is not swallowed: three trajectories or an exception,
    because `pairs_from_group` rejects an incomplete group anyway and a
    two-thirds group would quietly skew the gap distribution.
    """
    ensure_env()
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; have {sorted(DOMAINS)}")

    task = get_task(domain, task_id, task_index)
    key = agent_key or pick_agent(domain, str(task.id), criterion, seed)
    if key not in AGENTS:
        raise ValueError(f"unknown agent {key!r}; have {sorted(AGENTS)}")

    if do_warm:
        missing = warm([AGENTS[key][0], USER])
        if missing:
            raise RuntimeError(
                "deployment(s) did not come up: "
                + ", ".join(m.split("/")[-1].split("#")[0] for m in missing)
                + ". tau3 aborts a simulation on the first cold-start error, so "
                "re-run or pin --min-replica-count 1."
            )

    from .. steering.criteria import LEVELS

    out: dict[str, Trajectory] = {}
    for level in LEVELS:
        t0 = time.time()
        out[level] = run_steered(domain, task, criterion, level, key,
                                 seed=seed, max_steps=max_steps)
        if on_progress:
            on_progress(level, out[level], round(time.time() - t0, 1))
    return out
