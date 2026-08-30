"""tau3 -> canonical schema. One of the three modules in `metaeval` that imports tau2,
and the only one that touches its data types (`sources/tau2_source.py` runs the
simulations, `steering/agent.py` registers the steered agent). See `sources/base.py`.

What tau3 does not put in the simulation dump, and this module has to fetch:
the domain policy (present in our fork's dump but re-read for older files), the
tools that were *available*, the task's persona, instructions and
`required_documents`. Those come from the environment and the task, so building a
context constructs the environment -- local work, no LLM calls.

Conversion belongs immediately after generation, while the domain and retrieval
config that produced a run are still known. A saved trajectory then needs
nothing from tau3 ever again.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

from .models import (
    SCHEMA_VERSION,
    KnowledgeBaseInfo,
    Outcome,
    Provenance,
    RetrievedDoc,
    RewardRecord,
    TaskContext,
    ToolCallRecord,
    ToolResult,
    ToolSpec,
    Trajectory,
    TrajectoryStats,
    Turn,
)

# Tool names that return knowledge-base documents. Lowercased for comparison.
# `grep` is here because banking_knowledge exposes it as a retrieval tool
# alongside KB_search.
RETRIEVAL_TOOLS = {"kb_search", "grep", "search_kb", "kb_grep"}

# Fields this schema models explicitly. Everything else on a source message goes
# to `Turn.extra` / `ToolResult.extra`, which is what makes round-tripping exact.
_PARTICIPANT_MODELED = {"role", "content", "tool_calls", "turn_idx"}
_TOOL_MODELED = {"id", "role", "content", "requestor", "error"}
_TOOLCALL_MODELED = {"id", "name", "arguments", "requestor"}

_DOC_BLOCK = re.compile(r"^[ \t]*(\d+)\.[ \t]*(.*)$", re.M)


# --------------------------------------------------------------------------- #
# Retrieval results
# --------------------------------------------------------------------------- #

def parse_retrieved_docs(content: Optional[str]) -> tuple[list[RetrievedDoc], bool]:
    """Parse a retrieval tool's text result into documents.

    Returns `(docs, parse_failed)`. tau3 returns retrieval results as formatted
    text, not JSON:

        1. Gold Rewards Card: Overview
           ID: doc_credit_cards_gold_rewards_card_001
           Score: 20.2385
           Content: ## Eligibility ...

    `parse_failed` separates "retrieval found nothing" from "we could not read
    the result". Collapsing those would let a parser bug look like an agent that
    searched badly -- the same class of silent-zero error the LLM-judge fork
    exists to prevent.
    """
    text = (content or "").strip()
    if not text:
        return [], False

    starts = list(_DOC_BLOCK.finditer(text))
    docs: list[RetrievedDoc] = []
    for i, m in enumerate(starts):
        block = text[m.end(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        doc_id = re.search(r"^[ \t]*ID:[ \t]*(\S+)", block, re.M)
        if not doc_id:
            continue  # a numbered line inside a document body, not a new document
        score = re.search(r"^[ \t]*Score:[ \t]*([-\d.eE+]+)", block, re.M)
        body = re.search(r"^[ \t]*Content:[ \t]*(.*)", block, re.M | re.S)
        try:
            score_val = float(score.group(1)) if score else None
        except ValueError:
            score_val = None
        docs.append(
            RetrievedDoc(
                rank=int(m.group(1)),
                doc_id=doc_id.group(1),
                title=(m.group(2).strip() or None),
                score=score_val,
                content=(body.group(1).strip() if body else ""),
            )
        )

    if docs:
        return docs, False
    # Nothing parsed. A genuinely empty result is short and mentions no ids; a
    # long one, or one containing ids we failed to read, is a parse failure.
    looks_empty = len(text) < 200 and "ID:" not in text
    return [], not looks_empty


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #

def _tool_specs(tools: Any, available_to: str) -> list[ToolSpec]:
    specs = []
    for t in tools or []:
        fn = (t.openai_schema or {}).get("function", {})
        specs.append(
            ToolSpec(
                name=getattr(t, "name", fn.get("name", "")),
                description=fn.get("description", "") or "",
                parameters=fn.get("parameters", {}) or {},
                available_to=available_to,  # type: ignore[arg-type]
            )
        )
    return specs


def _env_kwargs(domain: str, task: Any, retrieval_config: Optional[str]) -> dict:
    """Mirror of `tau2.runner.build._build_env_kwargs` (build.py:337).

    Replicated rather than imported: it takes a RunConfig, and a private helper
    is not a stable seam across upstream merges. If the environment ever comes
    out different from the one a simulation ran against, this is the first place
    to look.
    """
    kwargs: dict = {}
    if retrieval_config is not None:
        kwargs["retrieval_variant"] = retrieval_config
        kwargs["task"] = task
    if domain == "banking_knowledge":
        allow = set()
        crit = getattr(task, "evaluation_criteria", None)
        for action in (getattr(crit, "actions", None) or []):
            if action.name == "call_discoverable_agent_tool":
                name = (action.arguments or {}).get("agent_tool_name")
                if name:
                    allow.add(name)
        kwargs["read_log_allowlist"] = allow
    return kwargs


def build_context(
    domain: str,
    task: Any,
    *,
    retrieval_config: Optional[str] = None,
    environment: Any = None,
    policy: Optional[str] = None,
) -> TaskContext:
    """Assemble the shared context for a group.

    `policy` overrides the environment's, for reading back a simulation saved
    with its own policy text -- so what we store is what the agent actually saw.
    """
    from tau2.runner.build import build_environment

    if environment is None:
        environment = build_environment(
            domain, env_kwargs=_env_kwargs(domain, task, retrieval_config)
        )

    scenario = getattr(task, "user_scenario", None)
    instructions = getattr(scenario, "instructions", None)
    desc = getattr(task, "description", None)
    purpose = getattr(desc, "purpose", None) if desc is not None else None

    kb = (
        KnowledgeBaseInfo(retrieval_config=retrieval_config)
        if retrieval_config is not None
        else None
    )

    # Domains without user tools (airline, retail) raise rather than return []
    # from get_user_tools. tau3 handles this with a bare `except Exception`
    # (build.py:158); testing the condition instead keeps real errors visible.
    user_tools: list[ToolSpec] = []
    if getattr(environment, "user_tools", None) is not None:
        user_tools = _tool_specs(
            environment.get_user_tools(include=getattr(task, "user_tools", None)), "user"
        )

    return TaskContext(
        domain=domain,
        task_id=str(task.id),
        policy=policy if policy is not None else environment.get_policy(),
        task_purpose=purpose,
        user_persona=getattr(scenario, "persona", None),
        # StructuredUserInstructions defines __str__; plain-string tasks pass
        # through unchanged. Both shapes occur across our 4 domains.
        user_instructions="" if instructions is None else str(instructions),
        agent_tools=_tool_specs(environment.get_tools(), "assistant"),
        user_tools=user_tools,
        required_documents=list(getattr(task, "required_documents", None) or []),
        knowledge_base=kb,
    )


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #

def _as_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    return json.loads(obj.model_dump_json())


def _tool_result(msg: dict, tool_names: dict[str, str]) -> ToolResult:
    """One tool result, with retrieval parsed when the call that produced it was
    a retrieval tool. Keyed off the tool *name*, looked up by call id -- guessing
    from the content shape would misclassify."""
    call_id = msg.get("id") or ""
    name = tool_names.get(call_id, "").lower()
    retrieved: Optional[list[RetrievedDoc]] = None
    parse_failed = False
    if name in RETRIEVAL_TOOLS and not msg.get("error"):
        retrieved, parse_failed = parse_retrieved_docs(msg.get("content"))
    return ToolResult(
        id=call_id,
        requestor=msg.get("requestor", "assistant"),
        content=msg.get("content"),
        error=bool(msg.get("error", False)),
        retrieved=retrieved,
        retrieval_parse_failed=parse_failed,
        extra={k: v for k, v in msg.items() if k not in _TOOL_MODELED},
    )


def _turns(messages: list[dict]) -> list[Turn]:
    tool_names: dict[str, str] = {}
    turns: list[Turn] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if role == "tool":
            if "tool_messages" in msg:
                # MultiToolMessage: one assistant turn that called several tools.
                # Not seen in the 8 calibration runs, but the orchestrator emits it
                # whenever a turn carries more than one call
                # (orchestrator.py:342), which these models do.
                results = [_tool_result(m, tool_names) for m in msg["tool_messages"]]
                extra = {k: v for k, v in msg.items() if k not in {"role", "tool_messages"}}
                turns.append(Turn(index=i, role="tool", tool_results=results,
                                  multi=True, extra=extra))
            else:
                turns.append(Turn(index=i, role="tool",
                                  tool_results=[_tool_result(msg, tool_names)]))
            continue

        calls = []
        for tc in (msg.get("tool_calls") or []):
            record = ToolCallRecord(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments") or {},
                requestor=tc.get("requestor", "assistant"),
                extra={k: v for k, v in tc.items() if k not in _TOOLCALL_MODELED},
            )
            tool_names[record.id] = record.name
            calls.append(record)

        turns.append(
            Turn(
                index=i,
                role=role,  # type: ignore[arg-type]
                content=msg.get("content"),
                tool_calls=calls,
                turn_idx=msg.get("turn_idx"),
                extra={k: v for k, v in msg.items() if k not in _PARTICIPANT_MODELED},
            )
        )
    return turns


def compute_stats(turns: list[Turn]) -> TrajectoryStats:
    """Counts that describe a trajectory's size.

    `spoken_turns` is the number anyone would recognise as conversation length;
    `n_messages` runs ~3x that, because every tool call and every tool result is
    its own message. Reporting the raw count as "turns" overstates length ~3x.
    """
    stats = TrajectoryStats(n_messages=len(turns))
    seen_docs: set[str] = set()
    for t in turns:
        if t.is_spoken:
            stats.spoken_turns += 1
            stats.spoken_words += len((t.content or "").split())
        for tc in t.tool_calls:
            target = (
                stats.user_tool_calls if t.role == "user" else stats.agent_tool_calls
            )
            target.append(tc.name)
        for res in t.tool_results:
            if res.error:
                stats.tool_errors += 1
            for doc in (res.retrieved or []):
                if doc.doc_id not in seen_docs:
                    seen_docs.add(doc.doc_id)
                    stats.retrieved_doc_ids.append(doc.doc_id)
    return stats


def _reward(sim: dict, hit_max_steps: bool) -> Optional[RewardRecord]:
    ri = sim.get("reward_info")
    if ri is None:
        return None
    breakdown = ri.get("reward_breakdown") or {}
    keys = {str(k).split(".")[-1].upper() for k in breakdown}
    return RewardRecord(
        reward=ri.get("reward"),
        reward_basis=[str(b).split(".")[-1].upper() for b in (ri.get("reward_basis") or [])],
        reward_breakdown={str(k).split(".")[-1].upper(): float(v)
                          for k, v in breakdown.items()},
        # A capped run is never graded: reward comes back 0.0 with an empty
        # breakdown, which is indistinguishable from a failed task unless the
        # distinction is recorded here.
        graded=bool(breakdown) and not hit_max_steps,
        nl_assertion_present="NL_ASSERTION" in keys,
    )


def tau2_commit() -> Optional[str]:
    try:
        import tau2

        repo = Path(tau2.__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        return None


def trajectory_from_simulation(
    sim: Any,
    context: TaskContext,
    *,
    provenance: Optional[Provenance] = None,
    trajectory_id: Optional[str] = None,
) -> Trajectory:
    """Convert one tau3 `SimulationRun` (object or loaded dict)."""
    data = _as_dict(sim)
    messages = data.get("messages") or []
    turns = _turns(messages)

    reason = str(data.get("termination_reason") or "").split(".")[-1].lower()
    hit_cap = reason == "max_steps"

    prov = provenance or Provenance()
    prov.domain = prov.domain or context.domain
    prov.task_id = prov.task_id or context.task_id
    if prov.seed is None:
        prov.seed = data.get("seed")
    prov.simulation_id = prov.simulation_id or data.get("id")
    prov.generated_at = prov.generated_at or data.get("timestamp")
    prov.tau2_commit = prov.tau2_commit or tau2_commit()

    return Trajectory(
        schema_version=SCHEMA_VERSION,
        trajectory_id=trajectory_id or f"traj-{uuid.uuid4()}",
        context=context,
        provenance=prov,
        turns=turns,
        outcome=Outcome(
            termination_reason=reason,
            hit_max_steps=hit_cap,
            duration_s=data.get("duration"),
        ),
        reward=_reward(data, hit_cap),
        stats=compute_stats(turns),
    )


def normalise_source_message(msg: dict) -> dict:
    """A source message in the form `Turn.to_tau2_dict()` reconstructs it.

    The single normalisation: for participant messages an empty `tool_calls`
    list becomes `None`, which is how tau3 itself represents "made no tool
    calls". Tool-role messages have no `tool_calls` key at all and are returned
    untouched.

    Lives here so every caller compares the same way. When the round-trip test and a
    conversion tool each had their own copy, the tool's version injected `tool_calls` into
    tool messages and declared every conversion lossy while the test passed.
    """
    if msg.get("role") == "tool":
        return dict(msg)
    out = dict(msg)
    if out.get("tool_calls") == []:
        out["tool_calls"] = None
    return out


def roundtrip_is_exact(traj: Trajectory, source_messages: list[dict]) -> bool:
    """Every source message is reconstructible from the converted trajectory."""
    return traj.to_tau2_messages() == [
        normalise_source_message(m) for m in source_messages
    ]


def trajectory_from_file(
    path: str | Path,
    domain: str,
    *,
    retrieval_config: Optional[str] = None,
    provenance: Optional[Provenance] = None,
) -> Trajectory:
    """Convert a saved tau3 simulation dump, rebuilding its task and environment.

    Matches the task by the dump's `task_id`, so the file decides which task is
    loaded rather than a caller-supplied index.
    """
    from tau2.run import get_tasks

    data = json.loads(Path(path).read_text())
    task_id = str(data.get("task_id", ""))
    tasks = get_tasks(domain)
    match = next((t for t in tasks if str(t.id) == task_id), None)
    if match is None:
        raise ValueError(
            f"{path}: task_id {task_id!r} is not in domain {domain!r} "
            f"({len(tasks)} tasks). Wrong domain, or a task set that has changed."
        )

    context = build_context(
        domain, match, retrieval_config=retrieval_config, policy=data.get("policy")
    )
    return trajectory_from_simulation(data, context, provenance=provenance)
