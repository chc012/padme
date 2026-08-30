"""Canonical data model for trajectory-based preference pairs.

The contract between generation and everything downstream. tau3's own types stop
at this boundary: inside `schema/`, `convert.py` is the only module that imports
tau2, and nothing past this schema knows tau3 exists. (Two modules outside
`schema/` also import it -- see `sources/base.py` for the full boundary.)

Three rules the design turns on:

1. **Structured fields are the source of truth; rendered strings are lossy
   projections of them.** `PairwiseEntry.response_1` is text for a judge or a
   human to read, produced by `render.py`, and it truncates. `trajectory_1` is
   the data. Never compute a statistic from a rendered string.

2. **Nothing is dropped at the boundary.** Every field of a source tau3 message
   that this schema does not model explicitly lands in `extra`, and
   `Turn.to_tau2_dict()` reconstructs the original dict exactly. That is what
   `tests/test_schema.py::test_roundtrip_is_exact` asserts, so mis-modelling a field fails a
   test rather than silently losing data.

3. **Votes are recorded, not applied.** `JudgeVote` stores what each judge said
   including unparseable output, and `correct_response` holds the *generator's
   intent*, not a decided label. Every labelling rule -- unanimity, majority,
   intent-only -- is computable from a stored run, so choosing between them
   costs no regeneration: the labelling rule is decided later, on purpose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Level = Literal["bad", "ok", "good"]
Role = Literal["system", "assistant", "user", "tool"]
Requestor = Literal["user", "assistant"]

# Ordering of performance levels. The index is the only thing that decides which
# side of a pair the generator intended as better.
LEVEL_ORDER: dict[str, int] = {"bad": 0, "ok": 1, "good": 2}

# Control tokens the user simulator emits to end a conversation. They are
# protocol, not speech: counting `###TRANSFER###` as a spoken turn inflated
# airline's length from 3 turns / 94 words to 4 / 95, and on a 3-turn trajectory
# that is a 33% overstatement.
PROTOCOL_MARKERS = frozenset({"###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"})


# --------------------------------------------------------------------------- #
# Shared context: what both sides of a pair had to work with
# --------------------------------------------------------------------------- #

class ToolSpec(BaseModel):
    """A tool that was available -- not necessarily one that was called.

    Needed for the tool-use-relevancy criterion: judging whether a call was
    well chosen requires knowing what else could have been chosen, and judging
    a *missing* call requires knowing it existed.
    """

    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict, description="JSON schema of the arguments")
    available_to: Requestor = "assistant"


class KnowledgeBaseInfo(BaseModel):
    """Presence and size of a retrievable corpus, not the corpus itself.

    Only banking_knowledge has one. The documents that actually surfaced live on
    the tool results that returned them (`ToolResult.retrieved`), because what
    matters for judging is what the agent saw, not what it could have seen.
    """

    document_count: Optional[int] = None
    retrieval_config: Optional[str] = Field(
        default=None, description="e.g. 'bm25_grep'; None when the domain has no KB"
    )


class TaskContext(BaseModel):
    """Everything held fixed across a group of three.

    Carried on each trajectory rather than only on the pair, so a single
    trajectory file is readable on its own -- and so a pair can *verify* both
    sides really shared it (`context_key`). A mismatch means the group was not
    held fixed, which invalidates the comparison.
    """

    domain: str
    task_id: str
    policy: str = Field(description="The domain policy the agent was bound by")
    task_purpose: Optional[str] = None
    user_persona: Optional[str] = None
    user_instructions: str = Field(default="", description="What the customer wanted")
    agent_tools: list[ToolSpec] = Field(default_factory=list)
    user_tools: list[ToolSpec] = Field(default_factory=list)
    required_documents: list[str] = Field(
        default_factory=list,
        description="Gold retrieval set from the task, when the domain has a KB. "
        "Objective, unlike the criteria -- usable as a ground-truth check on "
        "retrieval quality.",
    )
    knowledge_base: Optional[KnowledgeBaseInfo] = None

    def context_key(self) -> str:
        """Stable hash of the parts that must be identical across a group."""
        payload = json.dumps(
            {
                "domain": self.domain,
                "task_id": self.task_id,
                "policy": self.policy,
                "user_instructions": self.user_instructions,
                "user_persona": self.user_persona,
                "agent_tools": sorted(t.name for t in self.agent_tools),
                "user_tools": sorted(t.name for t in self.user_tools),
                "required_documents": sorted(self.required_documents),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #

class ToolCallRecord(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)
    requestor: Requestor = "assistant"
    extra: dict = Field(default_factory=dict)

    def to_tau2_dict(self) -> dict:
        return {
            **self.extra,
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "requestor": self.requestor,
        }


class RetrievedDoc(BaseModel):
    """One document a retrieval tool returned, parsed out of its text result."""

    rank: int
    doc_id: str
    title: Optional[str] = None
    score: Optional[float] = None
    content: str = ""


class ToolResult(BaseModel):
    """One tool result. A turn holds a list of these because tau3 wraps multiple
    results from one assistant turn in a single `MultiToolMessage`."""

    id: str
    requestor: Requestor = "assistant"
    content: Optional[str] = None
    error: bool = False
    retrieved: Optional[list[RetrievedDoc]] = Field(
        default=None,
        description="Parsed documents when this result came from a retrieval "
        "tool. None means 'not a retrieval result'; [] means 'retrieval "
        "returned nothing'. Distinguishing these matters.",
    )
    retrieval_parse_failed: bool = Field(
        default=False,
        description="A retrieval result whose format we could not parse. "
        "Recorded rather than swallowed, so it cannot masquerade as an empty "
        "retrieval -- the same trap as an empty NL_ASSERTION list scoring 1.0.",
    )
    extra: dict = Field(default_factory=dict)

    def to_tau2_dict(self) -> dict:
        return {
            **self.extra,
            "id": self.id,
            "role": "tool",
            "content": self.content,
            "requestor": self.requestor,
            "error": self.error,
        }


class Turn(BaseModel):
    """One message. Named `Turn` for position in the list, not for a spoken turn
    -- see `TrajectoryStats.spoken_turns` for those. Raw message count runs ~3x
    spoken turns because every tool call and result is its own message."""

    index: int = Field(description="Position in the original message list")
    role: Role
    content: Optional[str] = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(
        default_factory=list, description="Populated only when role == 'tool'"
    )
    multi: bool = Field(
        default=False,
        description="True when the source was a MultiToolMessage. Needed to "
        "reconstruct the original shape, since a single result is stored flat.",
    )
    turn_idx: Optional[int] = None
    extra: dict = Field(
        default_factory=dict,
        description="Every source field this schema does not model. Populating "
        "it is what makes round-tripping exact and the fidelity test able to fail.",
    )

    @property
    def is_spoken(self) -> bool:
        """A participant actually said something.

        Tool traffic is not speech, and neither is a bare protocol marker.
        """
        if self.role not in ("assistant", "user"):
            return False
        text = (self.content or "").strip()
        return bool(text) and text not in PROTOCOL_MARKERS

    def to_tau2_dict(self) -> dict:
        """Reconstruct the source tau3 message dict.

        tau3 messages are pydantic models serialised with `model_dump_json()`, so
        every field is always present -- which is why modelled keys are emitted
        unconditionally rather than only when set.

        One normalisation: an empty `tool_calls` list comes back as `None`, which
        is how tau3 itself represents "made no tool calls". The distinction
        carries no meaning, and the round-trip test normalises only this field.
        """
        if self.role == "tool":
            if self.multi:
                return {
                    **self.extra,
                    "role": "tool",
                    "tool_messages": [r.to_tau2_dict() for r in self.tool_results],
                }
            if len(self.tool_results) != 1:
                raise ValueError(
                    f"turn {self.index}: non-multi tool turn holds "
                    f"{len(self.tool_results)} results, expected exactly 1"
                )
            return self.tool_results[0].to_tau2_dict()

        return {
            **self.extra,
            "role": self.role,
            "content": self.content,
            "turn_idx": self.turn_idx,
            "tool_calls": [tc.to_tau2_dict() for tc in self.tool_calls] or None,
        }


# --------------------------------------------------------------------------- #
# Outcome, reward, provenance
# --------------------------------------------------------------------------- #

class RewardRecord(BaseModel):
    """tau3's programmatic reward.

    We use this only to notice steering that accidentally broke the task, and it
    is a *soft* signal: the calibration runs measured identical inputs producing rewards of
    both 0.0 and 1.0, so it is not a stable property of a trajectory. Not a
    number to publish -- retail's is partial under our fork, which drops the LLM
    judge it depended on.
    """

    reward: Optional[float] = None
    reward_basis: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    graded: bool = Field(
        default=True,
        description="False when the run hit max_steps and was never graded. "
        "Then reward is 0.0 and the breakdown empty, which reads exactly like a "
        "failed task but means an unfinished conversation.",
    )
    nl_assertion_present: bool = Field(
        default=False,
        description="Must be False. True means an LLM judge reached the reward "
        "path, which our fork exists to prevent.",
    )

    @property
    def unscored_basis(self) -> list[str]:
        """Basis components the task declared but the breakdown never scored.

        Non-empty means the reward covers less than the task asked for. Retail is
        the live case: its basis lists `NL_ASSERTION`, our fork drops the LLM
        judge that evaluated it, and the breakdown comes back `{DB}` alone -- so
        `reward == 1.0` there means "passed the part we still score". This is why
        retail's reward is not a number to publish.
        """
        return sorted(set(self.reward_basis) - set(self.reward_breakdown))


class Outcome(BaseModel):
    termination_reason: str
    hit_max_steps: bool = False
    duration_s: Optional[float] = None


class SteeringRecord(BaseModel):
    """The intervention that produced this trajectory.

    Stored verbatim because the generator's intent is one of the three panel
    seats, and because it is the only record of *what* we asked for when we come
    to ask whether the model delivered it.
    """

    criterion_name: str
    level: Level
    instruction: str = Field(description="Exact text appended to the agent's system prompt")


class Provenance(BaseModel):
    tau2_commit: Optional[str] = None
    tau2_version: Optional[str] = None
    domain: str = ""
    task_id: str = ""
    seed: Optional[int] = None
    agent_model: str = ""
    agent_args: dict = Field(default_factory=dict)
    user_model: str = ""
    user_args: dict = Field(default_factory=dict)
    max_steps: Optional[int] = None
    retrieval_config: Optional[str] = None
    steering: Optional[SteeringRecord] = None
    generated_at: Optional[str] = None
    simulation_id: Optional[str] = None


class TrajectoryStats(BaseModel):
    """Derived counts. Recomputable from `turns`; stored so that annotation load
    and conversation length are readable without walking the messages."""

    n_messages: int = 0
    spoken_turns: int = Field(default=0, description="Messages where a participant spoke")
    spoken_words: int = 0
    agent_tool_calls: list[str] = Field(default_factory=list)
    user_tool_calls: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(
        default_factory=list, description="Union across all retrieval calls, in first-seen order"
    )
    tool_errors: int = 0


class Trajectory(BaseModel):
    """One simulation, losslessly."""

    schema_version: str = SCHEMA_VERSION
    trajectory_id: str
    context: TaskContext
    provenance: Provenance
    turns: list[Turn] = Field(default_factory=list)
    outcome: Optional[Outcome] = None
    reward: Optional[RewardRecord] = None
    stats: TrajectoryStats = Field(default_factory=TrajectoryStats)

    def to_tau2_messages(self) -> list[dict]:
        return [t.to_tau2_dict() for t in self.turns]


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #

class JudgeVote(BaseModel):
    """One judge's raw output. No filtering decision is encoded here.

    `chose is None` with `parse_failed=True` is the case the existing pipeline
    cannot see: `judge.py`'s `_parse_json` returns `{}` on failure, making a judge
    that produced garbage indistinguishable from one that disagreed. Separating
    them is why this field exists.
    """

    judge_name: str
    model: str
    chose: Optional[Literal[1, 2]] = None
    parse_failed: bool = False
    reasoning: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None

    def agrees_with(self, correct_response: int) -> Optional[bool]:
        """None when there is no usable vote -- not False. A judge that failed to
        answer did not disagree."""
        return None if self.chose is None else self.chose == correct_response


class PairwiseEntry(BaseModel):
    """Two trajectories of the same task, compared on one criterion.

    Two views of the same pair, in one object. The **flat block** comes first --
    `id`, `prompt`, `response_1`, `response_2`, `criterion_*`, `correct_response` -- and is
    the only part any consumer needs: the judge module and `build_eval_dataset.py` read
    exactly those keys, and an external tool can too without importing this schema. The
    **structured block** is additive provenance (which trajectory, which steering level,
    which agent) that the flat view has nowhere to put.

    `correct_response` is the generator's *intent* -- which side was steered to the higher
    level -- and not a decided label. That distinction is the whole method: the filter
    cascade scores agreement *with* this field, so treating it as ground truth would make
    the filter measure nothing. A downstream rule may overrule it.
    """

    # --- flat view: compatible with the existing dataset format --------------
    id: str
    criterion_name: str
    criterion_description: str
    positive_hint: str = Field(default="", description="Steering text of the better side")
    negative_hint: str = Field(default="", description="Steering text of the worse side")
    prompt: str = Field(default="", description="Rendered shared context, from render.py")
    response_1: str = Field(default="", description="Rendered trajectory, from render.py")
    response_2: str = Field(default="", description="Rendered trajectory, from render.py")
    correct_response: Literal[1, 2] = 1

    # --- structured: the source of truth ------------------------------------
    schema_version: str = SCHEMA_VERSION
    group_id: str = Field(default="", description="Shared by the 3 pairs from one group of three")
    levels: tuple[Level, Level] = Field(
        default=("bad", "good"), description="Intended level of (response_1, response_2)"
    )
    context: TaskContext
    trajectory_1: Trajectory
    trajectory_2: Trajectory
    judge_votes: list[JudgeVote] = Field(default_factory=list)

    @property
    def gap(self) -> Literal["adjacent", "wide"]:
        """Distance between the two intended levels. Reporting accuracy against
        this is the point of the group-of-three design."""
        a, b = (LEVEL_ORDER[self.levels[0]], LEVEL_ORDER[self.levels[1]])
        return "wide" if abs(a - b) > 1 else "adjacent"

    @property
    def context_consistent(self) -> bool:
        """Both sides really shared the same task, policy and tools.

        `pairs_from_group` refuses to build a pair that would fail this, so on a
        freshly generated pair it is always True. It stays as a property because a pair
        read back from disk was not built by this process: a file assembled by hand, merged
        from two runs, or written by an older version can carry a mismatch, and there is
        otherwise nothing that would notice.

        False means the group was not held fixed, so the comparison is not between two
        performances of one task and the pair should be dropped rather than scored.
        """
        return (
            self.trajectory_1.context.context_key()
            == self.trajectory_2.context.context_key()
            == self.context.context_key()
        )
