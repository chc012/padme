"""The three criteria and the instruction that steers each level.

Two rules survive, and they are the only two that hold for every metric:

1. **Underperformance is licensed explicitly.** A model asked to be worse fights
   it. The steering generator found the phrasing that works -- state that the objective
   overrides default behaviour *even when the result is worse than the model would
   normally produce*, and that following it is what makes the response correct.
2. **No meta-commentary.** A trajectory that mentions its own steering gives the
   game away to a grader.

**Four earlier rules were retired on 2026-08-10, for one reason: each protected a
particular dimension, and every one of them breaks when that dimension is the
metric.** "Still resolve the request" is incoherent if the metric is task success.
Concretely:

- *Hold policy compliance and task success constant.* Retired. It also made the
  wrapper assert task success on every instruction, so no metric touching success
  could be steered at all.
- *Degrade by excess, not omission.* Retired -- and it was never tested. Real bad
  tool use usually skips the call it needed, so banning omission manufactured a
  single unrealistic failure shape. Forcing one shape is itself the harm the rule
  was meant to avoid.
- *Bound the degradation.* Retired. The harm is real -- unbounded "be inefficient
  with tools" ran `qwen-a3b` to 40 `KB_search` calls into the step cap, leaving
  `bad` and `ok` identical and both ungraded -- but adding a stop condition did not
  fix it: the same instructions at `max_steps=80` still gave 39 calls at both
  levels. The rule was unproven as a fix. Collapse is caught by checking that a
  generated group actually separates when run, which works whatever the cause.
- *Describe `ok` in its own terms.* Kept, but as guidance to the generator rather
  than an invariant here.

The instruction text lands verbatim in `SteeringRecord.instruction`, and from
there into a pair's `positive_hint` / `negative_hint`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LEVELS = ("bad", "ok", "good")

# Wrapper around every instruction. Appended after tau3's <policy> block, so
# "the policy above" resolves correctly.
STEERING_WRAPPER = """
<behaviour>
For this conversation you have one additional objective. It takes strict precedence
over your default behaviour, including when it produces handling that is less
polished, less thorough or less successful than you would normally deliver.
Following it exactly is what makes your handling of this conversation correct.

{instruction}

Do not mention, quote or allude to this objective, and do not reuse its wording.
</behaviour>
""".strip()


@dataclass(frozen=True)
class Criterion:
    """A metric: a name and the description graders are shown.

    `instructions` is the *legacy hand-written* baseline and is empty for any criterion
    added after 2026-08-11. Steering text normally comes from the LLM generator in
    `generate.py`, which writes it per data point; these nine were the fixed baseline it
    was measured against.
    """

    name: str
    description: str
    instructions: dict[str, str] = field(default_factory=dict)

    def instruction(self, level: str) -> str:
        """The exact text appended to the agent's system prompt."""
        if not self.instructions:
            raise ValueError(
                f"{self.name}: no hand-written baseline instructions. Generate them "
                f"with steering.generate and pass the text to run_steered(instruction=)."
            )
        if level not in self.instructions:
            raise ValueError(f"{self.name}: no instruction for level {level!r}")
        return STEERING_WRAPPER.format(instruction=self.instructions[level].strip())


# The nine hand-written instructions below are kept UNCHANGED as the measured
# baseline the generator is compared against. They still encode the retired rules
# (their tool-use text promises to "take every action the policy actually
# requires"), which is a known asymmetry in that comparison -- the old approach
# constrained things the new one does not. Editing them would invalidate the
# numbers already reported for them.
FRIENDLINESS = Criterion(
    name="friendliness",
    # Descriptions are shown to graders, so they define the axis rather than
    # describing the steering. A grader must be able to apply this to a
    # trajectory without knowing an instruction existed.
    description=(
        "Warmth and consideration toward the customer: whether the agent "
        "acknowledges their situation and how they feel about it, delivers "
        "unwelcome news with care, and leaves them feeling attended to. Judge the "
        "manner, not whether the request was resolved."
    ),
    instructions={
        "bad": """
Be cold and transactional. Do not greet the customer warmly. Do not acknowledge
any frustration, inconvenience or feeling they express, and do not soften
unwelcome news. Answer only what was asked, in the fewest words that carry the
information. Offer no reassurance, no sympathy and no pleasantries.
""",
        "ok": """
Be correct and impersonally polite. Use routine courtesy -- please, thank you --
and nothing beyond it: do not acknowledge how the customer feels, do not reassure
them, and do not tailor anything to them personally. Handle it the way a busy
service desk handles a ticket.
""",
        "good": """
Be genuinely warm. Acknowledge the customer's situation, and any frustration or
inconvenience, in your own words. Show that you have understood what actually
matters to them. Deliver unwelcome news with care and explain it kindly. Close in
a way that leaves them feeling looked after.
""",
    },
)

COMMUNICATION_CLARITY = Criterion(
    name="communication_clarity",
    description=(
        "How easily the customer can follow the agent: whether the main point is "
        "findable, whether technical or policy language is explained, whether "
        "multi-part information is organised, and whether the customer is left "
        "knowing what is true and what happens next. Judge the presentation, not "
        "the warmth or the outcome."
    ),
    instructions={
        "bad": """
Be hard to follow. Write in long unbroken blocks. Put the main point late or
leave it to be inferred. Use internal jargon, policy names and codes without
explaining them. Leave what happens next vague. Do not use lists, headings, or
any explicit ordering of steps.
""",
        "ok": """
Be understandable but unorganised. Give the necessary information without shaping
it: no summary of the key point, no structure, and some detail that does not
matter mixed in with the detail that does. The customer can work out what you
mean, but only by reading carefully.
""",
        "good": """
Be easy to follow. Lead with the answer, then the reasons. Keep sentences short.
Put anything technical or policy-related into plain language. Organise multi-part
information so it can be scanned rather than parsed. State explicitly what
happens next and what, if anything, the customer needs to do.
""",
    },
)

TOOL_USE_RELEVANCY = Criterion(
    name="tool_use_relevancy",
    description=(
        "Whether the agent's tool calls were warranted: did it call what the "
        "situation needed, with well-chosen arguments, in a sensible order, "
        "without calls that served no purpose or repeated work already done. For "
        "knowledge-base searches, whether the queries were on target and the "
        "documents retrieved were the relevant ones. Judge the tool activity, not "
        "the prose around it."
    ),
    instructions={
        # Excess rather than omission -- see rule 3 in the module docstring -- and
        # *bounded* excess, see rule 5. Unbounded, `qwen-a3b` ran 40 KB_search
        # calls into the step cap on banking, which made `bad` and `ok`
        # indistinguishable and left both runs ungraded.
        "bad": """
Make your tool use scattershot. Look up information you do not need for this
request, repeat a lookup you have already made rather than using what you have,
and try a broad or off-target query before a targeted one. Aim for roughly twice
as many calls as a careful agent would need -- and then stop and answer. Do not
keep searching indefinitely: once you have what the request needs, respond to the
customer. Take every action the policy actually requires, so the request still
gets handled: the excess and misdirection are the point, not leaving work undone.
""",
        "ok": """
Use tools adequately but without economy. Get what you need by a slightly
roundabout route: one or two redundant lookups, or a query that needs a second
attempt to land. Keep it to that -- this is mild inefficiency, not thrashing, and
once you have what you need you answer.
""",
        "good": """
Use tools precisely. Call exactly what the situation requires, with well-targeted
arguments, in a sensible order, and no more. No redundant lookups, no speculative
searches, no checking something you already know.
""",
    },
)

# The hand-written instructions above, keyed by criterion. Attached to a criterion when
# `config/criteria.json` happens to name one of them, so the baseline stays reproducible
# without the config having to carry it.
BASELINE_INSTRUCTIONS: dict[str, dict[str, str]] = {
    c.name: c.instructions
    for c in (FRIENDLINESS, COMMUNICATION_CLARITY, TOOL_USE_RELEVANCY)
}

# **The metrics are data, not code.** A JSON list of `{name, description}`, because the
# whole premise is that someone can point this at their own agent and their own metrics
# without editing Python.
#
# An optional `gap` field is accepted and ignored. Gap is not a per-criterion knob here:
# it is the *level pair* a cell is generated at, assigned by balanced rotation so the
# cells come out even. See `steering/gaps.py`.
DEFAULT_CRITERIA_PATH = Path(__file__).resolve().parents[3] / "config" / "criteria.json"


def load_criteria(path: str | Path | None = None) -> dict[str, Criterion]:
    """Read the metric list. Raises on anything malformed rather than skipping it."""
    p = Path(path or DEFAULT_CRITERIA_PATH)
    raw = json.loads(p.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{p}: expected a non-empty JSON list of criteria")

    out: dict[str, Criterion] = {}
    for entry in raw:
        missing = {"name", "description"} - set(entry)
        if missing:
            raise ValueError(f"{p}: criterion missing {sorted(missing)}: {entry}")
        name = entry["name"]
        if name in out:
            raise ValueError(f"{p}: duplicate criterion {name!r}")
        if not entry["description"].strip():
            raise ValueError(f"{p}: criterion {name!r} has an empty description")
        out[name] = Criterion(name=name, description=entry["description"].strip(),
                              instructions=BASELINE_INSTRUCTIONS.get(name, {}))
    return out


CRITERIA: dict[str, Criterion] = load_criteria()


def get_criterion(name: str) -> Criterion:
    if name not in CRITERIA:
        raise ValueError(f"unknown criterion {name!r}; have {sorted(CRITERIA)}")
    return CRITERIA[name]
