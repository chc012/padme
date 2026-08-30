"""Generate the three steering instructions for one (task, metric).

**Per data point, not per metric.** The previous generator took
only a criterion name and description, so its hints were generic: "one or two
redundant lookups" means something on a task where a careful agent makes 3 tool
calls and nothing on one where it makes 20. What "good", "ok" and "bad" look like
is a property of the metric *on this task*.

**Framework-agnostic by construction.** The inputs are four pieces of plain data --
metric name, metric description, a task description, a list of tool names. No
policy document, no framework objects, nothing only tau3 can supply. Producing a
task description is the caller's job; `sources/tau2_source.py` does it for tau3 by
handing over the task's user instructions verbatim, and another framework writes
three lines of its own.

**Leakage is prevented by what we withhold, not by checking the output.** The
generator is not shown ground truth -- not the expected resolution, not the
documents a correct handling retrieves -- so it cannot pass on an answer it does not
have. Earlier it was shown them and could not resist ("conclude after stating the
refusal"), and a reviewer built to catch that proved unreliable, catching the same
leak twice and passing it a third time.

Task *particulars* are a second kind of leak, handled in the prompt rather than
structurally: the generator needs the task description to write anything specific,
but must not copy names or identifiers into the agent's system prompt, since the
agent is supposed to learn those from the conversation.

**No validation layer.** See `generate_instructions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..util import parse_json
from .criteria import LEVELS, STEERING_WRAPPER, Criterion

# Switched from `qwen3-30b-a3b` on 2026-08-11. That model was picked when it was the
# strongest thing we had deployed; two things changed since. It is **serverless now** --
# no ~10 minute cold start before a generation run, which matters because the generator
# is called once per (domain, criterion, task) and a 36-cell run would otherwise start
# by waking a scale-to-zero deployment. And `gpt-oss-120b` measured better on the one
# thing this seat needs: it is the only agent with near-zero tool errors on telecom, and
# a3b was dropped as an agent for ignoring which tools were its own.
#
# Still inside an *active*-parameter budget at ~5B active, though not a total-parameter
# one. If the paper needs the generator to be small on total params too, `gpt-oss-20b`
# is the fallback and is also serverless.
#
# Measured caveat that applies to any small generator: at 3.3B active, a3b could not
# hold a five-part review rubric reliably. The rubric is a single call with no rubric
# now, so this matters less than it did.
DEFAULT_GENERATOR = "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"

META_PROMPT = """\
You are designing an experiment about how well automatic evaluators judge the
quality of AI agent behaviour.

For ONE specific task and ONE specific quality metric, write three system-prompt
instructions that steer an agent to perform at three levels on that metric: BAD, OK
and GOOD.

## The metric

Name: {criterion_name}
What it means: {criterion_description}

## The task

What the user is trying to get done:
{task_description}

Tools the agent can use: {tool_names}

## Rules for the instructions you write

1. BE SPECIFIC TO THE KIND OF WORK THIS TASK INVOLVES -- NOT TO ITS PARTICULARS.
   Generic advice that would fit any task is one failure mode. Naming the task's
   details is the other, and it is worse. Do refer to which tools matter here, what
   kind of information the user needs, and the shape this interaction will take. Do
   NOT restate the specifics you were given: no names, no user or account or
   booking identifiers, no retelling of this person's circumstances. The agent
   learns those from the conversation itself. Putting them in its system prompt
   hands it information out of band, and makes your three instructions differ by
   how much detail they carry rather than by the metric.
   Wrong: "You are assisting Emma Kim (user ID emma_kim_9957) with cancelling
   reservation EHGLP3, and she was out of town relying on prior insurance."
   Right: "When the customer explains why they are asking, acknowledge the
   circumstances they raise before you get to the outcome."
2. MAKE THE THREE LEVELS SEPARATE. Someone reading the three resulting transcripts
   should be able to rank them on this metric without being told which is which. If
   two of your instructions would produce similar behaviour, rewrite them.
3. FOR BAD, AIM AT THE OPPOSITE OF THE METRIC -- do not merely withhold good
   behaviour. Work out what the active opposite of this metric is, on this task,
   and instruct the agent to pursue it. If the metric is friendliness, BAD is not
   neutral or terse: it is cold, dismissive, impatient. If the metric is clarity,
   BAD is actively confusing, not just unpolished. If the metric is relevant tool
   use, BAD uses tools in ways that actively do not serve the request. Name the
   opposite explicitly and tell the agent to do it.
4. DESCRIBE OK IN ITS OWN TERMS, not as "somewhat good" or "slightly bad", and not
   as a milder version of BAD. OK is what unremarkable, adequate, uncared-for work
   looks like on this metric -- the agent is neither trying nor failing.
5. YOU DO NOT KNOW THE CORRECT OUTCOME, AND MUST NOT ASSUME ONE. You have not been
   told how this task should be resolved, and you may not guess. Do not write
   anything that presupposes an answer -- not "state the refusal", not "confirm it
   is not permitted". Where the outcome matters to how the agent should behave,
   phrase it conditionally: "if it turns out you cannot do what they are asking,
   ...".
6. STEER CONDUCT, NEVER DICTATE WORDS. Do not write the agent's reply for it, do
   not include a sentence for it to say, and never use "respond with", "say
   exactly" or a quoted message. You are shaping HOW it works, not WHAT it
   concludes.
7. WRITE ONLY THE INSTRUCTION BODY. Do not tell the agent to keep the instruction
   secret or that the objective overrides its defaults -- that is added separately.
   Two to five sentences each, addressed to the agent as "you".

Nothing is off limits in how badly the BAD level may perform on this metric. If
performing the opposite of this metric means leaving work out, being unhelpful,
being wrong, or failing the task outright, say so plainly. The experiment needs a
genuinely poor example, and a BAD level that is merely lukewarm is the most common
way this goes wrong.

Return JSON with exactly these keys:
- "reasoning": what good, adequate and poor {criterion_name} concretely look like
  for this task, and how you will make the three levels separate
- "bad": the instruction body for the BAD level
- "ok": the instruction body for the OK level
- "good": the instruction body for the GOOD level
"""

# Fireworks JSON mode suppresses the stop token on this model: with
# response_format={"type": "json_object"} the model never emits EOS and runs to
# max_tokens every call (2,996 of 3,000; 9,910 of 10,000). Without it, the same
# prompt finishes on its own in 472 tokens -- 6x fewer, and 20x at our old cap.
# It was also not constraining anything: with JSON mode on, the output did NOT
# start with "{" and DID contain markdown fences. Pure cost, plus it caused the
# truncated-JSON parse failures. The prompt asks for JSON and util.parse_json
# handles fences, so nothing is lost by dropping it.
_JSON_STOP = ["\n```", "Final Answer"]

@dataclass
class GeneratedInstructions:
    """The generator's output for one group."""

    criterion_name: str
    instructions: dict[str, str]  # level -> instruction body, unwrapped
    generator_model: str = ""
    domain: str = ""
    task_id: str = ""
    reasoning: str = ""
    raw_output: str = ""

    def wrapped(self, level: str) -> str:
        """The exact text appended to the agent's system prompt."""
        return STEERING_WRAPPER.format(instruction=self.instructions[level].strip())


def build_meta_prompt(
    criterion: Criterion,
    task_description: str,
    tool_names: Sequence[str] = (),
) -> str:
    """The generator's whole input. Four pieces of plain data, nothing more.

    Each omission is deliberate:

    - **No ground truth.** It was the leak source; see the module docstring.
    - **No policy document.** Not something an arbitrary agent can supply, and this
      has to work outside tau3.
    - **No agent model.** Irrelevant to what good conduct on a metric looks like.
    """
    return META_PROMPT.format(
        criterion_name=criterion.name,
        criterion_description=criterion.description,
        task_description=(task_description or "(not recorded)").strip(),
        tool_names=", ".join(tool_names) or "(none)",
    )


def generate_instructions(
    criterion: Criterion,
    task_description: str,
    tool_names: Sequence[str] = (),
    *,
    generator_model: str = DEFAULT_GENERATOR,
    domain: str = "",
    task_id: str = "",
    max_tokens: int = 4_000,
    temperature: float = 0.3,
    meter=None,
    item_id: str = "",
) -> GeneratedInstructions:
    """One call. Whatever the generator returns is what you get.

    There is no validation layer and no revision loop, deliberately. Earlier
    versions had regex checks, an LLM leak judge and a five-property semantic
    reviewer. Measured over 12 generations: the deterministic checks fired on 2,
    both for the same trivial pattern, and the LLM reviewer never caught a real
    defect while blocking 2 of 3 valid generations and costing ~10k tokens a call.
    The defect that actually mattered -- a `bad` level behaving better than `good`
    -- is invisible to any text check, because it only exists once the
    instructions are run.

    A bad instruction set is cheap to spot by reading and cheap to regenerate.
    Judge the output, not the guard rails.

    `item_id` overrides the accounting key. The default, `"{domain}:{task_id}"`, is **not
    unique per datapoint**: one task carries all three criteria, so three separate
    generations collapse into one row and the per-datapoint roll-up cannot see them apart.
    Worse, it does not match the key stages 3-5 use (the pair uuid), so a per-pair chain
    spanning stages cannot be assembled at all. A caller that knows the datapoint's identity
    should pass it; measured on a 2-pair smoke run, the default produced 5 accounting keys
    for 2 pairs with the two P1 calls merged.
    """
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True

    try:
        from metaeval.usage import timed_call
    except ModuleNotFoundError:
        from src.metaeval.usage import timed_call

    resp = timed_call(
        lambda: litellm.completion(
            model=generator_model,
            messages=[{"role": "user",
                       "content": build_meta_prompt(criterion, task_description, tool_names)}],
            max_tokens=max_tokens, temperature=temperature, stop=_JSON_STOP,
        ),
        meter, model=generator_model, role="instruction-gen",
        label=criterion.name, item_id=item_id or f"{domain}:{task_id}",
    )
    raw = resp.choices[0].message.content or ""
    parsed = parse_json(raw)

    return GeneratedInstructions(
        criterion_name=criterion.name,
        instructions={lvl: str(parsed.get(lvl) or "").strip() for lvl in LEVELS},
        generator_model=generator_model,
        domain=domain,
        task_id=task_id,
        reasoning=str(parsed.get("reasoning") or ""),
        raw_output=raw,
    )
