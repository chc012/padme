"""Pairwise judging: the filter stage of the pipeline.

`judge_side` is the only entry point -- one judge, one pair, returning **which side it
picked**. `tools/run_pipeline.py` derives agreement-with-intent from that side rather than
asking for a boolean, because a bag of "agreed?" flags cannot answer inter-judge agreement
or position bias afterwards.

Three things in here are the product of specific failures, and each carries its own note at
the point of definition:

- **`_extract_answer`** salvages a verdict from a malformed wrapper. Of 5 parse failures in
  the first 36-call run, 4 contained a usable answer.
- **`_judge_call`** imports call arguments rather than restating them. A local copy once
  omitted `temperature`, so every vote was silently resampled at the provider default.
- **`response_format={"type": "json_object"}` is deliberately absent.** On qwen3-30b-a3b it
  suppresses the stop token, taking one call from 472 to 2,996 tokens.

There is no batch helper here on purpose. The pipeline already calls one judge on one pair
inside its own worker, so a second concurrency layer in this module would be a thread pool
nested inside a thread pool, with two independent notions of how many calls are in flight.
"""

import json
import os
import random
import re

import litellm
from dotenv import load_dotenv

load_dotenv()

litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "ERROR"
os.environ.pop("ANTHROPIC_BASE_URL", None)


def _default_judges() -> dict[str, str]:
    """The panel, from its single source of truth in `sources/tau2_source.py`.

    Imported inside the function, not at module scope: `tau2_source` pulls in the tau3
    registry, and nothing here needs it unless a caller omits the models.
    """
    from .sources.tau2_source import JUDGES
    return JUDGES


def _usage_api():
    """`(UsageMeter, timed_call)`, imported lazily to keep this module importable on its own."""
    from .usage import UsageMeter, timed_call
    return UsageMeter, timed_call


def _parse_json(content: str) -> dict:
    text = (content or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            newline = inner.find("\n")
            text = inner[newline:].strip() if newline != -1 else inner.strip()
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return {}

# Judges emit a correct verdict in a wrapper that `json.loads` rejects, and they do it
# often: of 5 parse failures in the first 36-call run, **4 contained a usable answer**.
# Two shapes, both seen in real output:
#
#   1. JSON whose string values contain literal newlines -- illegal JSON, but the trailing
#      `"answer": "B"` is intact and unambiguous.
#   2. Markdown that ignores the JSON instruction entirely and ends `**Answer:** A`.
#
# The remaining failure was a 41,006-character ramble with no verdict at all, which hit the
# token cap. That one is a real failure and stays one -- recovering an answer that was never
# stated would be inventing data.
#
# Both patterns require the literal word "answer". A looser rule matching `Response ([AB])`
# would fire on "Response A fails the criterion" and read the verdict backwards.
_ANSWER_PATTERNS = (
    re.compile(r'"answer"\s*:\s*"?([AB])\b', re.IGNORECASE),
    re.compile(r'\banswer\b\s*\**\s*[:=]\s*\**\s*"?([AB])\b', re.IGNORECASE),
)


def _extract_answer(raw: str) -> str | None:
    """"A" or "B" from a judge's reply, or None if it never stated one.

    Tries strict JSON first, then the salvage patterns. The **last** match wins: a verdict
    comes after the reasoning, and the reasoning may quote the word "answer" on its way
    there.
    """
    parsed = _parse_json(raw)
    answer = (parsed.get("answer") or "").strip().upper()
    if answer in ("A", "B"):
        return answer
    for pattern in _ANSWER_PATTERNS:
        found = pattern.findall(raw or "")
        if found:
            return found[-1].upper()
    return None


PAIRWISE_PROMPT = """\
You are evaluating two AI assistant responses on a specific criterion.

Criterion: {criterion_name}
Definition: {criterion_description}

User prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response better fulfills the criterion?

Return JSON with exactly two keys:
- "reasoning": at most 3 sentences comparing the two responses on this criterion
- "answer": "A" or "B"

State the answer last, and state it even if the choice is close.
"""


def _judge_call() -> dict:
    """Call arguments for a judge, from the roster's `JUDGE_ARGS`.

    **Imported rather than restated.** A local copy of these arguments is what caused the
    worst bug in this file: it omitted `temperature`, so every judge call ran at the
    provider default (~1.0) while `JUDGE_ARGS` in the roster said 0.0 and only a test ever
    read it. Votes were therefore resampled on every run -- judge1 scored 80.0% and then
    72.2% on the same 36 pairs -- and no output said so.

    """
    from .sources.tau2_source import JUDGE_ARGS
    return dict(JUDGE_ARGS)


def judge_side(
    entry: dict,
    model: str,
    judge_name: str,
    api_key: str | None = None,
    api_base: str | None = None,
    max_retries: int = 2,
    meter=None,
    item_id: str = "",
) -> dict:
    """One pairwise judge call. Returns **which side the judge picked**, not whether it
    was right.

    The side is the primitive and agreement-with-intent is derived from it, because a
    boolean "agreed with `correct_response`" throws away the information every downstream
    analysis needs: you cannot recover a panel's majority *side*, measure inter-judge
    agreement, or detect position bias from a bag of agreement flags. The pipeline derives
    its own boolean at the call site, one line after asking.

    `a_was` records which response was shown in slot A. The A/B order is randomized per
    (entry, judge) so a judge that always answers "A" cannot look accurate -- keeping the
    assignment makes that bias measurable rather than merely defended against.

    `meter` is an optional `UsageMeter`. Each retry turn is recorded under its own
    `attempt`, because a retry is not a repeat: it re-sends the prompt plus the assistant's
    reply, so its prompt tokens strictly exceed the first attempt's.

    `item_id` overrides the accounting key, which defaults to the pair's uuid. A caller
    assembling a per-datapoint chain across stages needs one key for the whole chain, and the
    pair uuid does not exist yet when the instruction for that same datapoint is generated.
    """
    _, _timed = _usage_api()
    rng = random.Random(f"{entry['id']}:{judge_name}")
    a_is_response_1 = rng.random() > 0.5

    response_a = entry["response_1"] if a_is_response_1 else entry["response_2"]
    response_b = entry["response_2"] if a_is_response_1 else entry["response_1"]

    prompt_text = PAIRWISE_PROMPT.format(
        criterion_name=entry["criterion_name"],
        criterion_description=entry["criterion_description"],
        prompt=entry["prompt"],
        response_a=response_a,
        response_b=response_b,
    )

    resp = _timed(
        lambda: litellm.completion(
            model=model,
            api_base=api_base,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt_text}],
            **_judge_call(),
        ),
        meter, model=model, role="judge", label=judge_name,
        item_id=item_id or str(entry.get("id", "")), attempt=1,
    )
    raw = resp.choices[0].message.content
    output = _parse_json(raw)
    answer = _extract_answer(raw)

    # Retry when the judge never stated a verdict. `num_retries` does not cover this:
    # litellm retries transport failures, and this reply arrived as a healthy HTTP 200.
    #
    # The retry **must add something**. At temperature 0 re-sending the same prompt returns
    # the same text, so the model is shown its own reply and asked to commit -- a correction
    # turn, not a blind re-ask. Nothing about which side to pick is suggested, so this
    # cannot bias *which* answer comes back, only whether one arrives.
    #
    # It often will not help, and that is fine: the pair that provoked this is one where
    # both humans (confidence 100) and both judges chose the side the generated label calls
    # wrong. A judge refusing to commit on a mislabelled pair is signal, so exhausted
    # retries still raise.
    attempts, messages = 1, [{"role": "user", "content": prompt_text}]
    while answer is None and attempts <= max_retries:
        attempts += 1
        messages = messages + [
            {"role": "assistant", "content": raw or ""},
            {"role": "user", "content":
                "You did not state an answer. Reply with ONLY this and nothing else:\n"
                '{"reasoning": "<one sentence>", "answer": "A"}\n'
                "using A or B, whichever better fulfills the criterion."},
        ]
        resp = _timed(
            lambda: litellm.completion(model=model, api_base=api_base, api_key=api_key,
                                       messages=messages, **_judge_call()),
            meter, model=model, role="judge", label=judge_name,
            item_id=item_id or str(entry.get("id", "")), attempt=attempts,
        )
        raw = resp.choices[0].message.content
        output = _parse_json(raw) or output
        answer = _extract_answer(raw)

    if answer is None:
        raise ValueError(
            f"Judge stated no answer after {attempts} attempt(s): {raw!r}")

    # Map A/B answer back to response_1/response_2 slot
    if answer == "A":
        chosen = 1 if a_is_response_1 else 2
    else:
        chosen = 2 if a_is_response_1 else 1

    return {
        "side": chosen,
        "answer": answer,
        "a_was": 1 if a_is_response_1 else 2,
        "reasoning": (output.get("reasoning") or "").strip(),
        # >1 means the judge had to be asked to commit. Recorded so a vote that needed
        # prompting is distinguishable from one given freely.
        "attempts": attempts,
    }
