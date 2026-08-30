"""Render a trajectory and its context as text for a judge or a human to read.

These strings are **lossy projections** of the structured record, produced for
reading. They truncate, and every truncation is marked in the output rather than
silent. Statistics come from `TrajectoryStats`, never from parsing these.

**The LLM evaluator and the human annotator get the same text, from the same
function.** That is a methodological requirement, not tidiness: the study measures
whether an evaluator agrees with a human, and if the two read different documents
then disagreement no longer isolates misalignment.

Graders and the agent under test now see the same *kind* of information: the
policy, the tools, and the conversation. No answer key reaches either. A judge
deployed in the wild has no ground truth, so handing one to ours would measure a
scoring task nobody runs -- see `render_context`.

Two truncations are deliberate defaults:

- **Retrieval results show doc ids, scores and titles, not document bodies.** One
  `KB_search` result measured 12.5 KB across 10 documents; three calls would
  swamp a 3B judge's usable context. What tool-use relevancy actually needs is
  whether the query was sensible and whether what came back was on topic, and
  titles carry that. Raise `doc_content_chars` to include bodies.
- **Other tool results are cut at 600 characters.** Enough to see what came back
  and whether it was an error.

Both sides of a pair are rendered with identical settings, so a truncation can
never advantage one side.
"""

from __future__ import annotations

from .models import PairwiseEntry, TaskContext, Trajectory

SPEAKER = {"assistant": "Agent", "user": "Customer", "system": "System"}


def _clip(text: str, limit: int | None) -> str:
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... truncated, {len(text) - limit} more characters]"


def _one_line(text: str, limit: int) -> str:
    """Collapse whitespace and cut at a word boundary.

    Tool descriptions run to several hundred characters. Cutting mid-word left
    output like `... phone number, ad`, which reads as a typo rather than an
    elision.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip(" ,;:") + " ..."


def render_context(ctx: TaskContext, *, policy_chars: int | None = None) -> str:
    """The shared setup, identical for every trajectory in a group.

    Becomes `PairwiseEntry.prompt`. **This is the single view given to both the
    LLM evaluator and the human annotator.** One function, so the two cannot
    drift: if the human read more than the evaluator, human/evaluator
    disagreement would conflate "the evaluator is misaligned" with "they were
    shown different documents", and that is the one measurement the whole study
    rests on.

    **No ground truth reaches either grader.** This reverses an earlier decision
    (2026-08-09) that graders should get `task_purpose` and `required_documents`
    because a judge cannot otherwise check whether the right documents came back.
    Reversed on 2026-08-11 for external validity: **a judge deployed in the wild
    never has an answer key**, so a benchmark that hands one over measures a
    scoring task nobody actually runs. What the evaluator has to do here is what it
    would have to do in production -- decide from the policy, the tools and the
    conversation alone.

    An audit of the 36 pairs supports it independently: the reference was present in
    27 of 36 and informative in two domains at most. Airline got a real sentence
    ("agent refuses a cancellation that is not allowed"), banking got four document
    ids, telecom got "Test resolution path: Mobile Data/Slow Internet Issues", and
    retail got nothing. So it varied the grading context by domain without being
    reliably useful -- an uncontrolled difference across a third of the dataset.

    What this costs: on banking, tool-use relevancy can no longer be checked against
    the gold retrieval set, so a grader judges whether the queries were sensible and
    whether what came back was on topic. That is the judgment a deployed judge makes.

    Withheld, present in `TaskContext` and shown by `render_task_metadata()`:

    - **`task_purpose`** and **`required_documents`** -- ground truth, per above.
    - **`task_id`** -- removed 2026-08-11. Mostly content-free (`0`, `task_001`), but
      telecom's is
      `[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]`, which names
      the root cause the agent is supposed to diagnose. That is the same leak the
      reference block was removed for, hiding in a different field -- a reminder that
      "is this ground truth?" has to be asked of every field, not just the ones
      labelled as such.
    - **`domain`** -- removed with it. Harmless, and inferable from the policy
      anyway, so it was costing header space without telling a grader anything the
      policy did not.
    - **`user_instructions`** -- the user simulator's system prompt. Not ground
      truth about the agent's task but instructions for playing a role, including
      instructions to withhold ("ONLY MENTION THIS if you are asked"). It reports
      on how well the *simulator* followed orders, which is not what we evaluate,
      and the customer's actual request is in the transcript in the form the
      agent received it.
    - **`user_persona`** -- same class; `None` in every task we use.

    **What "fair" requires is parity**, not a favourable input: evaluator and human
    reading the same thing is what makes a disagreement interpretable, because
    without it we cannot separate a misaligned evaluator from an under-informed one.
    The render therefore takes no criterion argument -- withholding something for
    friendliness but not for tool use would hide exactly the cross-criterion
    distortions the benchmark exists to detect.

    Nor can an evaluator's distortion corrupt a pair's label: the label is the
    steering intent validated by humans. The evaluator's vote is the quantity
    being measured, never the quantity defining truth.
    """
    out: list[str] = []

    for label, tools in (
        ("THE AGENT", ctx.agent_tools),
        # The customer calls these in the transcript, so hiding them would leave a
        # visible action unexplained. Anything the transcript shows must be
        # intelligible from here.
        ("THE CUSTOMER, TO RUN THEMSELVES", ctx.user_tools),
    ):
        if not tools:
            continue
        out += ["", f"TOOLS AVAILABLE TO {label} ({len(tools)})"]
        for t in tools:
            desc = _one_line(t.description, 200)
            out.append(f"- {t.name}: {desc}" if desc else f"- {t.name}")

    kb = ctx.knowledge_base
    if kb is not None:
        out += ["", "KNOWLEDGE BASE"]
        detail = [f"retrieval config: {kb.retrieval_config or 'default'}"]
        if kb.document_count is not None:
            detail.append(f"{kb.document_count} documents")
        out.append("; ".join(detail))

    out += ["", "POLICY THE AGENT MUST FOLLOW", _clip(ctx.policy.strip(), policy_chars)]
    # lstrip: every section prepends a blank separator, and with the domain/task-id
    # header gone the first one would open the string with a stray newline.
    return "\n".join(out).lstrip("\n")


def render_task_metadata(ctx: TaskContext) -> str:
    """The fields `render_context` withholds, for our own inspection only.

    A separate function rather than a flag on `render_context`, so no stray
    keyword argument can leak these into a judge prompt or the annotation UI.
    """
    out = ["[NOT SHOWN TO EVALUATOR OR ANNOTATOR]",
           "", f"domain: {ctx.domain}", f"task_id: {ctx.task_id}"]
    if ctx.task_purpose:
        out += ["", f"task_purpose: {ctx.task_purpose}"]
    if ctx.required_documents:
        out += ["", "required_documents (the gold retrieval set): "
                + ", ".join(ctx.required_documents)]
    if ctx.user_persona:
        out += ["", f"user_persona: {ctx.user_persona}"]
    if ctx.user_instructions:
        out += ["", "user_instructions (the user simulator's system prompt):",
                ctx.user_instructions.strip()]
    return "\n".join(out)


def render_trajectory(
    traj: Trajectory,
    *,
    tool_result_chars: int | None = 600,
    doc_content_chars: int | None = 0,
    include_tool_calls: bool = True,
) -> str:
    """One conversation, with tool traffic inline.

    Becomes `PairwiseEntry.response_1` / `response_2`. Tool calls are shown by
    default: two of our three criteria (tool use relevancy, and communication
    clarity where the agent narrates what it is doing) are unjudgeable without
    them.
    """
    lines: list[str] = []
    for turn in traj.turns:
        n = turn.index + 1

        if turn.role == "tool":
            if not include_tool_calls:
                continue
            for res in turn.tool_results:
                who = SPEAKER.get(res.requestor, res.requestor)
                tag = "Tool error" if res.error else "Tool result"
                if res.retrieved is not None:
                    lines.append(
                        f"[{n}] {tag} -> {who}: retrieved {len(res.retrieved)} document(s)"
                    )
                    for doc in res.retrieved:
                        score = f" (score {doc.score:.2f})" if doc.score is not None else ""
                        lines.append(f"       {doc.rank}. {doc.doc_id}{score}")
                        if doc.title:
                            lines.append(f"          {doc.title}")
                        if doc_content_chars:
                            body = _clip(doc.content.strip(), doc_content_chars)
                            lines += [f"          {ln}" for ln in body.splitlines()]
                elif res.retrieval_parse_failed:
                    # Visible, because a parse failure that rendered as "0
                    # documents" would read as a retrieval that found nothing.
                    lines.append(
                        f"[{n}] {tag} -> {who}: [retrieval result could not be parsed]"
                    )
                    lines.append(_clip((res.content or "").strip(), tool_result_chars))
                else:
                    body = _clip((res.content or "").strip(), tool_result_chars)
                    lines.append(f"[{n}] {tag} -> {who}: {body}")
            continue

        speaker = SPEAKER.get(turn.role, turn.role)
        content = (turn.content or "").strip()
        if content:
            lines.append(f"[{n}] {speaker}: {content}")
        for tc in turn.tool_calls:
            if not include_tool_calls:
                continue
            args = ", ".join(f"{k}={v!r}" for k, v in (tc.arguments or {}).items())
            lines.append(f"[{n}] {speaker} calls {tc.name}({args})")
        if not content and not turn.tool_calls:
            # An empty message is a symptom worth seeing (max_tokens exhausted,
            # or a model that returned nothing), not something to skip.
            lines.append(f"[{n}] {speaker}: [empty message]")

    return "\n".join(lines)


def render_entry(entry: PairwiseEntry, **kwargs) -> PairwiseEntry:
    """Fill the flat view from the structured record, in place.

    Both sides get the same kwargs, so no rendering choice can favour one.
    """
    entry.prompt = render_context(entry.context)
    entry.response_1 = render_trajectory(entry.trajectory_1, **kwargs)
    entry.response_2 = render_trajectory(entry.trajectory_2, **kwargs)
    return entry
