"""Measured token and time accounting for the calls we actually made.

This module records what happened, as opposed to projecting what a planned workload would
cost.

Before it existed, neither production call path kept any of it: `judge_side` and
`_score_model` both read `resp.choices[0].message.content` and dropped `resp.usage` on the
floor. Token counts appeared only in throwaway probe scripts, so no number in any saved
result or paper table was traceable to a token count, and duration existed only as one
wall-clock figure per model.

Four things this gets right, each because getting them wrong hid something real:

- **Failed calls are recorded.** A rate-limited call still costs latency and often still
  bills for the prompt. `nemotron-lightning` threw 20 rate-limit errors in one sweep; with
  only successes counted, the most expensive model in the roster looks like the cheapest.
- **Every attempt is a separate record.** The judge's no-verdict retry re-sends the whole
  prompt *plus* the assistant's reply, so attempt 2 has strictly more prompt tokens than
  attempt 1. Keeping only the final attempt understates spend on exactly the calls that
  went wrong.
- **Reasoning tokens are tracked apart from completion tokens.** They are counted inside
  `completion_tokens`, they are billed, and they are usually invisible: setting
  `response_format={"type": "json_object"}` on qwen3-30b-a3b suppressed the stop token and
  took a call from 472 to 2,996 tokens. A total alone would not have said why.
- **Service time and wall clock are reported separately.** Under N workers the sum of
  per-call latencies exceeds elapsed time, and the ratio *is* the achieved concurrency.
  Reporting one number as "duration" would conflate "we waited this long" with "the
  provider worked this long", which are the two quantities the parallelization work needs
  to tell apart.
"""

from __future__ import annotations

import statistics
import threading
import time
import re
from dataclasses import asdict, dataclass, field

# Inline reasoning, as the qwen3 family emits it. Non-greedy, DOTALL, and tolerant of an
# unclosed tag: a truncated reply (hit max_tokens mid-thought) has an opening tag and no
# closing one, and that is exactly the runaway worth catching.
_THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)


@dataclass(frozen=True)
class CallRecord:
    """One provider call. `ok=False` records the cost of a failure, not its absence."""

    model: str
    role: str = ""            # "evaluator" | "judge" | "scorer" -- pipeline stage
    label: str = ""           # roster label or judge name, for grouping
    item_id: str = ""         # pair id, so cost can be attributed per datapoint
    attempt: int = 1          # 1 = first try; >1 = a retry turn, which is bigger
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Always populated: the provider's figure when it reports one, otherwise derived from
    # the character split. `reasoning_source` says which, so a derived number is never
    # mistaken for a reported one.
    reasoning_tokens: int = 0
    reasoning_source: str = ""   # "reported" | "derived" | "" (no call / error)
    content_chars: int = 0
    reasoning_chars: int = 0
    latency_s: float = 0.0
    ok: bool = True
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def extract_usage(resp) -> tuple[int, int, int | None]:
    """`(prompt, completion, reasoning)` from a litellm response.

    **Reasoning is `None` when the provider does not report it, never 0.** Fireworks
    returns `completion_tokens_details: null` on every model measured -- including
    `gpt-oss-20b` and `nemotron-lightning`, which certainly do reason -- so a 0 here would
    claim "no reasoning happened" on a call that spent 174 completion tokens to emit 12
    characters of content. `content_chars` below is the exact substitute.

    Providers disagree about all three: some omit `usage`, some omit
    `completion_tokens_details`, and a `None` in any field is normal rather than
    exceptional. Every read is therefore defensive -- an accounting helper that raises
    would take down the call it was measuring, which inverts the priority.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0, None
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = None
    if details is not None:
        raw = getattr(details, "reasoning_tokens", None)
        reasoning = int(raw) if raw is not None else None
    return prompt, completion, reasoning


def extract_content(resp) -> tuple[int, int]:
    """`(visible content chars, reasoning chars)` from the message.

    Two shapes in the wild, and both are handled because the roster contains both:

    - a separate `reasoning_content` field (gpt-oss, nemotron-lightning)
    - `<think>...</think>` inline in `content` (the qwen3 family)

    Inline think-tags are moved out of the visible count, so a model that hides its
    reasoning in-band is measured the same way as one that uses a separate field.
    """
    try:
        msg = resp.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return 0, 0
    content = getattr(msg, "content", None) or ""
    reasoning = ""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(msg, attr, None)
        if val:
            reasoning = str(val)
            break
    inline = _THINK_RE.findall(content)
    if inline:
        reasoning += "".join(inline)
        content = _THINK_RE.sub("", content)
    return len(content), len(reasoning)


def derive_reasoning_tokens(completion_tokens: int, content_chars: int,
                            reasoning_chars: int) -> int:
    """Split `completion_tokens` between answer and reasoning by character proportion.

    Fireworks reports `completion_tokens_details: null` on every model measured, so the
    provider's own number is unavailable and the choice is between deriving one and
    reporting nothing. Deriving is clearly better: both texts come from the same model and
    the same tokenizer, so their characters-per-token are close, and the proportion carries
    over to tokens.

    Checked against real calls. gpt-oss-20b: 82 completion tokens, 247 reasoning chars, 12
    content chars -> 78 reasoning / 4 visible, and `{"score": 1}` is indeed ~4-6 tokens.
    nemotron-lightning: 281 tokens, 928 / 12 chars -> 277 / 4.

    **Known bias, small and in a known direction.** Reasoning is prose at ~4 chars/token
    while a JSON verdict is punctuation-dense at ~2-3, so the visible half is slightly
    under-counted and the reasoning share slightly over-stated. It matters at the margin,
    not to the conclusion that a model spent 95% of its output thinking.
    """
    total = content_chars + reasoning_chars
    if completion_tokens <= 0 or total <= 0:
        return 0
    return max(0, min(completion_tokens,
                      round(completion_tokens * reasoning_chars / total)))


@dataclass
class UsageMeter:
    """Thread-safe collector. One per model per stage; `summary()` aggregates.

    Threading matters: `evaluate()` fans out over a `ThreadPoolExecutor`, so records arrive
    concurrently and a bare `list.append` would be the only unguarded shared state in the
    call path.
    """

    records: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _t0: float | None = field(default=None, repr=False)
    _t1: float | None = field(default=None, repr=False)
    # Per-thread view of "the call this worker just made", for `last_record`. A meter-wide
    # `records[-1]` cannot answer that question under a pool: eight workers append
    # concurrently, so the last element belongs to whichever thread finished most recently.
    _tls: threading.local = field(default_factory=threading.local, repr=False)

    # -- collection ------------------------------------------------------------------

    def start(self) -> "UsageMeter":
        """Mark the beginning of the wall-clock window. Returns self so it can chain."""
        self._t0 = time.time()
        return self

    def stop(self) -> "UsageMeter":
        self._t1 = time.time()
        return self

    def add(self, rec: CallRecord) -> CallRecord:
        with self._lock:
            self.records.append(rec)
            if self._t0 is None:            # measuring started implicitly at first call
                self._t0 = time.time() - rec.latency_s
        self._tls.record = rec
        return rec

    def last_record(self) -> dict | None:
        """This thread's most recent call, as a plain dict, or None if it has made none.

        Exists for the evaluator pool's checkpoint: a resumable run has to write down what
        each call cost at the moment it completes, and the scoring contract
        (`(entry, key) -> float`) has no room to return it. **Must be read on the thread
        that made the call** -- reading it from the collector loop returns the main thread's
        empty slot, which writes every checkpoint row with no usage.
        """
        rec = getattr(self._tls, "record", None)
        return asdict(rec) if rec is not None else None

    def record(self, resp, model: str, latency_s: float, **kw) -> CallRecord:
        """Record a successful call from its response object."""
        prompt, completion, reported = extract_usage(resp)
        chars, reason_chars = extract_content(resp)
        if reported is not None:
            reasoning, source = reported, "reported"
        else:
            reasoning = derive_reasoning_tokens(completion, chars, reason_chars)
            source = "derived"
        return self.add(CallRecord(
            model=model, prompt_tokens=prompt, completion_tokens=completion,
            reasoning_tokens=reasoning, reasoning_source=source,
            content_chars=chars, reasoning_chars=reason_chars,
            latency_s=latency_s, ok=True, **kw))

    def record_error(self, exc: BaseException, model: str, latency_s: float,
                     **kw) -> CallRecord:
        """Record a failed call. The tokens are unknown; the time is not, and is real."""
        return self.add(CallRecord(
            model=model, latency_s=latency_s, ok=False,
            error=f"{type(exc).__name__}: {str(exc)[:200]}", **kw))

    # -- aggregation -----------------------------------------------------------------

    def wall_s(self) -> float:
        """Elapsed real time across the window, or 0.0 if nothing was measured."""
        if self._t0 is None:
            return 0.0
        return max(0.0, (self._t1 or time.time()) - self._t0)

    def summary(self) -> dict:
        """Totals, latency distribution, and the two throughput figures.

        `service_s` (sum of latencies) over `wall_s` (elapsed) is the concurrency actually
        achieved, which is the number the parallelization work is trying to move. It is
        reported rather than a target because the two diverge for reasons outside our
        control: a serial run has a ratio near 1.0, and so does a fully rate-limited
        4-worker run.
        """
        recs = list(self.records)
        ok = [r for r in recs if r.ok]
        lat = sorted(r.latency_s for r in recs)
        service = sum(r.latency_s for r in recs)
        wall = self.wall_s()
        prompt = sum(r.prompt_tokens for r in recs)
        completion = sum(r.completion_tokens for r in recs)
        chars = sum(r.content_chars for r in recs)
        reasoning = sum(r.reasoning_tokens for r in recs)
        derived = sum(1 for r in recs if r.reasoning_source == "derived")

        def pct(p: float) -> float | None:
            if not lat:
                return None
            # Nearest-rank on the sorted sample. At n=36 an interpolating percentile
            # invents a latency no call had, and these are reported as observations.
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 2)

        return {
            "n_calls": len(recs),
            "n_ok": len(ok),
            "n_error": len(recs) - len(ok),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            # The headline: what fraction of billed output the model never showed you.
            "reasoning_share": round(reasoning / completion, 3) if completion else None,
            # "derived" for all calls on every Fireworks model. Reported alongside so the
            # share is never mistaken for the provider's own accounting.
            "reasoning_source": ("derived" if derived == len(recs) else
                                 "reported" if derived == 0 and recs else "mixed"),
            "visible_tokens": completion - reasoning,
            "content_chars": chars,
            "total_tokens": prompt + completion,
            # Per *successful* call: a mean over failures (0 tokens) would report a
            # rate-limited model as unusually terse.
            "tokens_per_ok_call": round((prompt + completion) / len(ok), 1) if ok else None,
            "wall_s": round(wall, 1),
            "service_s": round(service, 1),
            "concurrency": round(service / wall, 2) if wall > 0 else None,
            "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
            "latency_p50_s": pct(0.50),
            "latency_p95_s": pct(0.95),
            # Output tok/s against elapsed time -- the rate that provokes rate limits.
            "out_tps": round(completion / wall, 1) if wall > 0 else None,
        }

    def by(self, key: str) -> dict[str, dict]:
        """Sub-summaries grouped by any `CallRecord` field, e.g. "label" or "attempt".

        Each group keeps the parent's wall clock, since the groups ran interleaved inside
        one window and splitting elapsed time between them would be fiction.
        """
        groups: dict[str, list[CallRecord]] = {}
        for r in self.records:
            groups.setdefault(str(getattr(r, key)), []).append(r)
        out = {}
        for name, recs in sorted(groups.items()):
            sub = UsageMeter(records=recs)
            sub._t0, sub._t1 = self._t0, self._t1
            out[name] = sub.summary()
        return out

    def item_stage_matrix(self) -> dict[str, dict]:
        """`{item_id: {"stages": {role: service_s}, ...}}` -- the per-pair critical path.

        This is the number the "stages 1 and 3 are cheap" claim rests on, so it is built to
        be quotable per pair rather than only in aggregate:

        - `stages` is summed latency per role, i.e. how long that pair spent in each stage.
        - `service_s` is the pair's whole chain, the closest honest analogue of "how long
          does one datapoint take" under concurrency.
        - `retries` counts calls beyond the first, so a pair that only looks slow because it
          retried is distinguishable from one that is genuinely slow.

        Deliberately not a wall-clock figure per pair: `CallRecord` carries a duration and
        no start timestamp, so the gaps between a pair's calls -- time spent queued behind
        other workers -- are unmeasurable here. Reporting summed latency as though it were
        elapsed time would overstate throughput.
        """
        by_item: dict[str, list[CallRecord]] = {}
        for r in self.records:
            by_item.setdefault(r.item_id, []).append(r)

        out: dict[str, dict] = {}
        for item, recs in sorted(by_item.items()):
            stages: dict[str, float] = {}
            tokens: dict[str, int] = {}
            for r in recs:
                stages[r.role] = stages.get(r.role, 0.0) + r.latency_s
                tokens[r.role] = tokens.get(r.role, 0) + r.total_tokens
            out[item] = {
                "stages": {k: round(v, 2) for k, v in sorted(stages.items())},
                "stage_tokens": dict(sorted(tokens.items())),
                "service_s": round(sum(r.latency_s for r in recs), 2),
                "total_tokens": sum(r.total_tokens for r in recs),
                "n_calls": len(recs),
                "n_error": sum(1 for r in recs if not r.ok),
                # Calls past the first attempt. Separates "slow" from "retried".
                "retries": sum(1 for r in recs if r.attempt > 1),
            }
        return out

    def stage_service(self) -> dict[str, float]:
        """Summed latency per stage across every item. Additive, unlike wall clock."""
        out: dict[str, float] = {}
        for r in self.records:
            out[r.role] = out.get(r.role, 0.0) + r.latency_s
        return {k: round(v, 1) for k, v in sorted(out.items())}

    def untimed_calls(self) -> dict[str, int]:
        """Per stage, how many *successful* calls spent tokens but recorded no latency.

        This is what decides whether a stage's service time may be quoted as a total or only
        as a floor, and it has to be measured rather than assumed: the claim "the user
        simulator is untimed" was true of tau3 for months and then stopped being true the
        moment the dropped `generation_time_seconds` was restored in the fork. A report that
        hardcodes the caveat keeps printing it after it is false, which is the same failure as
        omitting it while it is true.

        Failed calls are excluded -- a call that raised has no latency to record and is
        already counted by `n_error`.
        """
        out: dict[str, int] = {}
        for r in self.records:
            if not r.ok:
                continue
            if (r.prompt_tokens or r.completion_tokens) and not r.latency_s:
                out[r.role] = out.get(r.role, 0) + 1
        return dict(sorted(out.items()))

    def to_json(self) -> dict:
        return {"summary": self.summary(), "records": [asdict(r) for r in self.records]}


def timed_call(fn, meter: UsageMeter | None, model: str, **kw):
    """Run `fn()`, record its usage and latency, and re-raise anything it throws.

    The wrapper deliberately does not swallow the exception: the caller's retry logic is
    what decides whether a failure is recoverable, and accounting must not change control
    flow. It only guarantees the failure is *counted* before it propagates.
    """
    t0 = time.time()
    try:
        resp = fn()
    except BaseException as exc:
        if meter is not None:
            meter.record_error(exc, model=model, latency_s=time.time() - t0, **kw)
        raise
    if meter is not None:
        meter.record(resp, model=model, latency_s=time.time() - t0, **kw)
    return resp


def merge(meters: dict[str, UsageMeter]) -> dict:
    """Roll several per-model meters into one report, with a grand total.

    **For strictly sequential stages only.** Wall clock is summed, not maxed, which is right
    when models run one after another and wrong the moment any two overlap: N models running
    concurrently for an hour would report N hours elapsed, deflating every derived rate by N.

    The evaluator sweep is no longer sequential -- it runs one pool across the whole roster
    -- so it uses `pooled_report` on a single meter instead. This function stays for the
    stages that really are serial, and because a `{"total", "per_model"}` reader should not
    have to care which produced its input.
    """
    per = {name: m.summary() for name, m in meters.items()}
    keys = ("n_calls", "n_ok", "n_error", "prompt_tokens", "completion_tokens",
            "reasoning_tokens", "visible_tokens", "total_tokens", "content_chars",
            "wall_s", "service_s")
    total = {k: round(sum(p.get(k) or 0 for p in per.values()), 1) for k in keys}
    total["reasoning_share"] = (round(total["reasoning_tokens"] / total["completion_tokens"], 3)
                               if total["completion_tokens"] else None)
    total["out_tps"] = (round(total["completion_tokens"] / total["wall_s"], 1)
                        if total["wall_s"] else None)
    return {"total": total, "per_model": per}


def pooled_report(meter: UsageMeter, key: str = "label") -> dict:
    """The same `{"total", "per_model"}` shape as `merge`, for one meter spanning one pool.

    Read this way because the arithmetic differs from `merge` in exactly one place that
    matters. Under a pool there is **one** wall clock -- the run's -- and every model shares
    it, so:

    - `total.wall_s` is real elapsed time, not a sum, and `total.out_tps` is the rate the
      providers actually saw.
    - each `per_model` entry carries that same window (see `UsageMeter.by`), so its
      `concurrency` reads as **that model's average share of the pool's workers**: nine
      models over 8 workers sit near 0.9 each, and a model at 3.0 is occupying three
      workers' worth of the pool. Under `merge` the same field meant "how parallel was this
      model's own private pool", which is a different question with a different answer.

    `latency_p50_s` / `latency_p95_s` survive per model and are pooled away in the total,
    for the reason `merge` also drops them: nemotron showed 124.2s against gpt-oss-20b's
    5.3s on the same six calls, so a pooled percentile describes no model.
    """
    total = meter.summary()
    for k in ("latency_mean_s", "latency_p50_s", "latency_p95_s"):
        total.pop(k, None)
    return {"total": total, "per_model": meter.by(key), "pooled": True}
