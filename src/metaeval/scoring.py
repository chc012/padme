"""Score each trajectory on its own; recover the preference from the two scores.

This is the evaluator under test, not a judge. `_score_model` sees one trajectory and one
criterion, and `evaluate` derives which side won by comparing a pair's two scores. A deployed
evaluator never gets the counterfactual, so it is not given one here.

Invariants that shape the file:

- **A tie is wrong, never excluded.** Both sides were steered in opposite directions on the
  criterion, so equal scores mean a real difference went undetected. See `compute_metrics`.
- **An entry needs all `2 * num_runs` calls to succeed**, so one transient failure would
  otherwise discard a whole pair. Hence the retry passes in `evaluate`, which narrow the pool
  rather than competing with themselves.
- **A reply without a valid score is rejected, not clamped** (`_extract_score`). Reading "7" on
  a 0-1 scale as 1.0 would invent an opinion the model never expressed.

The prompt gives the evaluator what the annotators were shown and no more; see `SCORING_PROMPT`.
"""

import collections
import contextlib
import json
import os
import random
import re
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import litellm
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .usage import UsageMeter, timed_call

load_dotenv()

litellm.suppress_debug_info = True
os.environ.pop("ANTHROPIC_BASE_URL", None)


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

# Written to give the evaluator the same information the human annotators had, and no more:
# if the evaluator reads a different string it is doing a different task.
#
# What that pins down: judge one criterion and nothing else (including that a conversation can
# fail its task and still be better on the criterion); judge the agent only, since the customer
# turns come from a simulator; the reference block is labelled as the agent's tools and policy
# rather than as a user prompt; the documents are optional, as they were for annotators; and no
# ground truth is present -- `render_context` withholds task purpose, expected outcome, domain
# and task id, because a deployed evaluator has no answer key.
#
# Reasoning is capped at 3 sentences. Unbounded "step-by-step" once ran a reply to 41,006
# characters and past the token cap without reaching a verdict.
SCORING_PROMPT = """\
You are evaluating one customer-service conversation on a single named criterion.

CRITERION: {criterion_name}
{criterion_description}

Judge this criterion and nothing else. Ignore everything the criterion does not
mention. A conversation can fail at its task and still be better on the criterion.

Judge the AGENT's conduct only. The customer's turns were produced by a simulator
and are not being evaluated.

REFERENCE DOCUMENTS (the agent's tools, and the policy it had to follow). Consult
them if you need to check whether the agent was permitted to do something. You will
not need them for most conversations.
{prompt}

CONVERSATION
Turns are numbered. "Agent calls X(...)" is a tool call the agent made, and
"Tool result ->" is what came back.
{response}

Score the agent on {criterion_name} from 0.0 (worst) to 1.0 (best). Use the full
range: reserve 0.0 and 1.0 for clear cases and place ordinary ones in between.

Return JSON with exactly two keys:
- "reasoning": at most 3 sentences, citing what in the conversation drove the score
- "score": a number between 0.0 and 1.0
"""


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# No `response_format={"type": "json_object"}`: on `qwen3-30b-a3b` it **suppresses the stop
# token**, taking one call from 472 to 2,996 tokens with no clean termination. `_parse_json`
# already recovers JSON from a code fence or the first balanced braces.
#
# `max_tokens` was absent entirely, which left the provider default in charge -- on a
# reasoning model the trace is billed against that budget and can consume all of it,
# returning empty content that surfaces as "No score in model response".
#
# `timeout`/`num_retries` mirror `CALL_CONTROL` in `sources/tau2_source.py`: without a
# timeout one stalled request hangs the batch at 0% CPU against a healthy API.
# `temperature` is explicit, and 0.0, because omitting it was the same bug that made the
# judges non-reproducible: the provider default (~1.0) silently takes over. It showed up
# here as a measurement, not a suspicion -- gpt-oss-20b scored 77.8% on one run and 55.6%
# across three, with a consistent preference on only **15 of 36** pairs. That is sampling
# noise being read as evaluator quality.
#
# Consequence for `--runs k>1`: at temperature 0 repeated runs are near-identical, so k
# measures residual provider nondeterminism (MoE routing varies with batch composition)
# rather than sampling spread. That is the number worth having -- a deployed evaluator is
# run once, and what matters is whether that once is reproducible.
SCORE_TEMPERATURE = 0.0
SCORE_CALL = {"temperature": SCORE_TEMPERATURE, "max_tokens": 10_000,
              "timeout": 120, "num_retries": 4}


# Models state a perfectly good score in a wrapper `json.loads` rejects, and gemini-2.5-flash
# does it about a third of the time: it ignores the JSON instruction and writes prose ending
# `Score: 1.0` or `score: 0.2`.
#
# That rate is why coverage collapsed rather than dipped. An entry needs all `2 * num_runs`
# calls to succeed, so at k=3 a 33% per-call failure leaves 0.67**6 ~ 9% of entries -- and
# gemini-2.5-flash returned **3 of 36**, which matches. A per-call annoyance became a
# 92% data loss through the entry-level guard.
#
# The score must be recovered, not the entry retried into submission: the model answered
# correctly and only the envelope was wrong. Failure is NOT truncation -- `finish_reason` is
# `stop` and reasoning used 1,836 of 10,000 tokens -- so a bigger budget fixes nothing.
#
# **Every pattern is anchored on a word that means "this number is the verdict"** -- `score`,
# or `criterion` -- so prose numbers are never mistaken for one: a reply saying "in turn 5 the
# agent apologised" must not yield 5, and "it made 1 tool call" must not yield 1.0. The last
# match wins, since the verdict comes after the reasoning.
#
# Patterns 3 and 4 were added after `llama3.1-8b` produced 8 of the 13 unparsed replies in a
# 2,100-call run, all of them *stating a perfectly good score* in a form the first two miss:
#
#     "The agent's score on task_resolution is 0.8."      <- `score ... is N`, not `score: N`
#     "The agent's friendliness score is 0.8."             <- same
#     "The agent's conduct on the criterion task_resolution is 0.0."   <- no "score" at all
#
# `is` joins `:` and `=` as a copula in pattern 2's successor; the `criterion` form gets its
# own pattern because the anchoring word is different. Both still require the anchor, so the
# guarantee above is unchanged -- what widened is the punctuation between anchor and number,
# not the licence to read any number as a verdict.
#
# Ordered least to most permissive, and `_extract_score` returns on the first pattern that
# yields an in-range value, so adding these cannot change a reply the earlier patterns
# already parsed.
_SCORE_PATTERNS = (
    re.compile(r'"score"\s*:\s*"?(-?\d*\.?\d+)', re.IGNORECASE),
    re.compile(r'\bscore\b\s*\**\s*[:=]\s*\**\s*"?(-?\d*\.?\d+)', re.IGNORECASE),
    # `score ... is 0.8` -- at most a few words between, so "the score, which the rubric in
    # section 5 defines, is 3" cannot drift into matching something unrelated.
    re.compile(r'\bscore\b[^.\n]{0,40}?\bis\s+"?(-?\d*\.?\d+)', re.IGNORECASE),
    # `criterion <name> is 0.0` -- the criterion name is the anchor when "score" is absent.
    re.compile(r'\bcriterion\b[^.\n]{0,40}?\bis\s+"?(-?\d*\.?\d+)', re.IGNORECASE),
)


def _extract_score(raw: str) -> float | None:
    """A 0-1 score from a model reply, or None if it never gave one.

    Out-of-range values are rejected rather than clamped: a reply scoring "7" is not
    answering the question that was asked, and silently reading it as 1.0 would invent an
    opinion the model did not express.
    """
    parsed = _parse_json(raw)
    if "score" in parsed:
        try:
            v = float(parsed["score"])
            if 0.0 <= v <= 1.0:
                return v
        except (TypeError, ValueError):
            pass
    for pattern in _SCORE_PATTERNS:
        found = pattern.findall(raw or "")
        for tok in reversed(found):
            try:
                v = float(tok)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                return v
    return None


def _score_model(entry: dict, response_key: str, model: str,
                 meter: UsageMeter | None = None, label: str = "",
                 extra: dict | None = None) -> float:
    """Score one response. `meter` is optional so the scoring contract stays
    `(entry, key) -> float` and every existing caller and test keeps working.

    A no-score reply is recorded as a *successful* call, because it was one: the provider
    answered, billed, and returned prose. Counting it as an error would hide its tokens,
    and this failure mode is expensive precisely because it is verbose -- gemini-2.5-flash
    ignored the JSON instruction on a third of calls and wrote paragraphs.

    `extra` is merged over `SCORE_CALL`, for a model that needs a different endpoint or a
    different spelling of the same parameter. **It can remove a key as well as add one**, by
    mapping it to `None` -- which is what a gateway rejecting `max_tokens` outright requires,
    since sending it alongside `max_completion_tokens` is a 400 rather than a preference. See
    `ENDPOINTS` in `tools/run_evaluators.py` for the two that need it.

    **No `reasoning_effort` anywhere, deliberately.** Every model runs at whatever reasoning
    it does by default, because the question is what a deployed evaluator gives you off the
    shelf. That default varies enormously and is measured rather than assumed: on one real
    scoring prompt, Claude through the gateway reported 0 reasoning tokens, gpt-5.4-mini 0,
    gpt-5.6-sol 86 of 151, gemini-3.7-flash 154 of 233, and gpt-5-nano 1,088 of 1,173. That
    spread is a finding about the roster, not noise to be normalised away.
    """
    prompt_text = SCORING_PROMPT.format(
        criterion_name=entry["criterion_name"],
        criterion_description=entry["criterion_description"],
        prompt=entry["prompt"],
        response=entry[response_key],
    )
    call = {**SCORE_CALL, **(extra or {})}
    call = {k: v for k, v in call.items() if v is not None}
    resp = timed_call(
        lambda: litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            **call,
        ),
        meter, model=model, role="evaluator", label=label or model,
        item_id=str(entry.get("id", "")),
    )
    raw = resp.choices[0].message.content
    score = _extract_score(raw)
    if score is None:
        raise ValueError(f"No score in model response: {raw!r}")
    return score


# --------------------------------------------------------------------------- #
# The work unit, and the pool that runs it
# --------------------------------------------------------------------------- #
#
# **One pool over every model, not one pool per model.** The sweep used to be a `for` loop
# over the roster, each model getting its own `ThreadPoolExecutor`, which cost three things
# that all get worse as the dataset grows from 36 pairs to 1,000:
#
# 1. **Cross-provider idle.** `gemini-2.5-flash` took 957s of a 61-minute sweep with every
#    Fireworks model sitting idle -- 26% of the wall clock, recoverable at zero added
#    pressure on either provider.
# 2. **Dedicated deployments went back to sleep.** They scale to zero after 5 minutes idle,
#    so a model warmed at its turn is asleep by the time the three models ahead of it
#    finish. `qwen3-4b` and `qwen3-1p7b` each scored 0 of 36 entries that way. A pool that
#    interleaves models keeps every deployment continuously in traffic, which is the fix --
#    warming is now a start-up step rather than a per-model one.
# 3. **Serial models concentrate load.** N workers all hitting *one* model is exactly the
#    shape a per-model token budget rejects: 8 workers on `nemotron-lightning` returned 20
#    rate-limit errors where 4 returned 1. Spread across nine models, 8 workers put ~1 call
#    in flight per model, so the same total throughput arrives without the concentration.
#
# Task order is shuffled. In dataset
# order the pool would run all nine models against pair 0, then all nine against pair 1 --
# nine near-simultaneous calls carrying the *same* 7k-token prompt, which is a burst against
# every provider at once. Shuffling decorrelates model from pair, so at any moment the eight
# in-flight calls are eight different models on eight different pairs. It is seeded, because
# a run has to be re-creatable.


@dataclass(frozen=True)
class Task:
    """One provider call: score one response, of one pair, for one model, on one run.

    `label` rather than the model string, because the label is what names the output files,
    the checkpoint rows and the roster row. The model string is looked up from it.
    """

    label: str
    index: int          # position in `dataset`
    key: str            # "response_1" | "response_2"
    run: int


def provider_of(model: str) -> str:
    """The provider a litellm model string routes to, as a fallback admission-control lane.

    `fireworks_ai/accounts/.../gpt-oss-20b` -> `fireworks_ai`, `gemini/gemini-3.7-flash` ->
    `gemini`. Used only to group models that share a rate limit.

    **The prefix is a guess, and for two cases it is the wrong one**, which is why
    `evaluate_models` takes a `lane_of` override and the roster sets it explicitly:

    - a **dedicated deployment** (`...#accounts/x/deployments/y`) shares nothing with
      serverless. Its limit is its own replica count -- one -- while the serverless prefix
      it appears under is an account-wide token budget. Capping them together caps the wrong
      thing in both directions.
    - a **gateway** reached as `openai/<model>` with an `api_base` is not OpenAI. Everything
      behind one gateway shares that gateway's limits, and nothing there shares OpenAI's.
    """
    return model.split("/", 1)[0] if "/" in model else model


class RateLimiter:
    """A token bucket over a sliding window: at most `n` starts per `window_s`, across threads.

    **A semaphore cannot bound a rate.** It bounds calls *in flight*, and the two only coincide
    when latency is constant: at a cap of 8 with 2.9s mean latency you get ~2.8 starts/s, but
    the same cap yields 8/s the moment the provider answers in 1s. The gateway serving the Claude
    and GPT rows states its limit as **requests per minute per credential** -- a rate -- and a
    concurrency
    cap of 8 tripped it anyway, 109 times in the first 2,100 calls of a sweep. Every one of
    those was on the gateway; no Fireworks or Gemini row hit a single 429.

    Rate limits are also where a retry storm comes from, which is why this sits *outside* the
    call rather than around a retry: litellm's own `num_retries` fires immediately and knows
    nothing about the other fifteen threads, so a saturated gateway turns one logical call into
    five requests and the failure feeds itself. Admitting fewer starts is the only thing that
    breaks that loop.

    **The rate is enforced over a short sub-window, because a long one does not bound the
    burst.** `n` is given per minute, which is how providers state their limits, but counting
    over a full 60s window permits all `n` requests in the first instant of it -- so a 240/min
    limiter happily fires 240 at once. That is not a hypothetical: measured at **56 req/min of
    throughput against a stated 300 req/min limit, and still 18 rejections**, because the
    gateway evidently counts over something much shorter than a minute. Smoothing to `n/60` per
    second bounds instantaneous starts to a handful while leaving the per-minute average
    untouched, and that is the whole difference between the two failing configurations and a
    working one.

    **Continuous refill, not an integer count per window, because the integer lied.** The
    first version kept a deque of starts and admitted `round(n_per_min * window_s / 60)` per
    window. At a 1s window that expression can only represent multiples of 60/min, and it
    rounds everything else to one of them -- silently, in whichever direction is nearer:

        configured 240/min -> 4/s -> 240/min   exact
        configured 100/min -> 2/s -> 120/min   +20%, over the limit it was asked to respect
        configured  72/min -> 1/s ->  60/min   -17%, throughput given away
        configured  40/min -> 1/s ->  60/min   +50%

    Both directions are bugs and the first is the dangerous one: a limiter set to the number
    the provider stated would have run 20% above it. So the budget is now a float refilled
    continuously -- `n_per_min / 60` tokens per second -- which represents any rate exactly.
    `window_s` survives as the burst allowance: capacity is `rate * window_s` tokens, so the
    instantaneous burst is still the handful that the sliding window was there to enforce, and
    the long-run average is now the number that was actually configured.

    At least one token of capacity, so a small `n` still makes progress rather than
    deadlocking.
    """

    def __init__(self, n_per_min: int, window_s: float = 1.0):
        self.n_per_min = n_per_min
        self.window_s = window_s
        self.rate = n_per_min / 60.0              # tokens per second, exact
        # Burst capacity, kept as `n` for continuity with the window formulation: at 240/min
        # and a 1s window this is 4.0, the same four-in-one-second the deque allowed.
        self.n = max(1.0, self.rate * window_s)
        self._tokens = self.n                     # start full: the first burst is free
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a start is allowed, then record it."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.n, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Sleep exactly as long as one token needs to accrue. Computed under the lock
                # and slept outside it, so waiting threads do not block each other.
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(min(max(wait, 0.005), 1.0))

    @contextlib.contextmanager
    def hold(self):
        self.acquire()
        yield


class Checkpoint:
    """Every call that has already been paid for, on disk, so a kill costs one call.

    An 11-hour sweep with no checkpoint is an 11-hour sweep you cannot interrupt: results
    were written only when a model finished, so a crash at hour 3 of a 4-hour model threw
    away all of it. At 1,000 pairs that is no longer a theoretical cost.

    **Append-only JSONL, one line per call, rather than a periodically-rewritten blob.**
    54,000 lines of ~120 bytes is 6MB, and appending needs no read of what is already there
    -- so the write cannot corrupt earlier rows, and a truncated final line costs exactly
    the call it described.

    **A failed call is written too, with `score: null`.** It is not treated as done, so a
    resume retries it, since rate limits are the dominant
    failure and they are transient. But it *billed*, and dropping it would understate the
    run's cost in the same way the lost invocation understated the collection run's. So
    failures appear as cost and not as progress, and a call retried across three resumes
    leaves three failure rows, which is three real calls.
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def load(self) -> tuple[dict[tuple, float], list[dict]]:
        """`({(label, id, key, run): score}, [usage records])` from disk.

        Unparseable lines are skipped rather than fatal: the last line of a killed run is
        routinely a half-written one, and refusing to resume because of it would defeat the
        feature.
        """
        done: dict[tuple, float] = {}
        records: list[dict] = []
        if not self.path or not self.path.is_file():
            return done, records
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("usage"):
                    records.append(row["usage"])
                if row.get("score") is None:
                    continue
                done[(row.get("label", ""), row.get("id"), row.get("key"),
                      row.get("run"))] = row["score"]
        return done, records

    def put(self, task: Task, entry_id: str, score: float | None,
            record: dict | None) -> None:
        if not self.path:
            return
        row = {"label": task.label, "id": entry_id, "key": task.key, "run": task.run,
               "score": score}
        if record:
            row["usage"] = record
        line = json.dumps(row, separators=(",", ":")) + "\n"
        # One lock and one `open(..., "a")` per call. O_APPEND makes the write atomic for a
        # line this short, and the lock keeps two threads from interleaving mid-line on the
        # platforms where it is not.
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(line)


def evaluate_models(
    dataset: list[dict],
    models: list[tuple[str, str]],
    score_fn,
    num_runs: int = 3,
    max_workers: int = 8,
    seed: int = 0,
    retry_passes: int = 3,
    retry_delay_s: float = 20.0,
    checkpoint: Checkpoint | None = None,
    resume: bool = True,
    provider_caps: dict[str, int] | None = None,
    lane_rates: dict[str, int] | None = None,
    lane_of=None,
    last_record=None,
    give_up_after: int = 20,
    progress: bool = True,
) -> dict[str, list[dict]]:
    """Score every (model, pair, side, run) over one shared pool. Returns `{label: results}`.

    `models` is `[(label, litellm_model)]`; the model string is used only to derive the
    provider for admission control. `score_fn(label, entry, response_key) -> float` does the
    call, and the caller closes over whatever else it needs (the model string, a meter).

    `last_record` is a zero-argument callable returning this thread's most recent
    `CallRecord` as a dict, or None. It exists so a checkpoint row carries what the call
    cost, without threading a return channel through the `(label, entry, key) -> float`
    contract every caller and test depends on -- `UsageMeter.last_record` satisfies it, and
    a caller with no meter passes nothing and gets rows with a score and no usage.

    `provider_caps` bounds how many calls may be in flight to one **lane** at once,
    regardless of pool size. The pool is the throughput knob; this is the guard rail, and it
    exists because the limit that actually binds is per-provider tokens/sec rather than our
    thread count. Absent an entry, a lane is bounded only by `max_workers`.

    A lane is whatever shares a rate limit, and `lane_of(label) -> str` says which -- because
    the model string cannot always tell you (see `provider_of`). It defaults to the string's
    provider prefix, which is right for a plain serverless roster and wrong for a gateway or
    a single-replica deployment.

    `lane_rates` maps a lane to **requests per minute**, enforced by `RateLimiter`. Use it when
    the provider states a rate rather than a concurrency -- the gateway's 429 body named its own
    limit, "300 req/min" -- because a `provider_caps` entry cannot express that and a cap of 8
    tripped it 109 times in 2,100 calls. The two compose: the cap bounds calls in flight, the
    rate bounds starts per minute, and a lane may set either, both or neither.

    Retry passes are unchanged in spirit and now global: one pass over every failed call
    across every model, narrower than the main pass and after a pause. The point is to stop
    competing with ourselves, and an entry still needs all `2 * num_runs` of its calls to
    survive. **Narrower, not serial** -- one worker was the instrument before `lane_rates`
    existed; at 4,417 failures out of 48,073 calls, serial retries take 3.7 hours.

    `give_up_after` abandons a model's remaining tasks once it has failed that many calls
    **without a single success**. The serial loop got this for free -- a whole-model exception
    ended that model's turn and the sweep moved on -- and one pool loses it: an unreachable
    model would otherwise work through all 2,000 of its calls, each retried four times inside
    litellm, and then appear again in all three retry passes. That is hours of a multi-hour
    run spent on a model that answered nothing. The `and no successes` clause is what keeps it
    from firing on a merely rate-limited model, which is the common case and the one the retry
    passes exist for. Set 0 to disable.
    """
    by_label = dict(models)
    tasks = [
        Task(label, i, key, run)
        for label in by_label
        for i in range(len(dataset))
        for key in ("response_1", "response_2")
        for run in range(num_runs)
    ]
    # Seeded, and shuffled after the full cross-product is built rather than per model, so
    # the interleaving is across models as well as across pairs.
    random.Random(seed).shuffle(tasks)

    ck = checkpoint or Checkpoint(None)
    # `resume=False` re-scores everything but still *appends*, so a deliberate re-measurement
    # costs calls without erasing what earlier invocations paid. Writing is the checkpoint's
    # other job -- reading is the only part being switched off.
    done, _prior = ck.load() if resume else ({}, [])
    scores: dict[Task, float | None] = {}
    todo = []
    for t in tasks:
        prior = done.get((t.label, dataset[t.index]["id"], t.key, t.run))
        if prior is None:
            todo.append(t)
        else:
            scores[t] = prior
    if done and len(todo) < len(tasks):
        print(f"  resuming: {len(tasks) - len(todo):,} of {len(tasks):,} calls already on "
              f"disk, {len(todo):,} to go", flush=True)

    caps = {p: threading.Semaphore(n) for p, n in (provider_caps or {}).items() if n}
    rates = {p: RateLimiter(n) for p, n in (lane_rates or {}).items() if n}
    lane = lane_of or (lambda label: provider_of(by_label[label]))
    lanes = {label: lane(label) for label in by_label}
    # Per-label tallies, read by the circuit breaker and by the error-print budget. Guarded,
    # because eight workers update them concurrently and `+= 1` on a dict value is not atomic.
    ok_n: collections.Counter = collections.Counter()
    err_n: collections.Counter = collections.Counter()
    # Tasks whose failure will reproduce -- a reply with no score in it, as opposed to a rate
    # limit. A `set` written from worker threads: CPython's GIL makes `set.add` atomic, and the
    # only reader runs between passes.
    deterministic: set = set()
    tally = threading.Lock()
    ERROR_PRINT_BUDGET = 10

    def dead(label: str) -> bool:
        """Has this model earned being abandoned? Failures count only while it has none."""
        if not give_up_after:
            return False
        with tally:
            return ok_n[label] == 0 and err_n[label] >= give_up_after

    # `last_record()` is read **inside the worker**, not beside `future.result()`. It is
    # thread-local by necessity -- eight workers append to one meter, so a meter-wide
    # `records[-1]` hands a worker somebody else's call -- and the loop that collects
    # results runs on the main thread, where that thread-local is empty. Reading it there
    # silently wrote every checkpoint row with no usage at all.
    def call(task: Task):
        entry = dataset[task.index]
        # Checked here rather than by filtering the batch: the whole pass is already submitted
        # when the breaker trips, so the only place left to stop is at the top of the task that
        # would have made the call.
        if dead(task.label):
            return task, None, None
        gate = caps.get(lanes[task.label])
        pace = rates.get(lanes[task.label])
        try:
            # Rate outside concurrency: a thread waiting for its slot in the window must not be
            # holding one of the lane's in-flight slots while it waits.
            with pace.hold() if pace is not None else contextlib.nullcontext():
                with gate if gate is not None else contextlib.nullcontext():
                    score = score_fn(task.label, entry, task.key)
            with tally:
                ok_n[task.label] += 1
            return task, score, last_record() if last_record else None
        except Exception as e:  # noqa: BLE001
            with tally:
                err_n[task.label] += 1
                n = err_n[task.label]
            # **Budgeted, because the console is a resource at this scale.** One line per
            # failure was fine for 36 pairs; an unreachable model on 1,000 pairs writes 6,000
            # of them and buries every other model's output. The count stays exact -- it is in
            # `err_n`, in the summary and in the checkpoint.
            if n <= ERROR_PRINT_BUDGET:
                print(f"\n  [ERROR] {task.label} id={entry['id']} {task.key} "
                      f"run={task.run}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            elif n == ERROR_PRINT_BUDGET + 1:
                print(f"\n  [ERROR] {task.label}: further errors silenced; the count is "
                      f"reported at the end", flush=True)
            # **Is this worth retrying?** A rate limit or a socket error is transient and a
            # second attempt is the whole point of the retry passes. A reply that parsed fine
            # and simply contained no score is *not*: at temperature 0 the model reproduces it,
            # and the evidence is direct -- 130 such calls were attempted 3 to 6 times in one
            # sweep and never once yielded a score.
            #
            # Retrying them is not merely useless, it is the most expensive thing in the run.
            # The failure mode is a `<think>` block that runs to the 10,000-token cap without
            # reaching a verdict, so every attempt generates the maximum output the budget
            # allows -- ~60s on a 2B model -- and litellm's own `num_retries` multiplies that by
            # five. Two retry passes over 130 such calls was on track for **five hours**, after
            # a main pass that had already scored 99.7% of the sweep, and no results are written
            # until the passes finish.
            #
            # Still retried **once**, not zero times: temperature 0 is not bitwise reproducible
            # on a MoE, so a truncated trace can occasionally come back complete. Once buys that
            # chance; three times buys nothing and costs hours.
            if isinstance(e, ValueError):
                deterministic.add(task)
            return task, None, last_record() if last_record else None

    def run_pass(batch: list[Task], workers: int, label: str) -> None:
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(call, t) for t in batch]
            it = as_completed(futures)
            if progress:
                it = tqdm(it, total=len(futures), desc=label)
            for future in it:
                task, score, record = future.result()
                scores[task] = score
                ck.put(task, dataset[task.index]["id"], score, record)

    run_pass(todo, max_workers, "evaluate")

    # **A retry pass runs narrower than the main pass, not at exactly one worker.**
    #
    # One worker was the right crude instrument when nothing else bounded pressure: the point
    # was "stop competing with ourselves", and a single thread guarantees that whatever the
    # provider's limit turns out to be. It does not survive contact with scale. A 48,073-call
    # pass left 4,417 failures, and retrying those serially at ~3s each is **3.7 hours** --
    # longer than most of the sweep, spent almost entirely idle.
    #
    # `lane_rates` now does that job properly, per lane and in the units the provider states,
    # so the retry pass can use real concurrency and still be gentle. A quarter of the main
    # pool (minimum 1) keeps the back-off intent while making the pass finish in minutes: the
    # rate limiter, not the thread count, is what bounds pressure now.
    retry_workers = max(1, max_workers // 4)
    for attempt in range(1, retry_passes + 1):
        failed = [t for t, v in scores.items() if v is None]
        if attempt > 1:
            # Everything still failing for a reason that will not change is dropped here, so
            # the later passes are spent only on calls that might actually land.
            stuck = [t for t in failed if t in deterministic]
            failed = [t for t in failed if t not in deterministic]
            if stuck:
                print(f"\n  {len(stuck)} call(s) returned no score on every attempt and are "
                      f"not retried again; see the note in `call`", flush=True)
        if not failed:
            break
        print(f"\n  retry pass {attempt}/{retry_passes}: {len(failed)} failed call(s), "
              f"{retry_workers} worker(s) after {retry_delay_s:.0f}s", flush=True)
        time.sleep(retry_delay_s)
        run_pass(failed, retry_workers, f"retry{attempt}")

    still = [t for t, v in scores.items() if v is None]
    if still:
        lost = {(t.label, t.index) for t in still}
        # "counted wrong", not "costing": the entry survives and is graded as a miss. See
        # `_assemble` for why that is the right charge rather than a dropped row.
        print(f"\n  {len(still)} call(s) unrecoverable after {retry_passes} retry pass(es); "
              f"{len(lost)} model-entry pair(s) counted WRONG", flush=True)
    for label in by_label:
        if dead(label):
            print(f"  {label}: abandoned after {err_n[label]} failure(s) and no successes "
                  f"(give_up_after={give_up_after})", flush=True)
        elif err_n[label] > ERROR_PRINT_BUDGET:
            print(f"  {label}: {err_n[label]} failed call(s), {ok_n[label]} succeeded",
                  flush=True)

    return {label: _assemble(dataset, scores, label, num_runs) for label in by_label}


def _assemble(dataset: list[dict], scores: dict, label: str, num_runs: int) -> list[dict]:
    """Pair scores back into per-entry results for one model.

    **An unrecoverable call counts as wrong, not as missing data**, and that follows from the
    same argument that makes a tie wrong. Steering produced two deliberately different
    trajectories and annotators agreed the difference is perceptible, so an evaluator that
    reports no difference has failed to detect one. An evaluator that cannot emit a parseable
    score at all has failed the same task, harder -- it did not functionally finish the
    example. Dropping those entries instead would quietly grade a model on the subset it
    managed to answer, which flatters exactly the models least able to do the job.

    That this is the coarse-scale failure again, not a separate plumbing problem, is visible in
    who it happens to: every unrecoverable call in the published sweep fell on `llama3.1-8b`
    and `qwen3-1p7b`, two of the three lowest-ranked rows, one of which was
    already the only one whose *direction* was wrong. "Emits no score" and "emits a score that
    means nothing" look like one weakness.

    So `n` is 1,000 for every model and the shortfall moves into an error count, which is
    reported beside every number. Errors stay distinguishable from ties: an error run carries
    `error: True` and a `None` score, a tie carries two equal scores.

    **A model with no successful calls at all is still dropped entirely**, returning `[]`. That
    is not an inconsistency: 0.0% accuracy with 100% errors reads as a uniquely terrible
    evaluator when the truth is that its deployment never came up, and that misreading has
    happened here before -- `qwen3-4b` and `qwen3-1p7b` once reported 0.0% accuracy with 0.0%
    errors after sleeping through their turns. Upstream turns `[]` into "NO DATA".
    """
    if not any(scores.get(Task(label, i, k, r)) is not None
               for i in range(len(dataset)) for k in ("response_1", "response_2")
               for r in range(num_runs)):
        return []

    results = []
    for i, entry in enumerate(dataset):
        scores_1 = [scores.get(Task(label, i, "response_1", r)) for r in range(num_runs)]
        scores_2 = [scores.get(Task(label, i, "response_2", r)) for r in range(num_runs)]

        runs = []
        for r in range(num_runs):
            s1, s2 = scores_1[r], scores_2[r]
            # A missing score is not a preference, so `higher_score` is None -- the same value a
            # tie carries, because both are "this evaluator named no winner". `error` is what
            # tells them apart downstream, and `_row` already keys its tie count on `score_1`
            # being present for exactly this reason.
            err = s1 is None or s2 is None
            higher_score = None if err or s1 == s2 else (1 if s1 > s2 else 2)
            runs.append({
                "run": r,
                "score_1": s1,
                "score_2": s2,
                "higher_score": higher_score,
                "correct": (not err) and higher_score == entry["correct_response"],
                **({"error": True} if err else {}),
            })

        def _stats(vals: list) -> tuple:
            """Mean and sd over the runs that produced a score; None when none did."""
            got = [v for v in vals if v is not None]
            if not got:
                return None, None
            return statistics.mean(got), (statistics.stdev(got) if len(got) > 1 else 0.0)

        m1, sd1 = _stats(scores_1)
        m2, sd2 = _stats(scores_2)
        results.append({
            "id": entry["id"],
            "criterion_name": entry["criterion_name"],
            "prompt": entry["prompt"],
            "response_1": entry["response_1"],
            "response_2": entry["response_2"],
            "correct_response": entry["correct_response"],
            "runs": runs,
            "pass_rate": sum(r["correct"] for r in runs) / num_runs,
            "score_1_mean": m1,
            "score_1_std": sd1,
            "score_2_mean": m2,
            "score_2_std": sd2,
        })
    return results


def evaluate(
    dataset: list[dict],
    score_fn,
    num_runs: int = 3,
    max_workers: int = 8,
    retry_passes: int = 3,
    retry_delay_s: float = 20.0,
    checkpoint: Checkpoint | None = None,
    seed: int = 0,
) -> list[dict]:
    """One model, over the same pool. `score_fn(entry, response_key) -> float`.

    Kept as its own entry point because the standalone CLI and every test call it, and
    because scoring one model is the common case when iterating. It is a one-model
    `evaluate_models`, so there is a single implementation of the retry rule, the
    all-calls-must-succeed guard and the result shape.
    """
    out = evaluate_models(
        dataset, [("", "")], lambda _label, e, k: score_fn(e, k),
        num_runs=num_runs, max_workers=max_workers, seed=seed,
        retry_passes=retry_passes, retry_delay_s=retry_delay_s, checkpoint=checkpoint,
    )
    return out[""]


def _wrong(correct_response: int) -> int:
    return 2 if correct_response == 1 else 1


def _pred_run(run: dict, correct_response: int) -> int:
    hs = run["higher_score"]
    return hs if hs is not None else _wrong(correct_response)


def compute_metrics(results: list[dict]) -> dict:
    """Per-run accuracy/F1 mean+std, mean score, and mean per-response score std."""
    if not results:
        return {
            "overall": {"n": 0, "mean_accuracy": 0.0, "std_accuracy": 0.0,
                        "mean_f1": 0.0, "std_f1": 0.0, "avg_raw_score": 0.0, "avg_per_response_std": 0.0},
            "by_criterion": {},
        }

    num_runs = len(results[0]["runs"])

    by_crit: dict[str, list] = defaultdict(list)
    for r in results:
        by_crit[r["criterion_name"]].append(r)

    def _group_metrics(entries: list[dict]) -> dict:
        run_accuracies, run_f1s = [], []
        for run_idx in range(num_runs):
            y_true = [e["correct_response"] for e in entries]
            y_pred = [_pred_run(e["runs"][run_idx], e["correct_response"]) for e in entries]
            run_accuracies.append(accuracy_score(y_true, y_pred))
            run_f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # **None-filtered, because an error run carries no score.** These two are descriptions
        # of the scores a model produced -- where they sit on the scale, and how much they move
        # between runs -- so a call that produced none contributes nothing rather than a zero.
        # Counting a failure as 0.0 would drag `avg_raw_score` toward the bottom of the scale
        # and read as a harsh evaluator, which is a different claim from "it did not answer".
        # The failure is charged where it belongs, in accuracy, via `_pred_run`.
        all_scores = [
            run[k] for e in entries for run in e["runs"] for k in ("score_1", "score_2")
            if run[k] is not None
        ]
        per_response_stds = [
            v for e in entries for v in (e["score_1_std"], e["score_2_std"])
            if v is not None
        ]

        return {
            "n": len(entries),
            "mean_accuracy": statistics.mean(run_accuracies),
            "std_accuracy": statistics.stdev(run_accuracies) if num_runs > 1 else 0.0,
            "mean_f1": statistics.mean(run_f1s),
            "std_f1": statistics.stdev(run_f1s) if num_runs > 1 else 0.0,
            "avg_raw_score": statistics.mean(all_scores) if all_scores else None,
            "avg_per_response_std": (statistics.mean(per_response_stds)
                                     if per_response_stds else None),
        }

    per_criterion = {crit: _group_metrics(entries) for crit, entries in sorted(by_crit.items())}
    overall = _group_metrics(results)
    return {"overall": overall, "by_criterion": per_criterion}


def print_metrics(metrics: dict) -> None:
    ov = metrics["overall"]
    print(f"\n=== Evaluator Metrics ===")
    print(f"  Overall (n={ov['n']}):  "
          f"accuracy={ov['mean_accuracy']:.1%}±{ov['std_accuracy']:.1%}  "
          f"f1={ov['mean_f1']:.3f}±{ov['std_f1']:.3f}  "
          f"avg_raw_score={ov['avg_raw_score']:.3f}  avg_per_response_std={ov['avg_per_response_std']:.3f}")
    for crit, m in metrics["by_criterion"].items():
        print(f"  {crit} (n={m['n']}):  "
              f"accuracy={m['mean_accuracy']:.1%}±{m['std_accuracy']:.1%}  "
              f"f1={m['mean_f1']:.3f}±{m['std_f1']:.3f}  "
              f"avg_raw_score={m['avg_raw_score']:.3f}  avg_per_response_std={m['avg_per_response_std']:.3f}")

# --------------------------------------------------------------------------- #
# Joining the blind dataset to its answers
# --------------------------------------------------------------------------- #

def load_answers(path: str) -> dict[str, dict]:
    """`{id: answer record}` from an `eval_answers.json`."""
    with open(path) as f:
        return {a["id"]: a for a in json.load(f)}


def answers_path_for(dataset_path: str) -> Path:
    """Where the answers for `dataset_path` live, by the naming rule
    `tools/build_eval_dataset.py` writes: `eval_dataset.json` -> `eval_answers.json`,
    anything else -> `<stem>_answers.json`."""
    p = Path(dataset_path)
    return (p.with_name("eval_answers.json") if p.name == "eval_dataset.json"
            else p.with_name(f"{p.stem}_answers{p.suffix}"))


def attach_answers(dataset: list[dict], answers: dict[str, dict]) -> list[dict]:
    """Put the label back on each entry, by id.

    **The dataset ships blind and the label ships separately** (see
    `tools/build_eval_dataset.py`), so scoring joins them here rather than reading one file
    that contains both. Nothing downstream of `_score_model` changes: the prompt is built
    from the blind fields only, and this runs after the calls are made -- or before them,
    harmlessly, since `_score_model` reads `prompt`, `response_N`, `criterion_name` and
    `criterion_description` and nothing else.

    **A missing id is an error, not a `None` label.** An entry with no answer would be scored
    against a label of `None`, which `_pred_run` reads as "wrong" -- so a half-joined dataset
    would report as a bad evaluator rather than as a broken join.
    """
    out = []
    missing = []
    for e in dataset:
        a = answers.get(e["id"])
        if a is None:
            missing.append(e["id"])
            continue
        out.append({**e, **{k: v for k, v in a.items() if k != "id"}})
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(dataset)} pair(s) have no answer record "
            f"(first: {missing[0]}). The dataset and the answers file are from different "
            f"builds; re-run tools/build_eval_dataset.py.")
    return out


def load_pairs(dataset_path: str, answers_path: str | None = None) -> list[dict]:
    """The labelled dataset: blind entries joined to their answers. The one loader every
    entry point uses, so no caller can accidentally score a dataset with no labels."""
    ap = Path(answers_path) if answers_path else answers_path_for(dataset_path)
    if not ap.is_file():
        raise SystemExit(f"no answers file at {ap}. tools/build_eval_dataset.py writes it "
                         f"beside the dataset; pass --answers to point elsewhere.")
    return attach_answers(load_dataset(dataset_path), load_answers(str(ap)))
