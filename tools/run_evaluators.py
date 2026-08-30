#!/usr/bin/env python3
"""Score every pair with each evaluator under the score-alone protocol.

    python tools/run_evaluators.py --dataset data/my_run/eval_dataset.json
    python tools/run_evaluators.py --dataset ... --models gpt-oss-20b deepseek-v4-pro
    python tools/run_evaluators.py --dataset ... --report-only   # no calls, no credentials

`--dataset` is the blind file `build_eval_dataset.py` writes. The labels live beside it in
`eval_answers.json` and are joined in `main` after the calls, never before them.

Each evaluator sees one trajectory at a time and scores it alone; the preference is recovered
afterwards from which of a pair's two scores is higher. A deployed evaluator never gets the
counterfactual, so it is not given one here.

**One pool over the whole roster, in shuffled order, resumable.** Every (model, pair, side, run)
is an independent task on one shared pool rather than each model getting its own; see
`evaluate_models`. Consequences:

- Dedicated deployments stay in traffic throughout, so warming happens once up front.
- `--workers` is the whole run's concurrency, not one model's.
- `checkpoint.jsonl` holds every completed call, so a re-run with the same `--out-dir` resumes.

Per evaluator the table reports accuracy, tie rate, and run-to-run stability.

**A tie counts as wrong and is never excluded.** Both trajectories in a pair were steered in
opposite directions on the criterion and both filter judges agreed the difference was visible,
so an evaluator that scores them equal has failed to detect a difference that is there.
Scoring a tie as an abstention would credit it for not doing the task. A call that yields no
parseable score is charged the same way, for the same reason. The tie rate is reported
separately because it distinguishes a coarse score scale from a wrong ranking.

The `kept` / `disc` columns compare the judge1-filtered subset against the rest. On a
`--pool kept` dataset every pair passed judge1 by construction, so `kept` equals the overall
accuracy and `disc` is empty; the comparison needs a `--pool all` dataset.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)

from typing import NamedTuple  # noqa: E402

from src.metaeval.sources.tau2_source import FW, ensure_env, warm  # noqa: E402
from src.metaeval.usage import UsageMeter, pooled_report  # noqa: E402

# --------------------------------------------------------------------------- #
# The roster
# --------------------------------------------------------------------------- #
#
# A row is `Ev(label, model, kind, lane)`. A NamedTuple rather than a bare tuple because the
# shape has now changed twice, and `for label, _model, kind, _w in ROSTER` in another file is
# how the last change broke a downstream reader.
#
# `lane` is the admission-control domain: what this model shares a rate limit with. It is
# stated rather than parsed off the model string, because for half this roster the string
# lies -- `openai/claude-opus-5` is not OpenAI, and a dedicated deployment's limit is its own
# single replica rather than the Fireworks account's token budget. See `provider_of`.
#
# **Reachability is measured, never inferred.** Every row below was called with a real
# 11.5k-character scoring prompt from the benchmark and returned a parseable score. That is
# not a formality: `deepseek-v4-flash` had been in this roster since the 36-pair sweep and
# now 404s outright, and the gateway's own model listing marked two of the rows below as
# SQL-only when both answered REST fine. No listing, flag or model card predicted either.
# **Re-probe before quoting a number** -- one scoring call per row against a real pair is
# enough, and `--models <label> --limit 1 --runs 1` is that probe.


class Ev(NamedTuple):
    label: str
    model: str
    kind: str
    lane: str
    # Per-model litellm overrides, merged over the lane's. **Literals only** -- anything
    # needing a credential belongs in `ENDPOINTS`, which is resolved at run time so a rotated
    # PAT cannot go stale inside a module constant.
    call: dict | None = None


# Lanes.
FW_SERVERLESS = "fw-serverless"     # one account-wide token budget
GEMINI = "gemini"                   # Google's own limits, the strictest here
GATEWAY = "gateway"                 # an OpenAI-compatible gateway in front of several vendors
ANTHROPIC = "anthropic"             # api.anthropic.com direct
OPENAI = "openai-direct"            # api.openai.com direct
# A dedicated deployment gets its own lane, one per deployment: its ceiling is its replica
# count, and it shares that with nobody. `dep:` prefix so `_dedicated_lane` can spot them.
def _dep(name: str) -> str:
    return f"dep:{name}"


GW = "openai"          # litellm provider prefix for an OpenAI-compatible gateway

# **A dedicated Fireworks deployment is addressed by two ids, and both are per-installation:**
# `<model>#accounts/<account>/deployments/<id>`. The account is whoever pays; the deployment id
# is whatever `firectl deployment create` returned. So the account comes from the environment
# and the rows below carry only the local deployment id.
#
# So the rows below are **templates**: they carry the deployment id and a literal
# `ACCOUNT_SLOT` where the account goes, and `resolve()` substitutes the real account at
# startup. Written that way rather than read at import time because the roster is a
# module-level constant: an import-time read bakes in whatever the environment held when the
# module was first touched, which is a different value from what the run is configured with
# whenever the two disagree.
#
# `resolve()` drops these rows -- with the `DEPLOY` recipe -- when `FIREWORKS_ACCOUNT_ID` is
# unset, rather than falling back to the bare serverless name. The fallback would look like it
# worked and 404: none of the three models below is served on Fireworks serverless, which is
# why they were deployed in the first place.
ACCOUNT_SLOT = "<account>"


def _fw_dep(model: str, dep_id: str) -> str:
    return f"{FW}/{model}#accounts/{ACCOUNT_SLOT}/deployments/{dep_id}"

# **The gpt-5 family cannot be run at temperature 0, and that is a caveat rather than a
# setting.** litellm refuses the call outright -- "gpt-5 models don't support temperature=0.0.
# Only temperature=1 is supported" -- and all five GPT rows returned NO DATA on the first live
# smoke run because of it.
#
# The tempting fix is `litellm.drop_params = True`, and it is the wrong one. Dropping the
# parameter hands temperature back to the provider default, which is *exactly* the bug that
# made the judges non-reproducible, and which showed up here as a measurement: gpt-oss-20b
# scored 77.8% on one run and 55.6% across three, with a consistent preference on only 15 of
# 36 pairs -- sampling noise read as evaluator quality. Dropping also leaves no trace, so
# nobody reading the table later would know these rows were not comparable to the others.
#
# So it is set explicitly to the one value the provider allows, per row, and **reported at
# startup and in `summary.json`**. Two consequences when reading results:
#
# - these rows are sampled rather than greedy, so one run is one draw;
# - `--runs k>1` measures real sampling spread for them and residual provider
#   nondeterminism for everything else. Do not pool the stability column across the two.
NEEDS_T1 = {"temperature": 1.0}

ROSTER: list[Ev] = [
    # -- sub-7B active MoE: the class the paper is about -------------------------------
    #
    # **All three of these built the dataset, and there is no clean fourth.** nemotron was
    # judge1 and gpt-oss-120b judge2, so "kept" *means* those two agreed, and all three were
    # also the agents that generated the trajectories (384 / 374 / 367 pairs). Their scores
    # here are partly self-agreement and belong in their own column rather than ranked against
    # the rest.
    #
    # **Nothing in this repository corrects for that, so read these three rows as contaminated
    # and say so wherever they are quoted.** Two distinct mechanisms, neither of them handled:
    # judge-side, because a "kept" pair is one both judges accepted, so both judges are scoring
    # their own admissions; and agent-side, because all three also wrote the trajectories they
    # are now scoring. It would be wrong to describe only the first as the contamination -- a
    # judge-only caveat reads as if gpt-oss-20b, which was never a judge, were clean. The rows
    # are worth running either way; the fix is in how they are reported, and it is manual.
    #
    # **That is now a hole in the roster rather than a caveat on it.** `gemma-4-26b-a4b-it`
    # (25.2B total, 3.8B active) was the one sub-7B-active MoE that was neither judge nor
    # agent, and it is unreachable -- see the retired list below for the four routes tried.
    # So every MoE row in this tier is contaminated, and the nearest clean comparison is
    # `qwen3-4b` at 4.4B *dense*, which changes architecture as well as size. Any claim of the
    # form "small MoE evaluators do X" rests on contaminated rows until a clean one exists;
    # `gemma-4-31b` below is clean but dense and 8x the active parameters.
    Ev("gpt-oss-20b",        f"{FW}/gpt-oss-20b",                    "slm-moe",   FW_SERVERLESS),
    Ev("gpt-oss-120b",       f"{FW}/gpt-oss-120b",                   "slm-moe",   FW_SERVERLESS),
    Ev("nemotron-lightning", f"{FW}/nemotron-lightning-3p5-30b-a3b", "slm-moe",   FW_SERVERLESS),

    # -- small dense --------------------------------------------------------------------
    Ev("qwen3-4b",           _fw_dep("qwen3-4b-instruct-2507",
                                     "meta-eval-qwen3-4b-instruct-2507"), "slm-dense",
       _dep("meta-eval-qwen3-4b-instruct-2507")),
    Ev("qwen3-1p7b",         _fw_dep("qwen3-1p7b", "meta-eval-qwen3-1p7b"), "slm-dense",
       _dep("meta-eval-qwen3-1p7b")),
    # 8B dense: just over the 7B line, and the only Meta model small enough to sit near it.
    # Reachable on the gateway where llama3.3-70b, llama4-scout, ministral-3-8b and
    # mistral-large3 are not -- "unavailable in your region", or simply unknown.
    Ev("llama3.1-8b",        f"{GW}/llama3.1-8b",                    "small-dense", GATEWAY),

    # -- mid open weights ----------------------------------------------------------------
    # This row's published numbers came off a deployment that was already running rather than
    # one created by `DEPLOY`, so the recipe is the intended path and not the path taken.
    Ev("gemma-4-31b",        _fw_dep("gemma-4-31b-it", "meta-eval-gemma-4-31b-it"),
       "mid-open", _dep("meta-eval-gemma-4-31b-it")),
    Ev("llama3.1-70b",       f"{GW}/llama3.1-70b",                   "mid-open",  GATEWAY),
    Ev("llama4-maverick",    f"{GW}/llama4-maverick",                "mid-open",  GATEWAY),

    # -- large open weights --------------------------------------------------------------
    #
    # One model per vendor: two DeepSeeks would measure DeepSeek twice rather than the
    # frontier.
    #
    # **Versions are pinned where the vendor offers a dated tag** (`deepseek-v4-pro-0813`),
    # for the reason the retired `gemini-2.5-flash` comment gave and which still holds: an
    # alias moves, and a paper's numbers have to be re-runnable against the same weights. The
    # unpinned `deepseek-v4-pro` also answers; the dated one is what goes in the table.
    Ev("deepseek-v4-pro",    f"{FW}/deepseek-v4-pro-0813",           "frontier",  FW_SERVERLESS),
    Ev("qwen3p8-max",        f"{FW}/qwen3p8-max",                    "frontier",  FW_SERVERLESS),
    Ev("kimi-k3",            f"{FW}/kimi-k3",                        "frontier",  FW_SERVERLESS),
    Ev("glm-5p2",            f"{FW}/glm-5p2",                        "frontier",  FW_SERVERLESS),
    Ev("nemotron-3-ultra",   f"{FW}/nemotron-3-ultra-nvfp4",         "frontier",  FW_SERVERLESS),
    Ev("mistral-large2",     f"{GW}/mistral-large2",                 "frontier",  GATEWAY),

    # -- closed frontier -----------------------------------------------------------------
    #
    # Google direct; Claude and GPT through the gateway -- see `ENDPOINTS` and `DIRECT_ROUTE`.
    # Google is *not* reachable via the gateway: every gemini and gemma name there returns
    # "unavailable in your region" or "unknown model", so the two lanes are not interchangeable.
    Ev("gemini-3.7-flash",   "gemini/gemini-3.7-flash",              "closed",    GEMINI),
    Ev("gemini-3.5-flash-lite", "gemini/gemini-3.5-flash-lite",      "closed",    GEMINI),
    # The three gpt-5.6 variants are a within-generation spread: they
    # differ in default reasoning (sol 86 of 151 output tokens, luna 68 of 136, terra 0 of 63)
    # with no other axis moving. The `openai-` prefix is the gateway's name for them;
    # `DIRECT_ROUTE` strips it for the vendor-direct path.
    Ev("gpt-5.6-sol",        f"{GW}/openai-gpt-5.6-sol",     "closed", GATEWAY, NEEDS_T1),
    Ev("gpt-5.6-terra",      f"{GW}/openai-gpt-5.6-terra",   "closed", GATEWAY, NEEDS_T1),
    Ev("gpt-5.6-luna",       f"{GW}/openai-gpt-5.6-luna",    "closed", GATEWAY, NEEDS_T1),
    Ev("gpt-5.4-mini",       f"{GW}/openai-gpt-5.4-mini",    "closed", GATEWAY, NEEDS_T1),
    Ev("gpt-5.4-nano",       f"{GW}/openai-gpt-5.4-nano",    "closed", GATEWAY, NEEDS_T1),
    Ev("claude-opus-5",      f"{GW}/claude-opus-5",                  "closed",    GATEWAY),
    Ev("claude-sonnet-5",    f"{GW}/claude-sonnet-5",                "closed",    GATEWAY),
    Ev("claude-haiku-4-5",   f"{GW}/claude-haiku-4-5",               "closed",    GATEWAY),
]

# The three dedicated rows need a deployment that does not exist by default. This recipe is
# printed rather than run: creating one is a manual step. All three models report
# `World Size: 1`, so one GPU serves each.
DEPLOY = """\
firectl deployment create accounts/fireworks/models/{model} \\
  --deployment-id  meta-eval-{model} \\
  --display-name   meta-eval-{model} \\
  --accelerator-type NVIDIA_H100_80GB --accelerator-count 1 \\
  --precision FP8 \\
  --min-replica-count 1 --max-replica-count 1 \\
  --scale-to-zero-window 5m"""


# Calls in flight to one lane, whatever the pool size. A guard rail, not the throughput knob
# -- the pool is that. Absent an entry, a lane is bounded only by `--workers`.
#
# - **gemini** is the strictest lane here and Google's limits are per-project, so its two
#   models share one small budget.
# - **fw-serverless** is left uncapped: its binding limit is tokens/sec across the whole
#   account, and eight workers spread over eight of its models is well inside what four
#   workers on a *single* model already sustained.
# - **each dedicated deployment** is capped at 2 by `_lane_caps`, not here. One replica cannot
#   serve more than a couple of 7k-token prompts at once, and extra workers queueing on it are
#   workers not spending time on the other 22 models.
#
# **The gateway is 10, and that number is a throughput calculation rather than a guess.**
# Twelve of the 25 rows sit on this lane -- 24,000 calls at ~8s observed mean latency -- so the
# lane's wall clock is `24000 * 8 / cap` seconds: 8.9 hours at a cap of 6, which is why the
# first full main pass took 6h59m. At 10 it is ~5.3h, and that is what makes the sweep finish
# in a working day. The 429s a higher cap provokes are absorbed by litellm's internal retries
# (see `_gateway_call`) and swept up by the retry passes, which run 4-wide rather than serially
# -- so the cost of raising it is a few more retried calls, not lost coverage.
#
# anthropic and openai-direct are 4, which is a guess rather than a measurement: neither lane
# carried a call in the published sweep. Raise it for an account of known tier.
PROVIDER_CAPS: dict[str, int] = {GEMINI: 3, GATEWAY: 10, ANTHROPIC: 4, OPENAI: 4}
DEDICATED_CAP = 2

# Requests per minute per lane, for a provider that states a rate rather than a concurrency.
# A concurrency cap cannot express a rate: 8 in flight is 2.8 starts/s at 2.9s mean latency and
# 8/s if the provider answers in one second. The gateway states 60/min and enforces it.
LANE_RATES: dict[str, int] = {GATEWAY: 60}


# --------------------------------------------------------------------------- #
# The gateway, and the two vendor-direct lanes that stand in for it
# --------------------------------------------------------------------------- #
#
# **Twelve of the 25 rows -- every Claude, every GPT, all three Llamas, Mistral -- reached the
# published sweep through one OpenAI-compatible gateway rather than through their vendors'
# own APIs.** That is a property of the serving path, not of the method, and it is
# stated here rather than hidden because it is the single biggest obstacle to reproducing the
# table: the gateway is a site-specific deployment, not a public endpoint.
#
# So this file supports both, and `resolve()` picks:
#
# - **the gateway**, when `GATEWAY_BASE_URL` and `GATEWAY_API_KEY` are set. One code path
#   covers every vendor on it, because it serves a plain `POST /v1/chat/completions` and
#   every model is addressed as `openai/<model>` with an `api_base`.
# - **the vendor's own API**, when the gateway is unset but the vendor key is present. This is
#   the reproducible route and the one an outside reader will take; `DIRECT_ROUTE` holds the
#   name translation. **It is not what produced the published numbers** -- a vendor endpoint
#   and a gateway in front of it are not guaranteed to be the same weights, the same system
#   prompt handling or the same sampling defaults, so treat a direct-route rerun as a
#   replication rather than as a check of arithmetic.
# - **neither**, for the four open-weight rows the gateway served under names with no
#   published vendor equivalent (`llama3.1-8b`, `llama3.1-70b`, `llama4-maverick`,
#   `mistral-large2`). They are dropped with a line saying so. Guessing a Fireworks or
#   Mistral slug for them would produce a row that answers and is not the model that was
#   measured, which is worse than a gap.
#
# **Two parameter facts about the gateway, both measured, both 400s if you get them wrong:**
#
# 1. It rejects `max_tokens` outright -- "max_tokens is deprecated in favor of
#    max_completion_tokens". litellm rewrites that automatically for model names it recognises
#    as OpenAI reasoning models, which is why `openai-gpt-5.4` worked on the first try and
#    every `claude-*` name failed. So the swap is done here, explicitly, for all of them.
# 2. Sending both is also a 400, hence `"max_tokens": None` -- `_score_model` drops a key
#    mapped to None rather than passing it through.
#
# There was also an Anthropic-native path on the same gateway, and it is not used: it works
# for Claude and **only** Claude -- a GPT model there returns a bare `internal error`. The
# OpenAI-compatible path serves both families. `scoring.py` pops `ANTHROPIC_BASE_URL` for the
# same reason: a stray export must not silently redirect an `anthropic/` call mid-sweep.


def _gateway_call() -> dict:
    """Per-call kwargs for the gateway, or raise with the fix if it is not configured.

    Read from the environment at run time rather than baked into `ROSTER`, because a gateway
    credential is usually short-lived and a literal in a module would go stale between
    sessions.
    """
    base = os.environ.get("GATEWAY_BASE_URL")
    token = os.environ.get("GATEWAY_API_KEY")
    if not (base and token):
        raise SystemExit(
            "gateway rows are in the roster but GATEWAY_BASE_URL / GATEWAY_API_KEY are not "
            "set.\n"
            "  Set both, or unset them and let `resolve()` reroute the Claude and GPT rows to\n"
            "  ANTHROPIC_API_KEY / OPENAI_API_KEY, or exclude the rows:  --models <labels>")
    # Some gateways attribute usage to a header rather than to the credential. Both the header
    # name and the value are per-installation, so both are optional and neither is invented.
    user = os.environ.get("GATEWAY_USER", "")
    header = os.environ.get("GATEWAY_USER_HEADER", "X-Gateway-User")
    headers = {header: user} if user else {}
    return {"api_base": base, "api_key": token, "extra_headers": headers,
            "max_completion_tokens": 10_000, "max_tokens": None,
            # **Internal retries stay, and this was settled by measurement after two wrong
            # theories.** `RateLimiter` counts calls it *starts*; litellm's `num_retries` fires
            # up to four more HTTP requests inside one start, invisibly. That looked like a
            # feedback loop worth breaking -- 57 starts/min is ~285 req/min at five attempts
            # each, right on the gateway's stated 300 -- so `num_retries: 0` was tried with the
            # smoothed limiter, the one combination not yet tested.
            #
            # It measured **79 req/min against that 300 limit and still drew 111 rejections**,
            # which refutes the arithmetic: at a quarter of the stated limit, with the
            # client-side count now provably equal to the server-side count, the gateway still
            # rejected. Whatever "300 req/min" describes, it is not what binds here -- likely
            # upstream load surfacing through the gateway's own error string.
            #
            # Head to head on real calls, same limiter, same pool:
            #
            #     num_retries: 4  ->   22 failures of 480   (4.6%)
            #     num_retries: 0  ->  149 failures of 600  (24.8%)
            #
            # So the internal retries are the effective mitigation, not the cause. Kept, and
            # the rate limiter stays as a ceiling rather than as the fix.
            "num_retries": 4}


# `lane -> extra litellm kwargs`, merged over `SCORE_CALL` per call. A lane with no entry calls
# the provider the plain way, which is every Fireworks and Gemini row -- and both vendor-direct
# lanes, where litellm reads `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` itself.
ENDPOINTS: dict[str, callable] = {GATEWAY: _gateway_call}

# `label -> (model, lane)` for the vendor's own API, used when the gateway is not configured.
#
# **The gateway renames things, and the rename is the whole content of this table.** Its GPT
# names carry an `openai-` prefix that the vendor's API does not know, and its Claude names are
# undated aliases where Anthropic pins a date. Only rows with a public equivalent appear; the
# four open-weight rows the gateway served are deliberately absent (see above).
DIRECT_ROUTE: dict[str, tuple[str, str]] = {
    "gpt-5.6-sol":     ("openai/gpt-5.6-sol",             OPENAI),
    "gpt-5.6-terra":   ("openai/gpt-5.6-terra",           OPENAI),
    "gpt-5.6-luna":    ("openai/gpt-5.6-luna",            OPENAI),
    "gpt-5.4-mini":    ("openai/gpt-5.4-mini",            OPENAI),
    "gpt-5.4-nano":    ("openai/gpt-5.4-nano",            OPENAI),
    "claude-opus-5":   ("anthropic/claude-opus-5",        ANTHROPIC),
    "claude-sonnet-5": ("anthropic/claude-sonnet-5",      ANTHROPIC),
    "claude-haiku-4-5": ("anthropic/claude-haiku-4-5-20251001", ANTHROPIC),
}

# Which environment variable makes a lane callable at all. `FW_SERVERLESS` is checked against
# the name `ensure_env` copies into litellm's variable rather than the one `.env` carries.
LANE_KEYS: dict[str, tuple[str, ...]] = {
    FW_SERVERLESS: ("FIREWORKS_AI_API_KEY", "FIREWORKS_API_KEY"),
    GEMINI: ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ANTHROPIC: ("ANTHROPIC_API_KEY",),
    OPENAI: ("OPENAI_API_KEY",),
}


def _has(names: tuple[str, ...]) -> bool:
    return any(os.environ.get(n) for n in names)


def resolve(roster: list[Ev]) -> tuple[list[Ev], list[str]]:
    """`(runnable, notes)` -- the roster this installation can actually call, and why not.

    **Every row is decided before the first call, and every dropped row is named.** A sweep
    that silently omits eight of 25 evaluators produces a table that looks complete, and the
    labels are the paper's rows -- so the omission has to be as loud as the run itself. Same
    argument as `warm`'s cold-deployment skip, one stage earlier.

    Rerouting is one-way on purpose: the gateway wins when it is configured, because that is
    what produced the published numbers. Only when it is absent does a Claude or GPT row fall
    back to its vendor -- and `notes` says so, since the two are not interchangeable.
    """
    gateway = _has(("GATEWAY_BASE_URL",)) and _has(("GATEWAY_API_KEY",))
    out, notes = [], []
    for r in roster:
        if r.lane == GATEWAY and not gateway:
            route = DIRECT_ROUTE.get(r.label)
            if not route:
                notes.append(f"{r.label}: DROPPED -- gateway-only name, and no published "
                             f"vendor equivalent to substitute")
                continue
            model, lane = route
            if not _has(LANE_KEYS[lane]):
                notes.append(f"{r.label}: DROPPED -- no gateway, and "
                             f"{LANE_KEYS[lane][0]} is not set")
                continue
            notes.append(f"{r.label}: {model} direct, NOT via the gateway the published "
                         f"numbers used")
            out.append(r._replace(model=model, lane=lane))
            continue
        if r.lane.startswith("dep:"):
            account = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
            if not account:
                notes.append(f"{r.label}: DROPPED -- FIREWORKS_ACCOUNT_ID is not set, so its "
                             f"dedicated deployment cannot be addressed (see DEPLOY)")
                continue
            out.append(r._replace(model=r.model.replace(ACCOUNT_SLOT, account)))
            continue
        keys = LANE_KEYS.get(r.lane)
        if keys and not _has(keys):
            notes.append(f"{r.label}: DROPPED -- {keys[0]} is not set")
            continue
        out.append(r)
    return out, notes


def _lane_rates(roster: list[Ev]) -> dict[str, int]:
    """`LANE_RATES` narrowed to the lanes in this run, so a Fireworks-only sweep is unthrottled."""
    return {lane: n for lane, n in LANE_RATES.items() if any(r.lane == lane for r in roster)}


def _lane_caps(roster: list[Ev]) -> dict[str, int]:
    """`PROVIDER_CAPS` plus one cap per dedicated deployment actually in this run.

    Built from the roster rather than written out, so adding a deployment row cannot forget
    to cap it -- and so a `--models gpt-oss-20b` run does not carry caps for lanes it never
    touches.
    """
    caps = {lane: n for lane, n in PROVIDER_CAPS.items()
            if any(r.lane == lane for r in roster)}
    caps.update({r.lane: DEDICATED_CAP for r in roster if r.lane.startswith("dep:")})
    return caps

# Callable but unused, so the roster choice is auditable rather than merely current:
# deepseek-v4-pro (unpinned), deepseek-v4-flash-0731, gemini-3.1-pro-preview, gemini-2.5-pro,
# gemini-flash-latest, kimi-k2p6, kimi-k2p7-code, minimax-m3, minimax-m2p7.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="eval_dataset.json from tools/build_eval_dataset.py -- the blind "
                         "file, with no labels in it")
    ap.add_argument("--answers", default=None,
                    help="the labels. Default: eval_answers.json beside --dataset. Scoring "
                         "does not read them; accuracy is computed after the calls.")
    ap.add_argument("--votes", default=None,
                    help="judge_votes.json, for the judge1-filtered subset column. "
                         "Default: beside --dataset; skipped if absent.")
    # **Defaults are derived from --dataset, never named here.** A literal run path in an
    # argparse default is how the old version of this file pointed at a run directory that no
    # longer exists, and a stale default is worse than a required argument: it starts, reads
    # nothing, and reports an empty table.
    ap.add_argument("--out-dir", default=None,
                    help="default: <dataset dir>/evaluations")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--runs", "-k", type=int, default=3)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the table from saved *.results.json, calling nothing")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N pairs. For smoke-testing the harness on a "
                         "handful of real calls -- NOT for reporting, since it takes the "
                         "first N in file order rather than a stratified sample.")
    ap.add_argument("--workers", "-w", type=int, default=8,
                    help="calls in flight across the WHOLE roster (default 8). Not per "
                         "model: 8 workers over 9 models is ~1 call per model, which is "
                         "why this is higher than the 4 the per-model loop used.")
    ap.add_argument("--seed", type=int, default=0,
                    help="task shuffle seed. The order is randomised so no provider gets a "
                         "burst of same-model calls; seeded so a run is re-creatable.")
    ap.add_argument("--give-up-after", type=int, default=20,
                    help="abandon a model after this many failed calls with no successes "
                         "(0 disables). Stops one unreachable model from spending hours of "
                         "a multi-hour run on calls that answer nothing.")
    ap.add_argument("--retry-passes", type=int, default=3,
                    help="coordinated retry passes over failed calls (default 3). 0 writes "
                         "results straight from what the main pass scored, which is what you "
                         "want when the residue is known-unrecoverable -- nothing is written "
                         "until the passes finish.")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore checkpoint.jsonl and re-score everything. The checkpoint is "
                         "still appended to, so this costs calls rather than history.")
    args = ap.parse_args()

    ensure_env()
    from src.metaeval.scoring import (
        SCORE_TEMPERATURE, Checkpoint, _score_model, compute_metrics, evaluate_models,
        load_pairs, save_json)

    # Blind file plus answers file, joined here. `load_pairs` refuses a partial join rather
    # than scoring an entry against a label of None, which `_pred_run` would read as a wrong
    # prediction -- a broken join must not be reportable as a bad evaluator.
    dataset = load_pairs(args.dataset, args.answers)
    smoke = bool(args.limit)
    if smoke:
        dataset = dataset[:args.limit]
        print(f"*** --limit {args.limit}: SMOKE TEST on {len(dataset)} pairs, "
              f"not a measurement ***")
    roster = ROSTER if not args.models else \
        [r for r in ROSTER if r.label in set(args.models)]
    if not roster:
        print(f"no model matched {args.models}; have {[r.label for r in ROSTER]}")
        return 2
    # **`--report-only` must not require credentials.** It calls nothing -- it rebuilds the
    # table from saved `*.results.json` -- so the reader with the published artifacts and no
    # keys at all is exactly the reader it exists for. `resolve()` would drop every row for
    # want of a credential and return 2 below, before the report branch was ever reached.
    # It also reports on the *unresolved* roster on purpose: a saved result should appear in
    # the table whether or not that model happens to be reachable right now.
    if args.report_only:
        notes, lane_call, caps, rates = [], {}, {}, {}
    else:
        # **Which rows this installation can call, decided before the first call and printed.**
        # `--models` asked for labels; `resolve` says which of them are reachable with the
        # credentials present, and reroutes the gateway rows to their vendors when there is no
        # gateway. Every line it returns is either a reroute that changes what a number means
        # or a row missing from the table, so none of them is a debug detail.
        roster, notes = resolve(roster)
        if notes:
            print(f"roster: {len(notes)} row(s) rerouted or dropped")
            for n in notes:
                print(f"  {n}")
            print()
        if not roster:
            print("no row in the roster is reachable with the credentials present; "
                  "nothing to run")
            return 2
        # Resolved once, before any call, so a missing credential is a startup error with the
        # fix in it rather than 2,000 identical auth failures discovered an hour in.
        lane_call = {lane: fn() for lane, fn in ENDPOINTS.items()
                     if any(r.lane == lane for r in roster)}
        caps = _lane_caps(roster)
        rates = _lane_rates(roster)

    # The judge1-filtered subset, so "is the filtered set still hard" is answerable.
    kept: set[str] = set()
    vp = Path(args.votes) if args.votes else Path(args.dataset).with_name("judge_votes.json")
    if vp.is_file():
        intent = {e["id"]: e["correct_response"] for e in dataset}
        kept = {v["id"] for v in json.loads(vp.read_text())["votes"]
                if v["judge"] == "judge1" and v["side"] == intent.get(v["id"])}

    out = Path(args.out_dir or (Path(args.dataset).parent / "evaluations"))
    # **A smoke run gets its own directory.** The comment here used to *claim* this and the
    # code never did it, which is the worst of both: a `--limit 3` run wrote `summary.json`,
    # `{label}.results.json` and `{label}.usage.json` straight over the real 36-pair sweep,
    # leaving a 3-pair table indistinguishable from a 36-pair one. The merge-by-label added
    # later made it worse rather than better -- the 3-pair row became *persistent*, carried
    # forward past later full runs under a reassuring "carried N row(s)" line.
    #
    # `--out-dir` given explicitly is respected as-is; only the default is redirected, so a
    # deliberate smoke target still works.
    # A smoke run gets its own directory -- but only when the directory was derived. An
    # explicit `--out-dir` is respected as given, so a deliberate smoke target still works.
    if smoke and args.out_dir is None:
        out = out / f"smoke-{args.limit}"
        print(f"    writing to {out} so the real sweep is not overwritten")
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        print(f"--report-only: rebuilding the table from saved results in {out}, "
              f"calling nothing")
        print(f"{len(dataset)} pairs; judge1-filtered subset: {len(kept)} pairs")
        print()
    else:
        print(f"{len(dataset)} pairs x 2 responses x {args.runs} run(s) "
              f"x {len(roster)} evaluator(s) = "
              f"{len(dataset) * 2 * args.runs * len(roster)} calls")
        print(f"judge1-filtered subset: {len(kept)} pairs")
        print(f"pool: {args.workers} worker(s) across the roster, shuffled with seed "
              f"{args.seed}")
        per_lane = collections.Counter(r.lane for r in roster)
        print("lanes: " + ", ".join(
            f"{lane} x{n}"
            + (f" (cap {caps[lane]}" if lane in caps else " (uncapped")
            + (f", {rates[lane]}/min)" if lane in rates else ")")
            for lane, n in sorted(per_lane.items())))
        # **Printed, never silent.** A row not at temperature 0 is sampled rather than greedy,
        # so one run is one draw and its stability column measures something different from
        # every other row's. `litellm.drop_params` would have hidden exactly this.
        sampled = [r.label for r in roster
                   if (r.call or {}).get("temperature") not in (None, 0.0)]
        if sampled:
            print(f"*** {len(sampled)} row(s) NOT at temperature 0, because the provider "
                  f"refuses it: {', '.join(sampled)}")
            print(f"    They are sampled, not greedy. Do not pool their stability column "
                  f"with the rest.")
        print()

    if args.report_only:
        summary = []
        for row in roster:
            label, kind = row.label, row.kind
            f = out / f"{label}.results.json"
            if not f.is_file():
                continue
            row = _row(label, kind, json.loads(f.read_text()), kept, 0.0,
                       n_expected=len(dataset))
            # Usage is reloaded rather than recomputed -- it cannot be derived from the
            # results file, which holds scores and no token counts. A run predating this
            # instrumentation has no usage file, and the table simply omits it.
            uf = out / f"{label}.usage.json"
            if uf.is_file():
                row["usage"] = json.loads(uf.read_text()).get("summary")
            summary.append(row)
        if not summary:
            print(f"no saved results in {out}")
            return 2
        # `--report-only` makes no calls, so it has no clock of its own; the run's is in
        # `usage.json` next to the flag saying which arithmetic wrote it.
        uf = out / "usage.json"
        pw = None
        if uf.is_file():
            blob = json.loads(uf.read_text())
            if blob.get("pooled"):
                pw = (blob.get("total") or {}).get("wall_s")
        _print_table(summary, len(dataset), len(kept))
        _print_usage(summary, pool_wall=pw)
        return 0

    # **Warm every dedicated deployment once, before the pool.** The old code warmed each
    # model at its own turn, correctly, because a serial sweep took ~30 minutes and a
    # deployment warmed at minute 0 has scaled to zero by minute 20 -- which is how qwen3-4b
    # and qwen3-1p7b each scored 0 of 36 entries. The pool removes the premise: with tasks
    # interleaved there are no turns, every deployment is in traffic continuously from the
    # first second, and warming has to happen before any of them can be needed.
    #
    # A deployment that will not come up is dropped from the roster rather than aborting the
    # run: the other eight models are unaffected, and losing one row beats losing the sweep.
    dedicated = [r.model for r in roster if "#" in r.model]
    if dedicated:
        print(f"pre-warming {len(dedicated)} dedicated deployment(s):")
        # verbose=True: `warm` prints the provider's actual error, and that line is the only
        # diagnostic on this path. Silencing it cost a real one -- qwen3-1p7b was skipped and
        # then found with Ready Replica Count: 1 moments later, with no record of what the
        # failing call had returned.
        cold = set(warm(dedicated, verbose=True))
    else:
        cold = set()

    summary = []
    live = []
    for r in roster:
        if r.model in cold:
            print(f"  {r.label}: SKIPPED, deployment did not come up (see the error above)")
            summary.append({"label": r.label, "kind": r.kind,
                            "error": "deployment did not come up"})
        else:
            live.append(r)
    if not live:
        print("no evaluator is reachable; nothing to run")
        return 2

    # **One meter for the whole run, not one per model.** `merge()` sums per-meter wall
    # clocks, which was right for a serial sweep and is wrong here: nine meters each
    # reporting the full pooled window would report nine times the elapsed time and deflate
    # every derived rate ninefold. `pooled_report` splits one real window by label instead --
    # see its docstring for what `concurrency` then means.
    meter = UsageMeter().start()
    ck = Checkpoint(out / "checkpoint.jsonl")
    if args.no_resume:
        prior_done, _ = ck.load()
        if prior_done:
            print(f"--no-resume: ignoring {len(prior_done):,} checkpointed call(s); they "
                  f"will be paid for again\n")

    models = [(r.label, r.model) for r in live]
    by_label = {r.label: r for r in live}

    def score(label: str, entry: dict, key: str) -> float:
        row = by_label[label]
        # Lane first, row second: a per-model override wins over its lane's, which is how
        # `NEEDS_T1` overrides the gateway's temperature without touching the other ten rows
        # that share the gateway.
        return _score_model(entry, key, row.model, meter=meter, label=label,
                            extra={**lane_call.get(row.lane, {}), **(row.call or {})})

    try:
        per_label = evaluate_models(
            dataset, models, score,
            num_runs=args.runs, max_workers=args.workers, seed=args.seed,
            checkpoint=ck, resume=not args.no_resume, retry_passes=args.retry_passes,
            provider_caps=caps, lane_rates=rates,
            lane_of=lambda label: by_label[label].lane,
            last_record=meter.last_record, give_up_after=args.give_up_after)
    except KeyboardInterrupt:
        # Every completed call is already in checkpoint.jsonl, so the cost of a Ctrl-C is the
        # calls in flight. Re-running the same command resumes. Reported rather than silent,
        # because the alternative reading -- that the run finished -- is expensive to act on.
        meter.stop()
        print(f"\ninterrupted. {meter.summary()['n_calls']:,} call(s) this invocation are in "
              f"{out / 'checkpoint.jsonl'}; re-run the same command to resume.")
        return 130
    wall = meter.stop().wall_s()

    usage_by_label = meter.by("label")
    for row in live:
        label, kind = row.label, row.kind
        results = per_label.get(label, [])
        u = usage_by_label.get(label)
        save_json(results, str(out / f"{label}.results.json"))
        metrics = compute_metrics(results)
        save_json(metrics, str(out / f"{label}.metrics.json"))
        if u:
            save_json({"summary": u}, str(out / f"{label}.usage.json"))

        row = _row(label, kind, results, kept, wall, n_expected=len(dataset))
        row["usage"] = u
        # The temperature this row actually ran at. `summary.json` outlives the console, and
        # "accuracy 61%" means a different thing at 1.0 than at 0.0.
        row["temperature"] = (by_label[label].call or {}).get("temperature", SCORE_TEMPERATURE)
        row["lane"] = by_label[label].lane
        summary.append(row)
        print(f"--- {label} ({kind}) ---")
        if row["accuracy"] is None:
            print(f"  NO DATA -- 0 of {len(dataset)} entries scored, every call failed")
        else:
            miss = (f"  [{row['n_errors']} call(s) counted wrong]"
                    if row.get("n_errors") else "")
            print(f"  accuracy {row['accuracy']:.1%}  ties {row['tie_rate']:.1%}  "
                  f"errors {row['error_rate']:.1%}  stability {row['stability']}{miss}")
        if not u:
            # A resumed run whose every call was already on disk makes no calls, so it has
            # no usage this invocation. Saying so beats printing zeros that read as "free".
            print(f"  no calls this invocation (all resumed from checkpoint)\n")
            continue
        print(f"  {u['n_calls']} calls ({u['n_error']} failed), "
              f"{u['total_tokens']:,} tok = {u['prompt_tokens']:,} in + "
              f"{u['completion_tokens']:,} out, of which "
              f"{u['reasoning_tokens']:,} reasoning ({(u['reasoning_share'] or 0):.1%})")
        print(f"  {u['out_tps']} out tok/s, {u['concurrency']} of the "
              f"{args.workers}-worker pool\n")
    print(f"pool wall clock: {wall:.0f}s for {meter.summary()['n_calls']:,} call(s)\n")

    # Merge with any rows already on disk, keyed by label. A `--models qwen3-1p7b` re-run
    # used to rewrite this file wholesale, leaving a roster-wide summary holding exactly one
    # row -- which is what was on disk, and what made the cost report show one evaluator out
    # of nine. Same rule as judge votes: a targeted re-run must not delete other models'
    # results. The per-model *.results.json files were never affected, which is why
    # --report-only could still rebuild the full table.
    sp = out / "summary.json"
    if sp.is_file():
        try:
            prior = {r["label"]: r for r in json.loads(sp.read_text())
                     if isinstance(r, dict) and r.get("label")}
        except (json.JSONDecodeError, OSError):
            prior = {}
        fresh = {r["label"] for r in summary if r.get("label")}
        carried = [r for lab, r in sorted(prior.items()) if lab not in fresh]
        if carried:
            print(f"\ncarried {len(carried)} row(s) for models not in this run: "
                  f"{', '.join(r['label'] for r in carried)}")
        summary = summary + carried
    save_json(summary, str(sp))
    # `pooled_report(meter)`, not `merge(per_model_meters)`. Same `{"total", "per_model"}`
    # shape, so a reader of either needs no special case, but the wall clock is the run's real
    # elapsed window instead of a sum over nine overlapping ones -- see `pooled_report`.
    if meter.records:
        save_json(pooled_report(meter), str(out / "usage.json"))
    _print_table(summary, len(dataset), len(kept))
    _print_usage(summary, pool_wall=wall)
    return 0


def _row(label: str, kind: str, results: list[dict], kept: set, wall: float,
         n_expected: int = 0) -> dict:
    """One summary row.

    `evaluate()` drops an entry entirely when a scoring call raises, so a model whose every
    call failed comes back with an empty result list. Reporting that as "accuracy 0.0%" is
    a lie that reads as a terrible evaluator, and it happened: qwen3-4b and qwen3-1p7b each
    scored 0 of 36 entries after their deployments scaled to zero mid-run, and the table
    said 0.0% accuracy with 0.0% errors. `n_scored` is therefore reported next to every
    number, and accuracy is None rather than 0.0 when nothing was scored.
    """
    runs = [(r, run) for r in results for run in r.get("runs", [])]
    n = len(runs)
    ties = sum(1 for _, run in runs if run.get("higher_score") is None
               and run.get("score_1") is not None)
    errs = sum(1 for _, run in runs if run.get("score_1") is None)
    hits = sum(1 for _, run in runs if run.get("correct"))

    def acc(subset: set | None) -> tuple[int, int]:
        sel = [(r, run) for r, run in runs if subset is None or r["id"] in subset]
        return sum(1 for _, run in sel if run.get("correct")), len(sel)

    # Self-consistency across runs, decomposed -- a single ratio hid two different things.
    #
    # A pair that tied in **every** run used to count as stable, because all its picks were
    # None and one distinct value reads as agreement. gpt-oss-20b's perfect 36/36 included 7
    # such pairs: it was credited for reliably failing to discriminate. Ties are wrong, so
    # an all-tie pair cannot be a stable success.
    #
    # The split also turns out to be the more useful number. Nearly all instability is
    # tie<->preference rather than 1<->2: deepseek-v4-flash wobbled that way on 15 pairs and
    # actually reversed direction on 1. It is not changing its mind about which side is
    # better, it is sitting at the resolution limit of its own score scale, where trivial
    # nondeterminism decides whether two near-identical scores come out equal.
    stable = flipped = wobbled = always_tied = 0
    for r in results:
        picks = [run.get("higher_score") for run in r.get("runs", [])]
        if len(picks) < 2:
            continue
        seen = set(picks)
        if seen == {None}:
            always_tied += 1                      # consistent, but consistently failed
        elif len(seen) == 1:
            stable += 1                           # same real preference every run
        elif None in seen:
            wobbled += 1                          # tie <-> preference
        else:
            flipped += 1                          # genuine 1 <-> 2 reversal
    n_multi = stable + flipped + wobbled + always_tied
    all_hit, all_n = acc(None)
    k_hit, k_n = acc(kept) if kept else (0, 0)
    d_hit, d_n = acc({r["id"] for r in results} - kept) if kept else (0, 0)
    scores = [s for _, run in runs for s in (run.get("score_1"), run.get("score_2"))
              if s is not None]
    return {
        "label": label, "kind": kind, "wall_s": round(wall, 1),
        "n_scored": len(results), "n_expected": n_expected,
        # Calls that never produced a score, counted as wrong rather than dropped -- see
        # `_assemble`. This replaced the `n` column, which is now the dataset size for every
        # model and so said nothing.
        "n_errors": errs,
        "accuracy": hits / n if n else None,
        "tie_rate": ties / n if n else None,
        "error_rate": errs / n if n else None,
        "stability": (f"{stable}/{n_multi}" if n_multi else "-"),
        "always_tied": always_tied, "wobbled": wobbled, "flipped": flipped,
        "acc_kept": k_hit / k_n if k_n else None,
        "acc_discarded": d_hit / d_n if d_n else None,
        "n_kept": k_n, "n_discarded": d_n,
        "distinct_scores": len(set(scores)),
        "score_sd": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
    }


def _print_usage(summary: list[dict], pool_wall: float | None = None) -> None:
    """Cost and time, per evaluator.

    Kept as its own table rather than more columns on the accuracy one, because the two
    answer different questions and get read at different times. The ordering is by spend,
    not accuracy: this table exists to show what the sweep cost and where the wall clock
    went, and the headline is that those two rankings do not match.

    **`pool_wall` is the one number that cannot be summed.** Every row shares the pool's
    single elapsed window, so adding the `wall` column down the table multiplies the run's
    real duration by the number of models: the first live pooled run printed 59s for a sweep
    that took 20s. Tokens and calls still sum -- they are per-call facts -- and `svc` sums by
    definition, which is why the ratio of the two is worth printing at the bottom.
    """
    rows = [r for r in summary if r.get("usage")]
    if not rows:
        return
    print("\n" + "=" * 96)
    print("  TOKENS AND TIME")
    print("=" * 96)
    print(f"  {'evaluator':20s} {'calls':>6s} {'fail':>5s} {'in tok':>9s} {'out tok':>9s} "
          f"{'reason':>9s} {'reason%':>8s} {'tok/call':>8s} {'wall':>7s} {'svc':>7s} "
          f"{'conc':>5s} {'tok/s':>7s} {'p95':>6s}")
    for r in sorted(rows, key=lambda x: -(x["usage"].get("total_tokens") or 0)):
        u = r["usage"]
        print(f"  {r['label']:20s} {u['n_calls']:6d} {u['n_error']:5d} "
              f"{u['prompt_tokens']:9,d} {u['completion_tokens']:9,d} "
              f"{u.get('reasoning_tokens', 0):9,d} "
              f"{(u.get('reasoning_share') or 0):7.1%} "
              f"{(u['tokens_per_ok_call'] or 0):8,.0f} "
              f"{u['wall_s']:6.0f}s {u['service_s']:6.0f}s "
              f"{(u['concurrency'] or 0):5.1f} {(u['out_tps'] or 0):7.1f} "
              f"{(u['latency_p95_s'] or 0):5.1f}s")
    tot = {k: sum((r["usage"].get(k) or 0) for r in rows)
           for k in ("n_calls", "n_error", "prompt_tokens", "completion_tokens",
                     "reasoning_tokens", "service_s")}
    # The run's own clock when the sweep was pooled, and the sum of per-model windows only
    # when it genuinely ran model-by-model.
    wall = pool_wall if pool_wall is not None else sum(
        (r["usage"].get("wall_s") or 0) for r in rows)
    share = tot["reasoning_tokens"] / tot["completion_tokens"] if tot["completion_tokens"] else 0
    print(f"  {'TOTAL':20s} {tot['n_calls']:6.0f} {tot['n_error']:5.0f} "
          f"{tot['prompt_tokens']:9,.0f} {tot['completion_tokens']:9,.0f} "
          f"{tot['reasoning_tokens']:9,.0f} {share:7.1%} {'':8s} "
          f"{wall:6.0f}s {tot['service_s']:6.0f}s "
          f"{(tot['service_s'] / wall if wall else 0):5.1f}")
    srcs = {r["usage"].get("reasoning_source") for r in rows}
    print("\n  in/out tok = billed prompt and completion tokens, summed over every call")
    print("  reason     = output tokens spent on reasoning the model never showed you")
    print("  reason%    = reason / out tok. **The runaway detector.** A jump here with flat")
    print("               accuracy means you are paying more for the same answer.")
    if srcs == {"derived"}:
        print("               DERIVED: Fireworks reports no reasoning breakdown, so this is")
        print("               the completion split by the character ratio between the")
        print("               reasoning trace and the answer. Accurate to a few tokens;")
        print("               biased slightly high, since JSON has fewer chars per token")
        print("               than prose.")
    print("  fail       = calls that raised. They still cost time, and often still bill,")
    print("               so they are counted here rather than dropped.")
    if pool_wall is not None:
        print("  wall       = the POOL's elapsed time. Every row shares it, so the column does")
        print("               not sum -- the TOTAL is that one window, not the sum of the rows.")
        print("  conc       = svc / wall: this model's average share of the pool's workers.")
        print("               The rows sum to the pool's achieved concurrency, on the TOTAL")
        print("               line. A model far below its 1/N share is one the pool is")
        print("               waiting on rarely; far above, and it is absorbing the pool.")
    else:
        print("  wall       = elapsed real time; svc = summed per-call latency")
        print("  conc       = svc / wall, the concurrency actually achieved. 1.0 means serial")
        print("               -- either by configuration or because rate limits serialised us.")
    print("  tok/s      = output tokens per elapsed second. This is what provokes rate")
    print("               limits, and it is not predicted by model size.")


def _print_table(summary: list[dict], n_pairs: int, n_kept: int) -> None:
    print("=" * 96)
    print("  EVALUATORS ON THE PAIRWISE BENCHMARK")
    print("=" * 96)
    print(f"  {'evaluator':20s} {'kind':10s} {'acc':>6s} {'ties':>6s} {'errs':>6s} "
          f"{'stable':>7s} {'tie3':>4s} {'wob':>4s} {'flip':>4s} {'lvls':>5s} "
          f"{'sd':>5s}   {'kept':>6s} {'disc':>6s}")
    # `.get("accuracy", -1)` is not enough: the key exists and holds None for a NO DATA
    # row, so the default never fires and negating None raises. Sort None last.
    for r in sorted(summary, key=lambda x: (x.get("accuracy") is None,
                                            -(x.get("accuracy") or 0.0))):
        if "error" in r:
            print(f"  {r['label']:20s} {r['kind']:10s}  FAILED: {r['error'][:48]}")
            continue
        if r.get("accuracy") is None:
            print(f"  {r['label']:20s} {r['kind']:10s}  NO DATA "
                  f"(0/{r.get('n_expected', 0)} entries scored)")
            continue
        k = f"{r['acc_kept']:.1%}" if r["acc_kept"] is not None else "  -"
        d = f"{r['acc_discarded']:.1%}" if r["acc_discarded"] is not None else "  -"
        print(f"  {r['label']:20s} {r['kind']:10s} {r['accuracy']:6.1%} "
              f"{r['tie_rate']:6.1%} {r.get('n_errors', 0):6d} {r['stability']:>7s} "
              f"{r.get('always_tied', 0):4d} {r.get('wobbled', 0):4d} "
              f"{r.get('flipped', 0):4d} {r['distinct_scores']:5d} {r['score_sd']:5.2f}   "
              f"{k:>6s} {d:>6s}")
    print(f"\n  acc  = higher score lands on the side the label calls better")
    print(f"  ties = both sides scored equal. **Counted as wrong, never excluded.**")
    print(f"  lvls = distinct score values used; a low count is why ties happen")
    print(f"  kept = accuracy on the {n_kept} judge1-filtered pairs, disc = the rest")
    print(f"  stable = pairs giving the SAME REAL preference in every run")
    print(f"  tie3 = pairs that tied in every run: consistent, but consistently failed,")
    print(f"         so excluded from stable rather than counted as agreement")
    print(f"  wob  = pairs wobbling between a tie and a preference (resolution limit)")
    print(f"  flip = pairs actually reversing 1 <-> 2 (changed its mind about direction)")
    print(f"  errs = calls that never produced a parseable score, after every retry pass.")
    print(f"         **Counted as wrong, like a tie, and never dropped.** An evaluator that")
    print(f"         cannot emit a score has not finished the example -- and the failures land")
    print(f"         on the weakest models, so grading each on the subset it managed to answer")
    print(f"         would flatter exactly those. `n` is the dataset size for every row and is")
    print(f"         no longer printed; a model that answered NOTHING shows NO DATA instead.")
    if n_kept and n_kept < n_pairs:
        print("\n  If kept >> disc for every evaluator, the filter removed the hard pairs and")
        print("  the filtered benchmark separates evaluators less well than the full one.")
    elif n_kept:
        print("\n  kept == the whole set, so disc is empty: this is a `--pool kept` dataset,")
        print("  where every pair passed judge1 by construction. Build with `--pool all` to")
        print("  compare the filtered subset against the pairs the filter rejected.")


if __name__ == "__main__":
    raise SystemExit(main())
