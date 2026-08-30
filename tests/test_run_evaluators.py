"""The evaluator sweep as an orchestrator: the roster, warming, and what lands on disk.

This file exists because the sweep stopped being cheap. `tools/run_pipeline.py` notes that
`run_evaluators` had no tests since it "fused its control flow to litellm"; moving the pool
into `evaluate_models` unfused it, and at 1,000 pairs a bug in the orchestration costs hours
rather than minutes. So `main()` is driven end to end with the provider mocked out.

What is pinned here is orchestration only -- scheduling, admission and resume belong to
`tests/test_eval_pool.py`, and scoring itself to `tests/test_independent.py`.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.run_evaluators import (
    ACCOUNT_SLOT,
    ANTHROPIC,
    DEDICATED_CAP,
    FW_SERVERLESS,
    GATEWAY,
    GEMINI,
    OPENAI,
    PROVIDER_CAPS,
    ROSTER,
    _gateway_call,
    _lane_caps,
    _lane_rates,
    main,
    resolve,
)


# --- the roster ------------------------------------------------------------------------

def test_roster_rows_carry_label_model_kind_and_lane():
    """A NamedTuple, because this shape has changed twice and the reporting code unpacks it too.

    The per-model worker count went first -- it has no meaning once there is one pool. `lane`
    replaced it: the admission-control domain, stated rather than parsed, because for half
    the roster the model string lies about who shares its rate limit.
    """
    assert ROSTER, "the roster is not empty"
    for r in ROSTER:
        assert r._fields[:4] == ("label", "model", "kind", "lane"), r._fields
        assert all(isinstance(x, str) and x for x in (r.label, r.model, r.kind, r.lane))
        assert r.call is None or isinstance(r.call, dict)


def test_only_the_gpt_5_family_departs_from_temperature_zero():
    """A per-row override is a caveat on that row's numbers, so the set is pinned.

    litellm refuses `temperature=0.0` for gpt-5 models outright, which is why the five GPT
    rows returned NO DATA on the first live smoke run. Everything else must stay greedy -- the
    alternative, `litellm.drop_params = True`, hands temperature back to the provider default
    and that is the bug that made the judges non-reproducible.
    """
    sampled = {r.label for r in ROSTER
               if (r.call or {}).get("temperature") not in (None, 0.0)}
    assert sampled == {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
                       "gpt-5.4-mini", "gpt-5.4-nano"}
    assert all((r.call or {}).get("temperature") == 1.0 for r in ROSTER
               if r.label in sampled), "1.0 is the only value the provider allows"
    # Nothing else may carry a call override without this test being updated on purpose.
    assert {r.label for r in ROSTER if r.call} == sampled


def test_the_temperature_caveat_reaches_summary_json(bench):
    """`summary.json` outlives the console, and "accuracy 61%" means a different thing at
    temperature 1.0 than at 0.0."""
    _run(bench, "--models", "gpt-5.4-mini", "claude-haiku-4-5")
    rows = {r["label"]: r for r in json.loads((bench["out"] / "summary.json").read_text())}
    assert rows["gpt-5.4-mini"]["temperature"] == 1.0
    assert rows["claude-haiku-4-5"]["temperature"] == 0.0
    assert rows["gpt-5.4-mini"]["lane"] == GATEWAY


def test_a_sampled_row_is_announced_before_any_call(bench, capsys):
    _run(bench, "--models", "gpt-5.4-nano", "gpt-oss-20b")
    out = capsys.readouterr().out
    assert "NOT at temperature 0" in out
    assert "gpt-5.4-nano" in out.split("NOT at temperature 0")[1]
    assert "sampled, not greedy" in out


def test_a_row_override_beats_its_lane(bench):
    """`NEEDS_T1` has to win over the gateway's own kwargs, which the other ten rows share."""
    seen = []

    def capture(*_a, **kw):
        seen.append(kw)
        return _reply(kw["messages"][0]["content"])

    _run(bench, "--models", "gpt-5.4-mini", "claude-haiku-4-5", completion=capture)
    gpt = [k for k in seen if "gpt-5.4-mini" in str(k.get("model", ""))]
    cla = [k for k in seen if "claude" in str(k.get("model", ""))]
    assert gpt and cla
    assert all(k["temperature"] == 1.0 for k in gpt)
    assert all(k["temperature"] == 0.0 for k in cla), "the lane default is untouched"
    # And the lane's own kwargs still arrive on the overridden row.
    assert all(k.get("max_completion_tokens") == 10_000 for k in gpt)


def test_every_lane_is_one_we_know_how_to_call():
    """A typo in a lane silently means "uncapped, called the plain way", which for a gateway
    row is 2,000 auth failures rather than an error.

    `ANTHROPIC` and `OPENAI` are known lanes but appear in no roster row: no published number
    came from a direct vendor call. They exist only as `resolve()`'s fallback targets, and
    that asymmetry is the point -- a roster row on one of them would be a row whose lane never
    matched how it was measured.
    """
    known = {FW_SERVERLESS, GEMINI, GATEWAY, ANTHROPIC, OPENAI}
    for r in ROSTER:
        assert r.lane in known or r.lane.startswith("dep:"), (r.label, r.lane)
    assert not any(r.lane in (ANTHROPIC, OPENAI) for r in ROSTER), (
        "the published sweep went through the gateway, so no row starts on a direct lane")


def test_every_dedicated_row_carries_the_account_slot_and_no_real_account():
    """The roster is a template, and a hardcoded account is both wrong for anyone else and a
    leak of whose it was."""
    deps = [r for r in ROSTER if r.lane.startswith("dep:")]
    assert deps
    for r in deps:
        assert ACCOUNT_SLOT in r.model, r.label
        assert "#accounts/" in r.model


def test_resolve_leaves_a_reachable_roster_untouched():
    """The no-op case, because `resolve` runs on every invocation: with every credential
    present, nothing may be dropped, rerouted, or reordered."""
    env = {"GATEWAY_BASE_URL": "u", "GATEWAY_API_KEY": "k", "FIREWORKS_AI_API_KEY": "k",
           "GEMINI_API_KEY": "k", "FIREWORKS_ACCOUNT_ID": "acct"}
    with patch.dict(os.environ, env, clear=True):
        runnable, notes = resolve(list(ROSTER))
    assert [r.label for r in runnable] == [r.label for r in ROSTER]
    assert notes == []
    assert all(ACCOUNT_SLOT not in r.model for r in runnable), "the account was substituted"


def test_a_dedicated_model_and_its_lane_agree():
    """`#` in the model string and a `dep:` lane are the same fact stated twice, and warming
    reads the first while capping reads the second."""
    for r in ROSTER:
        assert ("#" in r.model) == r.lane.startswith("dep:"), (r.label, r.lane)
        if r.lane.startswith("dep:"):
            assert r.lane.removeprefix("dep:") in r.model, r.label


def test_gateway_rows_are_addressed_as_an_openai_compatible_gateway():
    """It serves `POST /v1/chat/completions`, so `openai/<model>` is the litellm form -- for
    Claude as much as for GPT. The `anthropic/` path also works but serves Claude only."""
    gw = [r for r in ROSTER if r.lane == GATEWAY]
    assert gw, "the roster has gateway models"
    assert all(r.model.startswith("openai/") for r in gw), [r.model for r in gw]
    assert any("claude" in r.model for r in gw) and any("gpt" in r.model for r in gw)


def test_roster_labels_are_unique():
    """The pool keys tasks, checkpoint rows and output files by label; a duplicate would
    silently drop a model (`dict(models)` keeps the last)."""
    labels = [r[0] for r in ROSTER]
    assert len(labels) == len(set(labels)), labels


def test_fireworks_serverless_is_the_one_unthrottled_lane():
    """Its binding limit is tokens/sec across the whole account, and the pool spread over eight
    of its models is well inside what four workers on a *single* model sustained. It is also the
    only lane that produced no 429 at all in the full sweep."""
    caps, rates = _lane_caps(ROSTER), _lane_rates(ROSTER)
    assert FW_SERVERLESS not in caps and FW_SERVERLESS not in rates
    assert caps[GEMINI] == 3


def test_the_gateway_carries_both_a_rate_ceiling_and_a_throughput_cap():
    """Two numbers doing two different jobs, and neither is the other's substitute.

    The **rate** is a ceiling in the units the gateway stated its limit in, which a
    concurrency cap cannot express: 10 in flight is 1.25 starts/s at 8s latency and 10/s if
    the gateway answers instantly. The ceiling is now **60/min** -- the real server-side limit,
    the stated "300 req/min" having been refuted by a run that drew 429s at 79.

    **At 60 the rate binds unconditionally, which is a change in kind.** At cap 10 the lane
    wants `600/L` starts/min: 140 at the 4.3s a 96-call probe measured idle, 83 at the ~7.2s the
    campaign implies under load. Both are far above 60, so no latency this lane plausibly shows
    makes the ceiling slack, and it now sets a floor on wall clock that neither more workers nor
    fewer retries can lower: 24,000 calls / 60 per minute = 6.7h at k=1, minimum.

    The **cap** is a throughput decision. Twelve of 25 rows are on this lane -- 24,000 calls at
    ~8s -- so the lane's wall clock is `24000 * 8 / cap`: 8.9h at 6, ~5.3h at 10. A cap of 6 is
    what made the first full main pass take 6h59m.
    """
    caps, rates = _lane_caps(ROSTER), _lane_rates(ROSTER)
    assert rates[GATEWAY] == 60, "the real server-side limit, see LANE_RATES"
    assert rates[GATEWAY] < 300, "the limit is per user and shared with interactive work"
    assert caps[GATEWAY] == 10, "sized from the lane's 24,000 calls, see PROVIDER_CAPS"
    # The cap must not be able to outrun the ceiling at the latencies observed, or the rate
    # limiter becomes the thing throttling normal operation rather than a backstop.
    # At 72/min the cap CAN outrun the ceiling (10 in flight at 8s is 75/min), which is the
    # difference between this rate and the old 240: the limiter now paces normal operation
    # instead of only catching spikes. Asserted in the direction that is now true, so the
    # relationship stays visible rather than quietly inverting.
    assert caps[GATEWAY] * (60 / 8.0) > rates[GATEWAY], "the rate, not the cap, is what binds"
    assert rates[GATEWAY] * 1.0 < caps[GATEWAY] * (60 / 4.3), "and it binds harder when idle"


def test_lane_rates_are_scoped_to_the_models_in_this_run():
    """A Fireworks-only sweep must not carry the gateway's throttle."""
    fw = [r for r in ROSTER if r.lane == FW_SERVERLESS]
    assert _lane_rates(fw) == {}
    gw = [r for r in ROSTER if r.lane == GATEWAY]
    assert _lane_rates(gw) == {GATEWAY: 60}


def test_every_dedicated_deployment_is_capped_without_being_listed():
    """Built from the roster so adding a deployment row cannot forget to cap it. One replica
    cannot serve more than a couple of 7k-token prompts, and workers queueing on it are
    workers not spending time on the other 21 models."""
    caps = _lane_caps(ROSTER)
    deps = {r.lane for r in ROSTER if r.lane.startswith("dep:")}
    assert deps, "the roster has dedicated deployments"
    assert all(caps[d] == DEDICATED_CAP for d in deps)


def test_lane_caps_are_scoped_to_the_models_in_this_run():
    """A `--models gpt-oss-20b` run must not carry a Gemini or deployment cap it never uses:
    an unused semaphore is harmless, but a cap dict that does not match the run is the kind
    of thing that gets quoted in a table."""
    one = [r for r in ROSTER if r.label == "gpt-oss-20b"]
    assert _lane_caps(one) == {}
    gem = [r for r in ROSTER if r.lane == GEMINI]
    assert _lane_caps(gem) == {GEMINI: 3}


# --- driving main() --------------------------------------------------------------------

def _dataset(n=3):
    """The blind half, in the shape `build_eval_dataset.py` writes: no `correct_response`."""
    return [{
        "id": f"pair-{i}",
        "criterion_name": "clarity",
        "criterion_description": "Is it clear?",
        "prompt": "policy",
        "response_1": f"good {i}",
        "response_2": f"bad {i}",
    } for i in range(n)]


@pytest.fixture
def bench(tmp_path):
    """A blind dataset, its answers file, a judge-vote file, and an output directory.

    **Two files, because the sweep loads two.** The fixture used to carry `correct_response`
    inline, which no longer round-trips: `load_pairs` joins the answers in `main` after the
    calls, and a fixture that skipped the join would exercise a path the tool does not have.
    The answers file also has to sit beside the dataset under the derived name, which is the
    convention `--answers` exists to override.
    """
    data = _dataset()
    ds = tmp_path / "dataset.json"
    ds.write_text(json.dumps(data))
    answers = tmp_path / "dataset_answers.json"
    answers.write_text(json.dumps(
        [{"id": e["id"], "correct_response": 1, "levels": ["good", "bad"]} for e in data]))
    votes = tmp_path / "judge_votes.json"
    votes.write_text(json.dumps({"votes": [
        {"id": e["id"], "judge": "judge1", "side": 1} for e in data]}))
    return {"dataset": str(ds), "answers": str(answers), "votes": str(votes),
            "out": tmp_path / "out", "data": data}


def _reply(prompt: str):
    """A provider response for one scoring call.

    `litellm.completion` is patched rather than `_score_model`, so the real scoring path runs
    -- prompt assembly, score extraction and **usage recording**. That last one is the reason:
    a mocked `_score_model` records nothing into the meter, and then `usage.json`, the
    per-model usage files and the checkpoint's cost column are all absent for a reason that
    has nothing to do with the code under test.

    The score is read off the response text the prompt carries, which is how a fake evaluator
    gets to be right: fixtures name the better side "good N".
    """
    from types import SimpleNamespace
    score = 0.9 if "good" in prompt.split("CONVERSATION")[-1] else 0.2
    msg = MagicMock()
    msg.content = json.dumps({"score": score, "reasoning": "fixture"})
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = SimpleNamespace(prompt_tokens=7000, completion_tokens=300,
                                 completion_tokens_details=None)
    return resp


def _run(bench, *extra, cold=(), completion=None):
    """`main()` with the provider replaced. Returns (exit code, the `warm` mock)."""
    argv = ["run_evaluators.py", "--dataset", bench["dataset"], "--votes", bench["votes"],
            "--out-dir", str(bench["out"]), "--runs", "1", *extra]

    def default(*_a, **kw):
        return _reply(kw["messages"][0]["content"])

    warm = MagicMock(return_value=list(cold))
    # Every lane's credential, set here rather than inherited. `resolve()` decides which rows
    # are runnable from the environment, so a test that names a gateway or Gemini row must not
    # depend on whether the developer's shell happens to have that vendor configured -- and
    # must not silently reroute a gateway row to a vendor and then assert on gateway kwargs.
    env = {"GATEWAY_BASE_URL": "https://gateway.example/v1",
           "GATEWAY_API_KEY": "test-key",
           "GATEWAY_USER": "tester@example.com",
           "FIREWORKS_AI_API_KEY": "test-fw",
           "GEMINI_API_KEY": "test-gemini",
           "ANTHROPIC_API_KEY": "test-anthropic",
           "OPENAI_API_KEY": "test-openai",
           "FIREWORKS_ACCOUNT_ID": "test-account"}
    with patch.object(sys, "argv", argv), \
         patch.dict(os.environ, env), \
         patch("tools.run_evaluators.ensure_env"), \
         patch("tools.run_evaluators.warm", warm), \
         patch("src.metaeval.scoring.litellm.completion",
               side_effect=completion or default):
        code = main()
    return code, warm


THREE_LANES = ("gpt-oss-20b", "gemini-3.7-flash", "claude-haiku-4-5")


def test_a_full_sweep_writes_results_metrics_and_a_pooled_usage_file(bench):
    """One sweep across all three remote lanes: Fireworks serverless, Google, the gateway."""
    code, _ = _run(bench, "--models", *THREE_LANES)
    assert code == 0
    out = bench["out"]
    for label in THREE_LANES:
        assert json.loads((out / f"{label}.results.json").read_text())
        assert json.loads((out / f"{label}.metrics.json").read_text())
        assert (out / f"{label}.usage.json").is_file()
    summary = {r["label"]: r for r in json.loads((out / "summary.json").read_text())}
    assert set(summary) == set(THREE_LANES)
    assert all(r["accuracy"] == 1.0 for r in summary.values())
    assert all(r["n_scored"] == 3 for r in summary.values())


def test_the_gateway_swaps_the_token_parameter_and_the_others_do_not(bench):
    """The gateway 400s on `max_tokens` ("deprecated in favor of max_completion_tokens") and
    400s again if both are sent, so the swap has to be a swap rather than an addition.

    Checked at the litellm boundary because that is where the 400 comes from -- asserting on
    `_gateway_call()` alone would not catch `_score_model` failing to drop the None.
    """
    seen = []

    def capture(*_a, **kw):
        seen.append(kw)
        return _reply(kw["messages"][0]["content"])

    _run(bench, "--models", "claude-haiku-4-5", "gpt-oss-20b", completion=capture)
    claude_calls = [k for k in seen if "claude" in str(k.get("model", ""))]
    fw = [k for k in seen if "gpt-oss" in str(k.get("model", ""))]
    assert claude_calls and fw
    for k in claude_calls:
        assert k.get("max_completion_tokens") == 10_000
        assert "max_tokens" not in k, "sending both is a 400"
        assert k.get("api_base", "").endswith("/v1")
        assert k.get("api_key") == "test-key"
        assert k["extra_headers"]["X-Gateway-User"] == "tester@example.com"
    for k in fw:
        assert k.get("max_tokens") == 10_000, "a plain provider keeps the normal spelling"
        assert "max_completion_tokens" not in k
        assert "api_base" not in k


def test_no_reasoning_effort_is_ever_sent(bench):
    """Every model runs at its own default, because the question is what a deployed evaluator
    gives you off the shelf. That default varies hugely -- measured on one real prompt: Claude
    0 reasoning tokens, gpt-5.4-mini 0, gpt-5.6-sol 86 of 151, gpt-5-nano 1,088 of 1,173 --
    and normalising it away would erase a finding about the roster."""
    seen = []

    def capture(*_a, **kw):
        seen.append(kw)
        return _reply(kw["messages"][0]["content"])

    _run(bench, "--models", *THREE_LANES, completion=capture)
    assert seen
    for k in seen:
        assert "reasoning_effort" not in k
        assert "thinking" not in k
        assert k.get("temperature") == 0.0


def _run_with_env(bench, env, *models, completion=None):
    """`main()` with a hand-built environment, for the credential-resolution tests."""
    argv = ["run_evaluators.py", "--dataset", bench["dataset"], "--votes", bench["votes"],
            "--out-dir", str(bench["out"]), "--runs", "1", "--models", *models]
    calls = MagicMock(side_effect=completion
                      or (lambda *a, **kw: _reply(kw["messages"][0]["content"])))
    with patch.object(sys, "argv", argv), \
         patch.dict(os.environ, env, clear=True), \
         patch("tools.run_evaluators.ensure_env"), \
         patch("tools.run_evaluators.warm", MagicMock(return_value=[])), \
         patch("src.metaeval.scoring.litellm.completion", calls):
        return main(), calls


def test_a_row_with_no_reachable_credential_is_dropped_before_any_call(bench, capsys):
    """2,000 identical auth failures discovered an hour in is the alternative, and the
    checkpoint would have recorded every one of them as billed.

    The row is *named* on the way out. A sweep that silently drops 8 of 25 evaluators produces
    a table that looks complete, and those labels are the rows anyone would quote.
    """
    code, calls = _run_with_env(bench, {}, "claude-haiku-4-5")
    assert code == 2, "nothing was runnable, so this is a failure and not an empty success"
    assert calls.call_count == 0, "and nothing was billed"
    out = capsys.readouterr().out
    assert "claude-haiku-4-5" in out and "DROPPED" in out


def test_a_gateway_row_falls_back_to_its_vendor_and_says_so(bench, capsys):
    """The gateway served Claude and GPT under its own names, so without it those rows can
    still be run -- against the vendor directly. That is a *replication*, not the same
    measurement, so it is announced rather than substituted quietly.
    """
    code, calls = _run_with_env(bench, {"ANTHROPIC_API_KEY": "k"}, "claude-haiku-4-5")
    assert code == 0
    assert calls.call_count > 0
    model = calls.call_args_list[0][1]["model"]
    assert model.startswith("anthropic/"), model
    out = capsys.readouterr().out
    assert "NOT via the gateway" in out, "the caveat travels with the row"


def test_an_open_weight_gateway_row_is_dropped_rather_than_guessed(bench, capsys):
    """Four rows reached open-weight models through the gateway under private names. There is
    no published vendor slug for them, so a substitution would produce a row that answers and
    is not the model that was measured -- which is worse than a gap in the table."""
    code, calls = _run_with_env(bench, {"FIREWORKS_AI_API_KEY": "k"},
                                "llama3.1-8b", "gpt-oss-20b")
    assert code == 0, "the healthy row still runs"
    out = capsys.readouterr().out
    assert "llama3.1-8b" in out and "DROPPED" in out
    assert not (bench["out"] / "llama3.1-8b.results.json").exists()
    assert all("llama" not in kw["model"] for _a, kw in calls.call_args_list)


def test_a_run_without_gateway_models_needs_no_gateway_credential(bench):
    """Credentials are resolved per lane, so a Fireworks-only sweep works on a machine that
    has never seen the gateway."""
    code, calls = _run_with_env(bench, {"FIREWORKS_AI_API_KEY": "k"}, "gpt-oss-20b")
    assert code == 0
    assert calls.call_count > 0


def test_usage_json_is_the_pooled_shape_with_one_shared_wall_clock(bench):
    """`merge()` sums per-meter wall clocks, which is wrong the moment models overlap: nine
    concurrent models would report nine times the elapsed time and deflate every rate by
    nine. The flag is what tells a reader of the file which arithmetic produced it."""
    _run(bench, "--models", *THREE_LANES)
    u = json.loads((bench["out"] / "usage.json").read_text())
    assert u["pooled"] is True
    assert set(u["per_model"]) == set(THREE_LANES)
    assert u["total"]["n_calls"] == 3 * 2 * 3
    # One window, shared. Under `merge` each model's wall would have been its own slice.
    walls = {m["wall_s"] for m in u["per_model"].values()}
    assert len(walls) == 1
    assert u["total"]["wall_s"] == walls.pop()
    # Percentiles are dropped from the total and kept per model, as both writers intend.
    assert "latency_p95_s" not in u["total"]


def test_the_pooled_usage_file_names_every_model_in_the_sweep(bench):
    """`usage.json` is the only record of what a run cost once the console is gone, so a model
    absent from `per_model` is a model whose spend is unattributable after the fact."""
    _run(bench, "--models", *THREE_LANES)
    u = json.loads((bench["out"] / "usage.json").read_text())
    assert set(u["per_model"]) == set(THREE_LANES)
    assert all(u["per_model"][m]["n_calls"] > 0 for m in THREE_LANES)


# --- warming ---------------------------------------------------------------------------

def test_dedicated_deployments_are_warmed_once_before_the_pool(bench):
    """Warming per model was right for a serial sweep and impossible for a pooled one.

    Scale-to-zero is 5 minutes and a serial sweep took ~30, so a model warmed at minute 0
    had slept by its turn -- `qwen3-4b` and `qwen3-1p7b` each scored 0 of 36 that way. A
    pool has no turns: every deployment is in traffic from the first second, so warming is a
    single start-up step, and both dedicated models must appear in that one call.
    """
    _, warm = _run(bench, "--models", "gpt-oss-20b", "qwen3-4b", "qwen3-1p7b")
    assert warm.call_count == 1, "once, not once per model"
    warmed = warm.call_args[0][0]
    assert len(warmed) == 2, warmed
    assert all("#" in m for m in warmed), "only dedicated deployments need waking"
    assert any("qwen3-4b" in m for m in warmed) and any("qwen3-1p7b" in m for m in warmed)


def test_a_serverless_only_roster_is_not_warmed_at_all(bench):
    _, warm = _run(bench, "--models", "gpt-oss-20b", "kimi-k3")
    assert warm.call_count == 0


def _resolved_model(label: str) -> str:
    """The address `warm` is actually handed, account substituted.

    `ROSTER` carries the deployment template with `ACCOUNT_SLOT` in it -- taking `r.model`
    straight from the roster would name an address the run never uses, and the test would then
    assert on a cold-deployment path that never fires.
    """
    with patch.dict(os.environ, {"FIREWORKS_ACCOUNT_ID": "test-account"}):
        runnable, _ = resolve([r for r in ROSTER if r.label == label])
    return runnable[0].model


def test_a_deployment_that_will_not_come_up_costs_its_row_and_not_the_run(bench):
    """Eight healthy models must not be lost to one cold deployment."""
    code, _ = _run(bench, "--models", "gpt-oss-20b", "qwen3-4b",
                   cold=[_resolved_model("qwen3-4b")])
    assert code == 0
    summary = {r["label"]: r for r in json.loads((bench["out"] / "summary.json").read_text())}
    assert summary["qwen3-4b"]["error"] == "deployment did not come up"
    assert summary["gpt-oss-20b"]["accuracy"] == 1.0
    assert not (bench["out"] / "qwen3-4b.results.json").exists()


def test_every_deployment_cold_is_an_error_exit(bench):
    qwen = [_resolved_model(l) for l in ("qwen3-4b", "qwen3-1p7b")]
    code, _ = _run(bench, "--models", "qwen3-4b", "qwen3-1p7b", cold=qwen)
    assert code == 2


# --- resume, from the CLI --------------------------------------------------------------

def test_a_second_invocation_makes_no_calls(bench):
    _run(bench, "--models", "gpt-oss-20b")
    ck = bench["out"] / "checkpoint.jsonl"
    assert ck.is_file()
    before = ck.read_text()

    seen = []

    def counting(*_a, **kw):
        seen.append(kw["messages"][0]["content"])
        return _reply(kw["messages"][0]["content"])

    code, _ = _run(bench, "--models", "gpt-oss-20b", completion=counting)
    assert code == 0
    assert seen == [], "a completed sweep costs nothing to re-run"
    assert ck.read_text() == before
    # And the table is still built, from the resumed scores.
    summary = json.loads((bench["out"] / "summary.json").read_text())
    assert summary[0]["n_scored"] == 3


def test_no_resume_pays_again(bench):
    _run(bench, "--models", "gpt-oss-20b")
    seen = []

    def counting(*_a, **kw):
        seen.append(kw["messages"][0]["content"])
        return _reply(kw["messages"][0]["content"])

    _run(bench, "--models", "gpt-oss-20b", "--no-resume", completion=counting)
    assert len(seen) == 3 * 2


def test_the_checkpoint_lives_beside_the_results_it_belongs_to(bench):
    """Including under the smoke redirect, or a `--limit` run would resume from -- and top up
    -- the real sweep's checkpoint."""
    _run(bench, "--models", "gpt-oss-20b")
    assert (bench["out"] / "checkpoint.jsonl").is_file()
    rows = [json.loads(l) for l in
            (bench["out"] / "checkpoint.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 3 * 2
    assert {r["label"] for r in rows} == {"gpt-oss-20b"}
    assert all(r.get("usage") for r in rows), "cost travels with the score"


# --- the flags the run is actually launched with ----------------------------------------

def test_worker_default_is_eight_and_seed_default_is_zero(bench, capsys):
    _run(bench, "--models", "gpt-oss-20b")
    printed = capsys.readouterr().out
    assert "pool: 8 worker(s)" in printed
    assert "seed 0" in printed


def test_a_targeted_rerun_keeps_the_other_models_rows(bench):
    """The rule judge votes already follow: a `--models X` run must not delete Y's results."""
    _run(bench, "--models", "gpt-oss-20b", "kimi-k3")
    _run(bench, "--models", "kimi-k3")
    summary = {r["label"] for r in json.loads((bench["out"] / "summary.json").read_text())}
    assert summary == {"gpt-oss-20b", "kimi-k3"}


def test_an_unknown_model_name_is_an_error_not_an_empty_sweep(bench):
    code, _ = _run(bench, "--models", "not-a-model")
    assert code == 2
