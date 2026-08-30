# Reference

Three routes, cheapest first.

**Analysis only.** No credentials, no model calls. Needs the data release at
`dataset/`:

```bash
python tools/run_evaluators.py --dataset dataset/benchmark/eval_dataset.json \
                              --out-dir dataset/evaluations --report-only
python tools/report.py
python tools/report_human.py
```

**Sweep the shipped dataset yourself.** Needs provider credentials, not τ³. Section 3.

**Regenerate the dataset, then sweep it.** Needs τ³. Sections 1–3.

---

## 1. Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env          # fill in the keys you have
pytest                       # τ³-dependent tests skip when τ³ is absent
```

Generation additionally needs τ³-bench:

```bash
git clone https://github.com/sierra-research/tau2-bench.git ../tau2-bench
git -C ../tau2-bench checkout 668d3bcd135c02aa3438f987ef45735b7c163ee3
bash scripts/setup_tau3.sh          # applies patches/tau3.patch, installs, verifies
bash scripts/setup_tau3.sh --check  # verify only
```

`patches/tau3.patch` touches 9 files. It disables τ²-bench's own LLM judges by default
(`TAU2_DISABLE_LLM_JUDGES=0` restores them), excludes two yanked `litellm` releases from its
dependency bound, adds a `TAU2_LLM_OVERRIDE` test hook, and restores a dropped latency field on
user-simulator messages. Domains, tasks, policies, tools and the agent loop are unmodified.

The pin is against upstream, so the substrate is addressable by anyone. `setup_tau3.sh` detects
an already-patched checkout by content, so re-running it is safe, and it warns if the checkout is
at a different commit. A substrate mismatch does not surface downstream: the pipeline runs and
reports accuracies over different tasks.

## 2. Generation

One YAML file, two flags: `--config` and `--dry-run`.

```bash
python tools/run_pipeline.py --config config/smoke.yaml --dry-run   # no calls
python tools/run_pipeline.py --config config/pipeline.yaml
```

`--dry-run` resolves the task list and prints the plan. It makes no model calls but does need τ³
installed, because planning the cells means enumerating τ³'s tasks.

A run resumes from whatever is already in `out_dir`. Unknown config keys are an error, not a
warning.

| key | default | what it does |
|---|---|---|
| `out_dir` | `data/run` | where trajectories, pairs and bookkeeping are written |
| `domains` | all | any of `airline`, `banking_knowledge`, `retail`, `telecom` |
| `criteria` | the built-in three | `name` + `description`. Changing a description defines a new criterion |
| `task_indices` | all | `[0, 1, 2]` takes the first three tasks of each domain |
| `limit` | none | first N cells only, redirected to `<out_dir>/smoke-N/` |
| `generator` | `gpt-oss-120b` | writes the steering instructions |
| `judges` | the built-in cascade | exactly two, keyed `judge1` and `judge2`; the keys carry the order |
| `workers` | 6 | cells in flight |
| `sim_concurrency` | 4 | simulations in flight; match the user simulator's replica count |
| `serverless_concurrency` | 8 | cap on serverless calls |
| `sim_attempts` | 10 | retries for a failed simulation; each attempt spends its tokens |
| `sim_backoff` | 10.0 | seconds between those attempts |
| `filter_retries` | 0 | re-draw a pair the filter rejected, N more times |
| `seed` | 2026 | fixes level-contrast and agent assignment, not the trajectories |
| `warm` | true | wake dedicated deployments before the first simulation |

`filter_retries` settings are nested: a run at 2 contains the 1 and 0 results, because every
draw records its attempt number. Run at the highest setting you will pay for, once.

## 3. The evaluator dataset and sweep

```bash
python tools/build_eval_dataset.py --run data/full_run
```

Writes `eval_dataset.json` (blind: id, criterion, context, both conversations) and
`eval_answers.json` (the labels), intersecting on `id`. They are separate because `levels` is
slot-ordered and the steering instructions name their own direction, so one record carrying both
gives away the answer. `--pool all` adds the rejected cells, for measuring the filter.

```bash
python tools/run_evaluators.py --dataset data/full_run/eval_dataset.json --runs 3 --workers 16
```

`--answers` defaults to `eval_answers.json` beside `--dataset`. `--votes` adds the
judge1-filtered subset column; on a `--pool kept` dataset that subset is the whole set by
construction. Resumable from `checkpoint.jsonl`; `--no-resume` re-scores everything.

Useful on a first attempt: `--models gpt-oss-20b` for one row, `--limit 20` for the first 20
pairs (writes to `evaluations/smoke-20/`), `--report-only` to rebuild the table from saved
results without calling anything.

### Which rows your credentials reach

The roster is 25 rows across six lanes, and every row is decided before the first call. The tool
prints each row it drops or reroutes, with the reason and the variable to set.

| lane | rows | needs | cap |
|---|---|---|---|
| Fireworks serverless | 8 | `FIREWORKS_API_KEY` | uncapped; account tokens/sec binds |
| Fireworks dedicated | 3 | `FIREWORKS_API_KEY` + `FIREWORKS_ACCOUNT_ID` | 2 per deployment |
| Gateway | 12 | `GATEWAY_BASE_URL` + `GATEWAY_API_KEY` | 10 |
| Gemini | 2 | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | 3 |
| Anthropic direct | — | `ANTHROPIC_API_KEY` | 4 |
| OpenAI direct | — | `OPENAI_API_KEY` | 4 |

Twelve rows — every Claude, every GPT, all three Llamas, Mistral — were served through one
OpenAI-compatible gateway rather than the vendors' own APIs. It is a site-specific deployment,
not a public endpoint. Without `GATEWAY_BASE_URL` the Claude and GPT rows reroute to
`api.anthropic.com` and `api.openai.com`, and the tool says so on each one: same weights,
different serving path. The four gateway-only open-weight rows are dropped by name rather than
substituted with a guess.

The three dedicated rows (`qwen3-4b`, `qwen3-1p7b`, `gemma-4-31b`) are addressed as
`#accounts/$FIREWORKS_ACCOUNT_ID/deployments/meta-eval-<model>`. The tool prints the `firectl`
recipe; creating the deployment is a manual step. With `FIREWORKS_ACCOUNT_ID` unset the three
rows are dropped by name.

## 4. What a rerun will not match

Trajectories are not bitwise reproducible. Temperature is 0 everywhere and the seed fixes
level-contrast and agent assignment, but MoE routing on Fireworks depends on batch composition,
so the same prompt can return different text. A regenerated dataset has the same shape — same
cells, same tasks, same assignment — and different conversations in it, so pair counts differ and
evaluator accuracies computed on it are not comparable to the published table. Compare against
the released `benchmark/eval_dataset.json`.

Five rows run at temperature 1.0 (`gpt-5.6-sol`, `-terra`, `-luna`, `gpt-5.4-mini`, `-nano`)
because the provider refuses 0. They are sampled rather than greedy-decoded, so their spread
across runs measures something different from the rest.

Two rows (`llama3.1-8b`, `qwen3-1p7b`) failed to emit a parseable score on some calls, after
5–8 attempts each, as `<think>` blocks hitting the 10,000-token cap without a verdict. Those are
counted incorrect rather than excluded; the `errs` column reports them.

## 5. If something goes wrong

| symptom | cause |
|---|---|
| `ModuleNotFoundError: No module named 'tau2'` | τ³ is not installed; section 1. Only generation needs it |
| `patches/tau3.patch does not apply` | the checkout is not at the pinned commit |
| `DEPLOYMENT_SCALING_UP` on turn one | a cold dedicated deployment. Leave `warm: true` on, or pin `--min-replica-count 1` |
| scaling errors instead of throughput | `sim_concurrency` above the deployment's replica count |
| the sweep stalls on one model | `--give-up-after` abandons a model after N failed calls with no successes (default 20) |
| a table row is missing | read the `roster:` block the sweep prints first; every dropped row is named there |
| tests fail on paths | run from the repo root, or let `tests/conftest.py` handle it |
