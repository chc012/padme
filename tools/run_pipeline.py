#!/usr/bin/env python3
"""Generate data pairs as independent tasks over a worker pool.

    python tools/run_pipeline.py --dry-run            # plan only, zero LLM calls
    python tools/run_pipeline.py                      # the run

Every parameter comes from `config/pipeline.yaml`; `--config` and `--dry-run` are the only
flags. Set `limit: 3` there for a smoke run, which is redirected to `<out_dir>/smoke-3/`.

**One task is one data pair**, and the task runs its whole chain itself:

    P1 instructions   1 call    generator writes bad/ok/good for this cell
    P2 trajectories   2 sims    only the two levels the assigned gap needs
    P3 pair build     0 calls   the two trajectories become one pair
    P4 filter judge1  1 call    disagrees with intent -> rejected, task ends
    P5 filter judge2  1 call    disagrees with intent -> rejected, else kept

The alternative -- a barrier between stages, which is what the three standalone tools do --
makes every unit wait for the slowest member of its stage. Our stage times are extremely
skewed: one trajectory call took 103.2s against a 0.75s median. With no barrier that tail
costs one task instead of all of them.

Things this deliberately does NOT do:

- **No stratum quota.** Retention differs sharply by criterion (judge1 keeps friendliness
  12/12, task_resolution 7/12) and that *is* a measurement: 100% retention with high
  evaluator accuracy means the criterion is too easy to discriminate, while low retention
  with low accuracy means it is hard to label and hard to score. Forcing balance would
  delete the signal that separates those. Attempts and kept are recorded per stratum so the
  ratio is reportable; nothing is capped.
- **No per-stage wall clock.** Once pair 7's P1 overlaps pair 2's P4 there is no interval
  belonging to a stage. Time is reported as summed latency per pair per stage (additive,
  concurrency-independent) plus one whole-pipeline elapsed figure, with the worker count
  recorded beside it -- see `pipeline_run.json`.
- **No rewrite of an existing pair file.** Pair ids are uuids, so regenerating a pair would
  mint new ids and orphan its votes. Resume reads the pair back instead.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)

# tau3 dumps its whole component registry at import. With N workers interleaving, that noise
# buries the one line per phase transition that is the pool's actual progress output.
from loguru import logger  # noqa: E402

logger.remove()

from src.metaeval.usage import UsageMeter  # noqa: E402

_DEFAULT_OUT = Path("data/step5/pipeline")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

_DEFAULTS = {
    "out_dir": "data/run",
    "domains": None,              # None -> every registered domain
    "criteria": None,             # None -> the built-in three
    "task_indices": None,         # None -> every task in each domain
    "limit": None,
    "generator": None,            # None -> DEFAULT_GENERATOR
    "judges": None,               # None -> the built-in cascade
    "workers": 6,
    "sim_concurrency": 4,
    "serverless_concurrency": 8,
    "sim_attempts": 10,
    "sim_backoff": 10.0,
    "filter_retries": 0,
    "seed": 2026,
    "warm": True,
}


@dataclass
class Config:
    """Every run parameter, from one YAML file.

    Config lives in a file rather than on the command line because a run is only reproducible
    if its parameters travel with it -- and because there are fifteen of them, which is past
    the point where a shell invocation is readable. `--dry-run` stays a flag: it describes the
    invocation, not the run.

    Unknown keys are an error, not a warning. A typo in `sim_concurency` that silently fell
    back to the default would be invisible in the output and would misattribute whatever the
    run then measured.
    """

    out_dir: Path
    domains: list[str]
    criteria: dict            # name -> Criterion
    task_indices: Optional[list[int]]
    limit: Optional[int]
    generator: str
    judges: dict[str, str]
    workers: int
    sim_concurrency: int
    serverless_concurrency: int
    sim_attempts: int
    sim_backoff: float
    filter_retries: int
    seed: int
    warm: bool

    @property
    def as_record(self) -> dict:
        """What goes into `pipeline_run.json`. Criteria by name; the text is in the config."""
        return {"workers": self.workers, "sim_concurrency": self.sim_concurrency,
                "serverless_concurrency": self.serverless_concurrency, "seed": self.seed,
                "generator": self.generator, "judges": dict(self.judges),
                "sim_attempts": self.sim_attempts, "sim_backoff": self.sim_backoff,
                "filter_retries": self.filter_retries,
                "criteria": sorted(self.criteria), "domains": sorted(self.domains),
                "task_indices": self.task_indices}


def load_config(path: Optional[Path]) -> Config:
    """Read the YAML, fill defaults, and fail loudly on anything unrecognised."""
    import yaml
    from src.metaeval.sources.tau2_source import DOMAINS, JUDGES
    from src.metaeval.steering import CRITERIA
    from src.metaeval.steering.criteria import Criterion
    from src.metaeval.steering.generate import DEFAULT_GENERATOR

    raw: dict = {}
    if path is not None:
        if not path.is_file():
            raise SystemExit(f"config not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise SystemExit(f"{path}: top level must be a mapping")

    unknown = sorted(set(raw) - set(_DEFAULTS))
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) {unknown}. "
                         f"Known keys: {sorted(_DEFAULTS)}")

    v = {**_DEFAULTS, **raw}

    domains = v["domains"] or sorted(DOMAINS)
    bad = [d for d in domains if d not in DOMAINS]
    if bad:
        raise SystemExit(f"unknown domain(s) {bad}. Available: {sorted(DOMAINS)}")

    # **Criteria come from the config as name + description.** That is the whole contract:
    # `generate_instructions` reads only those two fields. `Criterion.instructions` -- the
    # hand-written per-level steering text -- is used by the older static path
    # (`steered_instruction`) and never by this pipeline, so a config-defined criterion needs
    # no instructions and gets an empty dict. A name matching a built-in reuses the built-in,
    # so the three shipped criteria keep their exact text.
    criteria: dict = {}
    if v["criteria"] is None:
        criteria = dict(CRITERIA)
    else:
        if not isinstance(v["criteria"], list):
            raise SystemExit("criteria: must be a list of {name, description} mappings")
        for i, item in enumerate(v["criteria"]):
            if not isinstance(item, dict) or "name" not in item:
                raise SystemExit(f"criteria[{i}]: needs at least a `name`")
            name = str(item["name"])
            desc = (item.get("description") or "").strip()
            if name in CRITERIA and not desc:
                criteria[name] = CRITERIA[name]
                continue
            if not desc:
                raise SystemExit(f"criteria[{i}] ({name}): needs a `description`, since it is "
                                 f"not one of the built-ins {sorted(CRITERIA)}")
            criteria[name] = Criterion(name=name, description=desc, instructions={})
    if not criteria:
        raise SystemExit("criteria: at least one is required")

    idx = v["task_indices"]
    if idx is not None:
        if not isinstance(idx, list) or not all(isinstance(i, int) for i in idx):
            raise SystemExit("task_indices: a list of integers, or null for every task")

    for key in ("workers", "sim_concurrency", "serverless_concurrency", "sim_attempts"):
        if int(v[key]) < 1:
            raise SystemExit(f"{key}: must be >= 1")
    if int(v["filter_retries"]) < 0:
        raise SystemExit("filter_retries: must be >= 0")

    # **The cascade is exactly two judges, and its shape is validated rather than assumed.**
    # `run_task` zips this mapping against `("P4", "P5")`, so a third entry would be generated,
    # paid for and then silently ignored, and a single entry would leave P5 unfilled. The key
    # *names* carry the order -- `judge1` runs first, `judge2` sees only its survivors -- so a
    # YAML file whose insertion order disagrees with its key names must not be allowed to
    # reorder the cascade. Checked here because both failures are invisible in the output: a
    # dropped judge looks like a lenient filter, and a swapped one looks like a strict one.
    j = dict(v["judges"] or JUDGES)
    if sorted(j) != ["judge1", "judge2"]:
        raise SystemExit(f"judges: need exactly judge1 and judge2, got {sorted(j)}")
    judges = {k: j[k] for k in ("judge1", "judge2")}

    return Config(
        out_dir=Path(v["out_dir"]),
        domains=list(domains),
        criteria=criteria,
        task_indices=list(idx) if idx is not None else None,
        limit=int(v["limit"]) if v["limit"] else None,
        generator=v["generator"] or DEFAULT_GENERATOR,
        judges=judges,
        workers=int(v["workers"]),
        sim_concurrency=int(v["sim_concurrency"]),
        serverless_concurrency=int(v["serverless_concurrency"]),
        sim_attempts=int(v["sim_attempts"]),
        sim_backoff=float(v["sim_backoff"]),
        filter_retries=int(v["filter_retries"]),
        seed=int(v["seed"]),
        warm=bool(v["warm"]),
    )


# --------------------------------------------------------------------------- #
# The work unit
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Cell:
    """One data pair's identity and plan.

    `gap` is fixed here and never recomputed. `balanced_cycle` assigns gaps over *the combos
    in the invocation that calls it*, so a narrowed re-run would reassign them and the dataset
    would silently change shape. So the first run assigns once, writes the assignment into
    `pair_index.json`, and every later run reads it back.
    """

    domain: str
    criterion: str
    task_index: int
    task_id: str
    gap: tuple[str, str]            # (worse, better), in bad<ok<good order
    agent: str

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.criterion}.t{self.task_index}"

    @property
    def levels(self) -> tuple[str, str]:
        return self.gap


@dataclass
class Stages:
    """The four things that talk to a model. Injected, so the state machine is testable
    without spending a token. That is why `tests/test_run_pipeline.py` exists at all, and why
    `run_evaluators.py` has no equivalent test: it fuses its control flow to litellm.

    `trajectory` returns `(trajectory, attempts_used)` -- the attempt count is the retry
    evidence, and dropping it would make a first-try success indistinguishable from a
    third-try one.
    """

    instructions: Callable[[Cell], dict[str, str]]
    trajectory: Callable[[Cell, str, str], tuple[Any, int]]
    build_pair: Callable[[Cell, dict[str, Any]], dict]
    # `judge` takes the cell too. It was the only callback without it, and the workaround --
    # a pair-id -> group-key dict populated by `build_pair` -- was silently wrong on the
    # resume path: `run_task` short-circuits P3 when the pair is on disk, so the dict was
    # empty exactly when resuming, and the judge recorded under the pair uuid while P1 and P2
    # for the same datapoint recorded under the group key. That is the accounting
    # fragmentation this was all meant to fix, reappearing on the path the tool is built for.
    judge: Callable[[str, dict, Cell], dict]


@dataclass
class Limits:
    """Per-resource admission control, shared by every worker.

    A task pool of W workers says nothing about how many calls are in flight to one model,
    and that is the limit that actually binds. Two separate measurements, and they disagree
    about how tight the cap should be:

    - the **evaluator sweep** at 8 workers returned 20 rate-limit errors against 1 at 4
      workers (the model that tripped them was nemotron-lightning at ~2,500 output tokens
      per call, not a request-count effect);
    - the **judge** batch at 8 workers lost 3 of 36 calls, an 8% loss rate for no wall-clock
      gain.

    The 8 here was originally justified by quoting the first measurement as though it were
    the second. It is kept at 8 because it is a guard rail on a mix of stages rather than a
    judge-only cap -- but the judge evidence argues for 4, and this is the number to lower
    first if rate limits appear.

    Phases also self-align, since every task starts at P1 together, so without this the pool
    opens with a thundering herd on the generator.

    `sim` is the one that matters. Every simulation needs the user simulator, which is the
    pipeline's only dedicated deployment; its size should match that deployment's replica
    count. `serverless` is a guard rail on everything else, all of which is serverless.
    """

    # Re-entrant context managers, so an unlimited resource costs nothing at the call site
    # and needs no branch. This was a `hold(which: str)` helper doing `getattr(self, which)`,
    # which re-implemented `nullcontext` and made a typo -- `hold("serverles")` -- an
    # AttributeError deep inside a worker. `with limits.sim:` is checked by attribute access.
    sim: Any = field(default_factory=contextlib.nullcontext)
    serverless: Any = field(default_factory=contextlib.nullcontext)

    @staticmethod
    def of(sim: int = 0, serverless: int = 0) -> "Limits":
        return Limits(
            sim=threading.Semaphore(sim) if sim else contextlib.nullcontext(),
            serverless=(threading.Semaphore(serverless) if serverless
                        else contextlib.nullcontext()),
        )


# --------------------------------------------------------------------------- #
# Disk
# --------------------------------------------------------------------------- #

class Store:
    """Every artifact, and the resume decisions that read them.

    The layout constants live in `src/metaeval/runs.py`, not here, so the writer and every
    reader share one definition: a disagreement about a subdirectory name fails silently, as
    "0 trajectory files" rather than as an error.

    Shared files (instructions, votes, rejects, index) are read-modify-written under one
    lock. They are small and the critical sections hold no I/O to a model, so a single lock
    costs nothing measurable and removes a whole class of lost-update bug: several workers
    finishing P1 at once would otherwise each write a file built from a stale read.
    """

    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- paths --
    @staticmethod
    def _attempt_suffix(attempt: int) -> str:
        """`""` for attempt 1, `.aN` after.

        Attempt 1 has no infix so that a `filter_retries: 0` run produces exactly the
        filenames a single-attempt run would, and any reader written against those keeps
        working. Later attempts get their own files rather than overwriting: a rejected pair's
        trajectories are evidence (they are what `rejects.json` points at), and pair ids are
        uuids, so overwriting would orphan the votes already cast on the earlier attempt.

        The infix sits before the level so `name.split(".")[:3]` still yields the cell key,
        which is how anything that groups files by cell recovers the grouping.
        """
        return "" if attempt <= 1 else f".a{attempt}"

    @property
    def traj_dir(self) -> Path:
        """Trajectories go in their own subdirectory.

        A flat run directory at the full tau3 space holds 2,250 trajectory files beside 1,125
        pair files and six bookkeeping files. The name is `runs.TRAJECTORY_SUBDIR` so the
        writer and every reader share one constant: a disagreement here fails silently, as
        "0 trajectory files" rather than as an error.
        """
        from src.metaeval.runs import TRAJECTORY_SUBDIR
        d = self.out / TRAJECTORY_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    def traj_path(self, cell: Cell, level: str, attempt: int = 1) -> Path:
        return self.traj_dir / f"{cell.key}{self._attempt_suffix(attempt)}.{level}.json"

    @property
    def pairs_dir(self) -> Path:
        """Pairs get their own subdirectory too, for the same reason as trajectories.

        The name is `runs.PAIRS_SUBDIR` so the writer and every reader share one constant;
        `runs.pair_files` reads a flat directory as well, so a run written before the
        subdirectory existed still reads.
        """
        from src.metaeval.runs import PAIRS_SUBDIR
        d = self.out / PAIRS_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    def pair_path(self, cell: Cell, attempt: int = 1) -> Path:
        return self.pairs_dir / f"{cell.key}{self._attempt_suffix(attempt)}.pairs.json"

    def pair_files(self) -> list[Path]:
        """Both layouts, via the one shared reader. Imported lazily like the other `runs`
        uses here, which keeps the two modules' import order free."""
        from src.metaeval.runs import pair_files
        return pair_files(self.out)

    @property
    def checkpoint_path(self) -> Path:
        return self.out / "usage_checkpoints.json"

    def read_checkpoints(self) -> dict:
        return self._read(self.checkpoint_path, {})

    def write_checkpoint(self, run_id: str, record: dict) -> None:
        """Snapshot this invocation's accounting, so a kill costs a window and not the run.

        Its own small file rather than `pipeline_run.json`: that file carries `per_datapoint`,
        which grows with the run, and rewriting it on a short interval is the O(N^2) cost that
        already takes 84s once at 1,125 cells. This blob is ~20 keys per stage.
        """
        with self._lock:
            blob = self.read_checkpoints()
            blob[run_id] = record
            self._write(self.checkpoint_path, blob)

    def clear_checkpoint(self, run_id: str) -> None:
        """Drop this invocation's checkpoint once its real record is in `runs`.

        Leaving it would make the next invocation treat a completed run as a killed one and
        count it twice.
        """
        with self._lock:
            blob = self.read_checkpoints()
            if blob.pop(run_id, None) is not None:
                self._write(self.checkpoint_path, blob)

    @property
    def instr_path(self) -> Path:
        return self.out / "generated_instructions.json"

    @property
    def votes_path(self) -> Path:
        return self.out / "judge_votes.json"

    @property
    def rejects_path(self) -> Path:
        return self.out / "rejects.json"

    @property
    def index_path(self) -> Path:
        return self.out / "pair_index.json"

    # -- generic json --
    @staticmethod
    def _read(p: Path, default):
        if not p.is_file():
            return default
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return default

    @staticmethod
    def _write(p: Path, data) -> None:
        """Write atomically: temp file beside the target, then `os.replace`.

        `write_text` truncates in place, and `_read` returns its default on a
        `JSONDecodeError`. Together that is a silent total-loss path: a kill or ENOSPC
        mid-write leaves `judge_votes.json` truncated, the next `put_vote` reads `{}`, and it
        rewrites the file containing only the new vote -- every prior vote deleted, no error.
        For `pipeline_run.json` it resets `cumulative` to 0, which is precisely the failure
        `cumulative` exists to prevent.

        `os.replace` is atomic on POSIX, so a reader sees either the old file or the new one.
        The temp file is in the same directory because `os.replace` across filesystems is not.
        """
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, p)

    # -- P1 instructions, merged by group key --
    def instructions_for(self, cell: Cell) -> Optional[dict[str, str]]:
        recs = self._read(self.instr_path, {}).get("records", {})
        rec = recs.get(cell.key)
        if not rec:
            return None
        bodies = rec.get("instructions") or {}
        # An empty body is a failed generation, not a completed phase -- otherwise resume
        # would happily simulate a group whose steering text is "".
        if any(not (bodies.get(lvl) or "").strip() for lvl in cell.levels):
            return None
        return bodies

    def put_instructions(self, cell: Cell, bodies: dict[str, str], meta: dict) -> None:
        with self._lock:
            blob = self._read(self.instr_path, {})
            blob.setdefault("records", {})[cell.key] = {
                "domain": cell.domain, "criterion": cell.criterion,
                "task_index": cell.task_index, "task_id": cell.task_id,
                "gap": list(cell.gap), "agent": cell.agent,
                "instructions": bodies, **meta,
            }
            self._write(self.instr_path, blob)

    # -- P2 trajectories, one file per level so a partial group keeps what it has --
    def has_trajectory(self, cell: Cell, level: str, attempt: int = 1) -> bool:
        return self.traj_path(cell, level, attempt).is_file()

    # -- P3 pair --
    def pair_for(self, cell: Cell, attempt: int = 1) -> Optional[dict]:
        """The existing pair for this cell, from either layout.

        **Probing both is what makes "No rewrite of an existing pair file" true for a flat run.**
        Reading only `pairs/` meant resuming any pre-subdirectory run -- including
        `data/step5/pipeline36`, a validated reference the paper quotes -- found nothing, so P3
        rebuilt the pair with a fresh uuid, orphaning the votes already cast on the old id and
        re-spending both judges. The trajectories are flat in those runs too, so P2 re-ran as
        well: ~97% of a task's cost, to recreate something already on disk.
        """
        for path in (self.pair_path(cell, attempt),
                     self.out / self.pair_path(cell, attempt).name):
            data = self._read(path, None)
            if isinstance(data, list) and data:
                return data[0]
        return None

    def put_pair(self, cell: Cell, pair: dict, attempt: int = 1) -> None:
        self._write(self.pair_path(cell, attempt), [pair])

    # -- P4/P5 votes, merged by (pair id, judge) --
    def votes_for(self, pair_id: str) -> dict[str, int]:
        blob = self._read(self.votes_path, {})
        return {v["judge"]: v["side"] for v in blob.get("votes", [])
                if v.get("id") == pair_id and v.get("side") is not None}

    def put_vote(self, pair_id: str, judge: str, model: str, cell: Cell,
                 result: dict) -> None:
        with self._lock:
            blob = self._read(self.votes_path, {})
            votes = {(v["id"], v["judge"]): v for v in blob.get("votes", [])}
            votes[(pair_id, judge)] = {
                "id": pair_id, "judge": judge, "model": model,
                "criterion_name": cell.criterion, "group_key": cell.key,
                "side": result.get("side"), "a_was": result.get("a_was"),
                "attempts": result.get("attempts"),
                "reasoning": result.get("reasoning", ""),
            }
            blob["votes"] = list(votes.values())
            blob["judges"] = blob.get("judges", {}) | {judge: model}
            self._write(self.votes_path, blob)

    # -- rejects, merged by pair id --
    def put_reject(self, pair_id: str, cell: Cell, phase: str, judge: str,
                   side: int, intent: int, attempt: int = 1) -> None:
        with self._lock:
            blob = self._read(self.rejects_path, {})
            items = {r["id"]: r for r in blob.get("rejected", [])}
            items[pair_id] = {
                "id": pair_id, "group_key": cell.key, "phase": phase,
                "rejected_by": judge, "judge_side": side, "intent": intent,
                "criterion": cell.criterion, "domain": cell.domain,
                "gap": list(cell.gap),
                # Which draw this was, and whether a later one went on to succeed. Keyed by
                # pair id so a retried cell keeps one row per attempt rather than losing the
                # earlier ones to the merge.
                "attempt": attempt,
            }
            blob["rejected"] = list(items.values())
            self._write(self.rejects_path, blob)

    def publish_stage_usage(self, cumulative: dict, kept_pairs: int) -> None:
        """Fold the method stages' usage into the artifacts those stages wrote.

        The instruction file and the vote file each gain a `usage` block, so the record of what
        a stage cost sits beside the stage's own output rather than only in the run blob.
        `n_pairs` goes in beside the judge block because it is that stage's per-call
        denominator, and a vote count is not: the cascade judges a different number of pairs
        than it admits.

        **The value written is already cumulative** -- `cumulative["stage_usage"]` is
        `_merge_stage_usage(runs)` over every invocation recorded in `pipeline_run.json`, and by
        the time this runs `runs` also holds whatever `recover_checkpoints` folded in -- so this
        assigns rather than adds. Adding here would double-count every prior invocation.

        The residue: a kill before the first checkpoint (fewer than 100 completed cells) still
        loses that window, because these stages have no per-artifact usage the way trajectories
        do. `recover_checkpoints` explains why.
        """
        stages = cumulative.get("stage_usage") or {}

        instr = stages.get("instruction-gen")
        if instr:
            with self._lock:
                blob = self._read(self.instr_path, {})
                blob["usage"] = {"summary": instr}      # nested: to_json() shape
                self._write(self.instr_path, blob)

        judge = stages.get("judge")
        if judge:
            with self._lock:
                blob = self._read(self.votes_path, {})
                blob["usage"] = judge                    # flat: summary() shape
                blob["n_pairs"] = kept_pairs
                self._write(self.votes_path, blob)

    def n_pairs(self) -> int:
        return sum(len(json.loads(f.read_text()))
                   for f in self.pair_files())

    def write_dataset(self) -> None:
        """Concatenate every pair file into `dataset.json`.

        Rewritten in full each time rather than merged, because it is a derived view of the
        pair files and they are the source of truth.

        **At `filter_retries > 0` this holds more than one pair per cell**, one per draw, and
        that is deliberate -- a rejected draw is a real judged pair and part of the filter's
        denominator. But it means two things for consumers:

        - `run_evaluators.py --dataset` would score every draw, paying up to 3x to score pairs
          that were rejected. Dedupe to the winning draw first, via `pair_index.json`, whose
          `pair_id` and `attempt` name it.
        - draws from one cell are **not independent** -- same task, same criterion, same
          steering text -- so any accuracy statistic must cluster by cell, not by pair.

        Returns nothing; `n_pairs()` and `n_cells()` report the two counts separately so the
        difference between them is visible rather than something a reader has to notice.
        """
        # **Streamed, not accumulated.** Each pair embeds both trajectories in full -- 204KB
        # on disk, and several times that as live Python objects. Building one list held every
        # pair in memory at once: measured 28MB resident for 36 pairs, so ~0.8GB at the 1,125
        # cells of the full tau3 space and ~1.7GB at 2,250. Writing pair-by-pair holds one.
        #
        # Still atomic: the temp file is renamed over the target only after the last pair, so
        # a kill mid-write leaves the previous dataset.json rather than a truncated one.
        target = self.out / "dataset.json"
        tmp = target.with_name(f".{target.name}.tmp")
        with self._lock:
            n = 0
            with tmp.open("w") as fh:
                fh.write("[")
                for f in self.pair_files():
                    try:
                        pairs = json.loads(f.read_text())
                    except (json.JSONDecodeError, OSError):
                        continue
                    for pair in pairs:
                        fh.write(("," if n else "") + "\n")
                        json.dump(pair, fh, indent=2)
                        n += 1
                fh.write("\n]" if n else "]")
            os.replace(tmp, target)

    def n_cells(self) -> int:
        """Distinct cells with at least one pair, ignoring the attempt infix.

        `n_pairs()` counts draws; this counts datapoints. They differ only under retries, and
        printing both is what stops a 52-pair / 36-cell run reading as 52 datapoints.
        """
        return len({".".join(f.name.split(".")[:3])
                    for f in self.pair_files()})

    # -- the index: resume state and the stage-1-to-stage-4 join key --
    def index(self) -> dict:
        return self._read(self.index_path, {})

    def append_draw(self, cell: Cell, attempt: int, pair_id: str, verdict: str,
                    rejected_by: str = "", phase: str = "") -> None:
        """Record one draw's verdict in the cell's own row, keyed by attempt.

        **Every draw's artifacts already survive** -- its trajectories, its pair file, its
        votes (each draw has a distinct pair uuid) and, if it failed, its `rejects.json` row.
        Nothing is discarded. But reading "what happened to this cell" meant joining three
        files, and the one question that needs it most is whether a re-drawn pair behaves like
        a first-draw passer -- the filter's own selection effect. That check should not require
        a join.

        Keyed by attempt so it is idempotent: a resume that re-reads a draw from disk
        overwrites its row rather than appending a duplicate.
        """
        with self._lock:
            blob = self._read(self.index_path, {})
            row = blob.setdefault(cell.key, {})
            draws = {int(d["attempt"]): d for d in row.get("draws", [])}
            draws[int(attempt)] = {"attempt": int(attempt), "pair_id": pair_id,
                                   "verdict": verdict, "rejected_by": rejected_by,
                                   "phase": phase}
            row["draws"] = [draws[k] for k in sorted(draws)]
            blob[cell.key] = row
            self._write(self.index_path, blob)

    def update_index(self, cell: Cell, **fields) -> None:
        with self._lock:
            blob = self._read(self.index_path, {})
            row = blob.get(cell.key, {})
            row.update({
                "domain": cell.domain, "criterion": cell.criterion,
                "task_index": cell.task_index, "task_id": cell.task_id,
                "gap": list(cell.gap), "agent": cell.agent,
            })
            row.update(fields)
            blob[cell.key] = row
            self._write(self.index_path, blob)


# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #

@dataclass
class Outcome:
    cell: Cell
    state: str                       # kept | rejected | failed
    phase: str = ""                  # where it ended
    pair_id: str = ""
    detail: str = ""
    resumed: list[str] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)
    # Which filter attempt produced this outcome. 1 unless `filter_retries` is on.
    # **Recorded on every pair, and the reason is methodological, not cosmetic.** Human
    # alignment is established at retries=0, i.e. over first-draw passers; a retry adds pairs
    # that already failed a draw and are plausibly more marginal. Whether that alignment
    # transfers is checkable only if the attempt number is on record, and it cannot be
    # reconstructed afterwards.
    attempt: int = 1

    def as_dict(self) -> dict:
        return {"state": self.state, "phase": self.phase, "pair_id": self.pair_id,
                "detail": self.detail, "resumed": self.resumed,
                "attempts": self.attempts, "attempt": self.attempt}


def run_task(cell: Cell, stages: Stages, store: Store, judges: dict[str, str],
             limits: Limits | None = None, log: Callable[[str], None] = print,
             filter_retries: int = 0) -> Outcome:
    """One pair, start to finish. Never raises; a failure becomes `state="failed"`.

    Every phase is skipped when its artifact is already on disk, so Ctrl-C resume and
    per-phase failure retry are the same mechanism. That matters most at P5: re-running the
    trajectories to recover one judge call would re-spend ~97% of the task's cost.

    **`filter_retries` re-draws a rejected pair, and defaults to 0.** Measured across two
    independent runs of the same 36 cells: a cell rejected once passes 55% of the time on the
    next draw, against a 69% base rate, and Cohen's kappa on the verdict is 0.21 -- so a
    reject is mostly the draw, not the cell. Recovery is highest exactly where it is wanted
    (task_resolution 60%, communication_clarity 45%).

    Two properties of the loop matter more than the yield:

    - **P1 is generated once and reused by every attempt.** A retry therefore re-draws the
      trajectory under an unchanged intended gap, which is what makes it the *same* datapoint
      redrawn rather than a different one. Regenerating the steering text would re-roll the
      intent and destroy the interpretation. It also gives retry a floor: 5 of 36 cells passed
      in neither run, all of them clarity or task_resolution and none friendliness, so the
      criteria that are hard stay visibly hard even at retries=2. That floor is the
      differential-retention finding surviving, not a shortfall.
    - **Every setting is nested in the highest one.** Attempt 1 is the same draw regardless of
      `filter_retries`, so one run at 2 yields the 0, 1 and 2 results by truncating on the
      recorded attempt number. Running the three settings separately would cost 3x and let
      run-to-run nondeterminism -- which at kappa 0.21 is large -- swamp the effect.
    """
    limits = limits or Limits()
    out = Outcome(cell=cell, state="running")

    from src.metaeval.schema import load_trajectory, save_trajectory

    bodies: Optional[dict[str, str]] = None   # P1, at most once for the whole task
    p1_noted = False

    for attempt in range(1, max(0, filter_retries) + 2):
        out.attempt = attempt
        last = attempt > max(0, filter_retries)

        # -- P3 first: an existing pair short-circuits P1 AND P2 ---------- #
        # A `PairwiseEntry` embeds both trajectories in full, so once the pair exists neither
        # the instructions nor the trajectory files are needed by anything downstream.
        #
        # P1 used to sit above this check, which meant a resumed task whose instruction record
        # was missing (lost file, or an empty body caught by the guard in `instructions_for`)
        # spent a full generator call -- ~6-7k prompt tokens -- and then discarded it, because
        # `bodies` is only read inside the `pair is None` branch. On a finished 1000-pair run
        # with a truncated instructions file that is 1000 wasted calls.
        pair = store.pair_for(cell, attempt)
        if pair is not None:
            tag = "" if attempt == 1 else f"@a{attempt}"
            out.resumed.append(f"P3{tag}")
            if attempt == 1 and not p1_noted:
                out.resumed.append("P1")
                p1_noted = True
            for level in cell.levels:
                out.resumed.append(f"P2{tag}:{level}")
        else:
            if bodies is None:
                bodies = store.instructions_for(cell)
                if bodies is None:
                    try:
                        with limits.serverless:
                            bodies = stages.instructions(cell)
                    except Exception as exc:  # noqa: BLE001
                        return _fail(out, store, "P1", exc, log)
                    store.put_instructions(cell, bodies, {})
                    log(f"  {cell.key:<44} P1 instructions")
                elif not p1_noted:
                    out.resumed.append("P1")
                p1_noted = True

            trajs: dict[str, Any] = {}
            for level in cell.levels:
                tag = "" if attempt == 1 else f"@a{attempt}"
                if store.has_trajectory(cell, level, attempt):
                    out.resumed.append(f"P2{tag}:{level}")
                    trajs[level] = load_trajectory(store.traj_path(cell, level, attempt))
                    continue
                try:
                    with limits.sim:
                        traj, used = stages.trajectory(cell, level, bodies[level])
                except Exception as exc:  # noqa: BLE001
                    return _fail(out, store, "P2", exc, log)
                out.attempts[f"P2{tag}:{level}"] = used
                save_trajectory(traj, store.traj_path(cell, level, attempt))
                trajs[level] = traj
                extra = f" ({used} attempts)" if used > 1 else ""
                log(f"  {cell.key:<44} P2{tag} {level}{extra}")

            try:
                pair = stages.build_pair(cell, trajs)
            except Exception as exc:  # noqa: BLE001
                return _fail(out, store, "P3", exc, log)
            store.put_pair(cell, pair, attempt)
            # The pair embeds both trajectories, so `trajs` is now a second full copy --
            # 116KB to 2.3MB per worker, retained across both judge calls for nothing.
            del trajs

        # Guarded, because `run_task` promises never to raise and `main` catches only
        # `KeyboardInterrupt` -- so one escaping exception here would kill the process before
        # `pipeline_run.json` is written and discard every Outcome already collected. One
        # hand-edited pair file is enough: a missing `correct_response` is a plain KeyError.
        try:
            out.pair_id = pair["id"]
            intent = pair["correct_response"]
        except (KeyError, TypeError) as exc:
            return _fail(out, store, "P3", exc, log)

        # -- P4, P5: the cascade. Sequential on purpose -------------------- #
        # Judge 2 only runs on judge 1's survivors, which is the cascade's whole cost saving
        # (28 of 36 rather than all 36). Running them concurrently would save one call of
        # latency and give that up.
        rejected: Optional[tuple[str, str, Any]] = None
        prior = store.votes_for(out.pair_id)
        for phase, (judge, model) in zip(("P4", "P5"), judges.items()):
            if judge in prior:
                out.resumed.append(f"{phase}:{judge}")
                side = prior[judge]
            else:
                try:
                    with limits.serverless:
                        result = stages.judge(judge, pair, cell)
                except Exception as exc:  # noqa: BLE001
                    return _fail(out, store, phase, exc, log)
                side = result["side"]
                store.put_vote(out.pair_id, judge, model, cell, result)
            if side != intent:
                store.put_reject(out.pair_id, cell, phase, judge, side, intent, attempt)
                rejected = (phase, judge, side)
                break

        if rejected is None:
            out.state, out.phase = "kept", "P5"
            store.append_draw(cell, attempt, out.pair_id, "kept", phase="P5")
            store.update_index(cell, **out.as_dict())
            extra = f" (attempt {attempt})" if attempt > 1 else ""
            log(f"  {cell.key:<44} KEPT{extra}")
            return out

        phase, judge, side = rejected
        store.append_draw(cell, attempt, out.pair_id, "rejected",
                          rejected_by=judge, phase=phase)
        if not last:
            log(f"  {cell.key:<44} {phase} rejected by {judge}, "
                f"re-drawing ({attempt}/{filter_retries + 1})")
            # The index is updated per attempt too, so a Ctrl-C between draws leaves a record
            # of how far this cell got rather than looking untouched.
            out.state, out.phase = "retrying", phase
            store.update_index(cell, **out.as_dict())
            continue

        out.state, out.phase = "rejected", phase
        out.detail = f"{judge} chose {side}, intent {intent}"
        store.update_index(cell, **out.as_dict())
        extra = f" after {attempt} attempt(s)" if attempt > 1 else ""
        log(f"  {cell.key:<44} {phase} REJECTED by {judge}{extra}")
        return out

    # Unreachable: the loop returns on every terminal state and `range` always yields at
    # least attempt 1. Present so a future edit to the bounds cannot fall through to None.
    return _fail(out, store, out.phase or "P4",
                 RuntimeError("attempt loop exhausted without a verdict"), log)


def _fail(out: Outcome, store: Store, phase: str, exc: BaseException,
          log: Callable[[str], None]) -> Outcome:
    out.state, out.phase = "failed", phase
    out.detail = f"{type(exc).__name__}: {str(exc)[:200]}"
    store.update_index(out.cell, **out.as_dict())
    log(f"  {out.cell.key:<44} {phase} FAILED {out.detail[:70]}")
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def strata(outcomes: Iterable[Outcome]) -> dict[str, dict[str, dict]]:
    """Cells, draws and kept per stratum. **Reported, never enforced.**

    Retention is a finding: 100% with high evaluator accuracy means a criterion too easy to
    discriminate, low retention with low accuracy means one that is hard to label *and* hard
    to score. But it is only a finding if the denominator is on record -- otherwise
    "friendliness is easy" cannot be told apart from "we generated more friendliness".

    `cells` and `attempts` differ only when `filter_retries` is on, and keeping them apart is
    what preserves the finding under retries. Retention is per *cell* (did this cell ever yield
    a pair) while the cost multiplier is per *draw*, so a criterion that needs three draws per
    pair reads as expensive rather than as free -- the signal migrates from the retention
    column into `attempts_per_kept` instead of vanishing.
    """
    axes = {"criterion": lambda o: o.cell.criterion,
            "domain": lambda o: o.cell.domain,
            "gap": lambda o: "-".join(o.cell.gap)}
    out: dict[str, dict[str, dict]] = {}
    items = list(outcomes)
    for axis, keyfn in axes.items():
        counts: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        for o in items:
            c = counts[keyfn(o)]
            c["cells"] += 1
            c["attempts"] += max(1, getattr(o, "attempt", 1))
            c[o.state] += 1
        out[axis] = {}
        for name, c in sorted(counts.items()):
            kept, cells, att = c["kept"], c["cells"], c["attempts"]
            out[axis][name] = {
                "cells": cells,
                "attempts": att, "kept": kept, "rejected": c["rejected"],
                "failed": c["failed"],
                "retention": round(kept / cells, 3) if cells else None,
                # The multiplier a reader needs to size a run: 1.71x for task_resolution
                # against 1.00x for friendliness in the serial run. Counts re-draws, so it
                # keeps rising with `filter_retries` even as retention approaches 1.0.
                "attempts_per_kept": round(att / kept, 2) if kept else None,
            }
    return out


def _stage_block(block: dict) -> dict:
    """One stage's usage, with the wall-clock fields removed.

    `UsageMeter.by()` deliberately gives every group the parent's window, because the groups
    ran interleaved and splitting elapsed time between them would be fiction. That is right
    for the meter and wrong to write to disk under a per-stage key: a reader would copy
    `wall_s` and `concurrency` and present them as that stage's. So they are nulled here, and
    `interval` says why.
    """
    out = dict(block)
    for key in ("wall_s", "concurrency", "out_tps"):
        out[key] = None
    out["interval"] = "overlapped"
    return out


def _merge_stage_usage(runs: list[dict]) -> dict:
    """Sum the per-stage blocks across every invocation that built this dataset."""
    keys = ("n_calls", "n_ok", "n_error", "prompt_tokens", "completion_tokens",
            "reasoning_tokens", "total_tokens", "content_chars", "service_s")
    roles = {role for r in runs for role in (r.get("stage_usage") or {})}
    out: dict[str, dict] = {}
    for role in sorted(roles):
        blocks = [(r.get("stage_usage") or {}).get(role) or {} for r in runs]
        agg = {k: sum(b.get(k) or 0 for b in blocks) for k in keys}
        agg["service_s"] = round(agg["service_s"], 1)
        agg["reasoning_share"] = (round(agg["reasoning_tokens"] / agg["completion_tokens"], 3)
                                  if agg["completion_tokens"] else None)
        agg["interval"] = "overlapped"
        out[role] = agg
    return out


def run_record(meter, run_cfg: dict, n_tasks: int, run_id: str) -> dict:
    """One invocation's accounting, the unit `cumulative` sums over.

    Factored out because it is built twice: once at the end, and once per checkpoint while the
    run is still going. Two copies of this dict drifting apart would make a recovered record
    incomparable with a completed one.
    """
    return {"run_id": run_id, "config": run_cfg, "n_tasks": n_tasks,
            "usage": meter.summary(),
            "stage_service_s": meter.stage_service(),
            # Per stage, calls that spent tokens and recorded no latency. Decides whether
            # service time may be quoted as a total or only as a floor -- measured, so the
            # caveat appears exactly while it is true. See `UsageMeter.untimed_calls`.
            "untimed_calls": meter.untimed_calls(),
            # Per-stage blocks, so the two "method" stages can be reported on their own.
            # `by("role")` gives each group the parent's wall clock, which is meaningless
            # per stage once stages overlap -- `_stage_block` nulls it rather than
            # shipping a number whose name lies.
            "stage_usage": {role: _stage_block(b)
                            for role, b in meter.by("role").items()}}


def recover_checkpoints(store: "Store", runs: list[dict], this_run_id: str) -> list[dict]:
    """Fold in any invocation that was killed before it published.

    **The one accounting hole a resume could not close.** `cumulative` sums the records in
    `runs`, and a record only lands there when a run reaches its final write -- so an
    invocation killed mid-flight contributes nothing, and the tokens it really spent are
    unrecoverable from any file. Measured on the full tau3 run: the first pass built ~470 cells
    and was killed, and afterwards `generated_instructions.json` described 657 instruction
    calls beside 1,125 records. Every method-stage cost figure was ~40% low, and the only way
    to quote a true number was to scale cost-per-call by the record count -- an estimate
    standing in for a measurement already paid for.

    Trajectories never had this problem because tau3 writes usage onto every turn: the cost is
    in the artifact, so it survives any kill. Stages 1 and 3 have no such per-artifact record,
    so the meter is checkpointed instead, and an orphaned checkpoint -- one whose `run_id` never
    appears in `runs` -- is exactly a killed invocation.

    Keyed by `run_id`, so a checkpoint is never counted beside the completed record it preceded.
    """
    seen = {r.get("run_id") for r in runs if r.get("run_id")}
    recovered = []
    for rid, rec in sorted((store.read_checkpoints() or {}).items()):
        if rid == this_run_id or rid in seen:
            continue
        rec = dict(rec)
        rec["recovered_from_checkpoint"] = True
        recovered.append(rec)
    if recovered:
        lost = sum((r.get("usage") or {}).get("total_tokens") or 0 for r in recovered)
        print(f"  recovered {len(recovered)} killed invocation(s) from checkpoints: "
              f"{lost:,} tokens that would otherwise be missing")
    return recovered



def print_report(blob: dict, outcomes: list[Outcome]) -> None:
    """Print the run, through `runs.print_summary`.

    **The blob is built first and printed from, rather than printed from live objects.** The
    renderer takes the same bytes that go into `run.json`, so the terminal output and the
    saved run cannot drift apart, and re-rendering a finished run needs no live state -- which
    is what makes the printer testable at all.

    It also keeps this file out of the reporting business: `runs.py` holds the layout and the
    summary, computes no cost and no accuracy, and is the only module `run_pipeline` needs for
    either.

    The failure list stays here: it needs `Outcome.detail`, and it is the one thing a reader
    wants while the run is still fresh.
    """
    from src.metaeval.runs import print_summary

    print_summary(blob)

    failed = [o for o in outcomes if o.state == "failed"]
    if failed:
        print(f"\n  failures, resumable -- re-run the same command:")
        for o in failed:
            print(f"    {o.cell.key:<44} {o.phase} {o.detail[:60]}")


# --------------------------------------------------------------------------- #
# Real stages
# --------------------------------------------------------------------------- #

def build_cells(domains: list[str], criteria: list[str],
                task_indices: Optional[list[int]], seed: int, store: Store) -> list[Cell]:
    """The task list, with gaps assigned once and preserved across resumes.

    The gap must not be recomputed per invocation: `balanced_cycle` balances over the combos
    it is given, so running a subset later would reassign gaps that trajectories on disk
    were already generated for. Any gap already in `pair_index.json` wins.

    **`task_indices=None` means every task in each domain**, which is the full tau3 base
    space: 375 tasks over the four domains, so 1,125 cells at three criteria. Domains differ
    in size (airline 50, banking_knowledge 97, retail 114, telecom 114), so the indices are
    per domain rather than shared -- taking a fixed range across all of them would silently
    drop two thirds of retail and telecom.
    """
    from src.metaeval.sources.tau2_source import domain_tasks, get_task, pick_agent
    from src.metaeval.steering.gaps import GAPS, balanced_cycle

    combos = []
    for d in domains:
        idxs = task_indices if task_indices is not None else range(len(domain_tasks(d)))
        combos.extend((d, c, i) for c in criteria for i in idxs)
    gaps: dict[tuple, tuple] = {}
    for criterion in criteria:
        cells = [x for x in combos if x[1] == criterion]
        for combo, gap in zip(cells, balanced_cycle(len(cells), list(GAPS),
                                                    seed=seed, salt=criterion)):
            gaps[combo] = tuple(gap)

    index = store.index()
    out = []
    for domain, criterion, idx in combos:
        task = get_task(domain, index=idx)
        prior = index.get(f"{domain}.{criterion}.t{idx}", {})
        # `pair_index.json`'s stored gap wins over a freshly cycled one -- that is the whole
        # point of storing it. A run directory written before this key was named `gap` will
        # re-cycle instead of resuming; regenerate rather than hand-edit, since a half-migrated
        # index changes which levels a cell is simulated at.
        gap = tuple(prior.get("gap") or gaps[(domain, criterion, idx)])
        out.append(Cell(
            domain=domain, criterion=criterion, task_index=idx, task_id=str(task.id),
            gap=gap,
            agent=prior.get("agent") or pick_agent(domain, str(task.id), criterion, seed),
        ))
    return out


def real_stages(meter: UsageMeter, generator: str, judges: dict[str, str],
                seed: int, sim_attempts: int, sim_backoff_s: float,
                criteria: dict | None = None) -> Stages:
    """The four model-facing stages, wired to the live call paths."""
    from src.metaeval.judge import judge_side
    from src.metaeval.schema.pairs import pairs_from_group
    from src.metaeval.sources.tau2_source import (
        DOMAINS, get_task, run_steered_resilient)
    from src.metaeval.steering import CRITERIA
    from src.metaeval.steering.criteria import STEERING_WRAPPER
    from src.metaeval.steering.generate import generate_instructions
    from src.metaeval.schema.convert import build_context
    from src.metaeval.sources.tau2_source import task_description

    crit = criteria if criteria is not None else CRITERIA

    def instructions(cell: Cell) -> dict[str, str]:
        task = get_task(cell.domain, task_id=cell.task_id)
        ctx = build_context(cell.domain, task,
                           retrieval_config=DOMAINS[cell.domain]["retrieval"])
        g = generate_instructions(
            crit[cell.criterion], task_description(task),
            [t.name for t in ctx.agent_tools],
            generator_model=generator, domain=cell.domain, task_id=cell.task_id,
            meter=meter,
            # All five phases must record under ONE key or the per-datapoint chain cannot be
            # assembled. Measured on a 2-pair smoke run before this: 5 accounting keys for 2
            # pairs -- instruction-gen keyed "{domain}:{task_id}", which is not unique per
            # datapoint (one task carries all three criteria, so both P1 calls merged into
            # one row), trajectories keyed by group, judges by pair uuid.
            item_id=cell.key,
        )
        return g.instructions

    def trajectory(cell: Cell, level: str, body: str):
        task = get_task(cell.domain, task_id=cell.task_id)

        def on_retry(attempt: int, exc: BaseException, elapsed_s: float) -> None:
            """Record what a lost simulation cost.

            The conversation is gone -- the exception took it -- so its tokens are unknowable
            and stay 0, exactly as `record_error` treats every other failure. The elapsed time
            is real. Without this the most expensive stage in the pipeline reported its
            failures nowhere: `n_error` stayed 0 and `item_stage_matrix`'s `retries` was
            structurally blind to simulation retries, which is the one case its docstring
            promises to distinguish.
            """
            meter.record_error(exc, model=f"tau3:sim:{cell.agent}", latency_s=elapsed_s,
                               role="trajectory", label=level, item_id=cell.key,
                               attempt=attempt)

        traj, used = run_steered_resilient(
            cell.domain, task, cell.criterion, level, cell.agent, seed=seed,
            instruction=STEERING_WRAPPER.format(instruction=body.strip()),
            # The tag must carry the task index: tau3's registry is keyed by agent name
            # with no removal, so without it the three tasks of one (domain, criterion)
            # collide and registration correctly refuses to overwrite.
            instruction_tag=f"{cell.domain}-{cell.criterion}-t{cell.task_index}-{level}",
            attempts=sim_attempts, backoff_s=sim_backoff_s, on_retry=on_retry,
        )
        _record_trajectory_usage(meter, traj, cell, level, attempt=used)
        return traj, used

    def build_pair(cell: Cell, trajs: dict[str, Any]) -> dict:
        worse, better = cell.levels
        pairs = pairs_from_group(
            trajs, criterion_name=cell.criterion,
            criterion_description=crit[cell.criterion].description,
            pair_levels=[(worse, better)])
        return pairs[0].model_dump(mode="json")

    def judge(name: str, pair: dict, cell: Cell) -> dict:
        # Keyed by group, not by pair uuid: see the note in `instructions` above. The pair's
        # own id is still recorded in the vote and in `pair_index.json`.
        return judge_side(pair, judges[name], name, meter=meter, item_id=cell.key)

    return Stages(instructions=instructions, trajectory=trajectory,
                  build_pair=build_pair, judge=judge)


def _record_trajectory_usage(meter: UsageMeter, traj: Any, cell: Cell,
                             level: str, attempt: int = 1) -> None:
    """Fold tau3's own per-turn accounting into our meter.

    Trajectory calls happen inside tau3 and never pass through `timed_call`, so without
    this the most expensive stage in the pipeline contributes nothing to the per-pair
    roll-up. tau3 records `usage` on every message it generates and
    `generation_time_seconds` on assistant messages, which is enough to reconstruct the
    stage as synthetic records.

    Latency used to be missing for every user-simulator turn: tau3's role flip copied
    `cost`, `usage` and `raw_data` off the timed response and dropped
    `generation_time_seconds`, so 473 of 473 user calls were counted in tokens and not in
    seconds, leaving 18% of simulation time unattributable. Fixed by patches/tau3.patch at
    `tau2-bench/src/tau2/user/user_simulator.py`, so this now reads a real latency for user
    turns too -- but only for trajectories generated after that fix. Whether a given run's
    service time is a total or a floor is therefore a property of the data, counted by
    `UsageMeter.untimed_calls` and reported from that rather than asserted.

    `attempt` stamps the surviving trajectory with the attempt that produced it, so a pair
    that is only expensive because it retried is distinguishable from one that is genuinely
    slow -- which is what `item_stage_matrix`'s `retries` promises and could not deliver while
    every trajectory record claimed to be attempt 1.
    """
    from src.metaeval.usage import CallRecord

    for turn in getattr(traj, "turns", []) or []:
        extra = getattr(turn, "extra", None) or {}
        if not isinstance(extra, dict):
            continue
        usage = extra.get("usage") or {}
        if not usage:
            continue
        secs = extra.get("generation_time_seconds")
        meter.add(CallRecord(
            model=f"tau3:{getattr(turn, 'role', '?')}",
            role="trajectory", label=level, item_id=cell.key, attempt=attempt,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=float(secs) if isinstance(secs, (int, float)) else 0.0,
            reasoning_source="unavailable",
        ))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def smoke_out(out: Path, limit: Optional[int]) -> Path:
    """A smoke run gets its own directory, so it cannot contaminate a real one.

    Without this, `limit: 2` wrote its two pairs, its `dataset.json` and above all its
    `pipeline_run.json` into the same default directory a full run uses -- and
    `pipeline_run.json` *accumulates* by design, so the smoke run's tokens would be folded
    into `cumulative` and quoted as part of what the real dataset cost. That is the same
    failure `run_evaluators.py:133` documents, where a 3-pair table became indistinguishable
    from the real 36-pair sweep and then persistent.

    **A limited run is a smoke run whatever the out-dir says.** This used to exempt an explicit
    out-dir, on the reasoning that typing one was a deliberate act -- true while it was a flag,
    false once the flags moved into the YAML, because `out_dir` is then always written and never
    equals the built-in default. The redirect therefore never fired, and every `limit` run went
    straight into the real directory, reintroducing exactly the contamination above. It is now
    unconditional on `limit`.
    """
    if limit:
        return out / f"smoke-{limit}"
    return out


def main() -> int:
    from src.metaeval.sources.tau2_source import AGENTS, USER, ensure_env, warm

    ap = argparse.ArgumentParser(
        description="Generate meta-evaluation data pairs. All parameters come from the YAML "
                    "config; --dry-run describes the invocation, not the run.")
    ap.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and what resume would skip; no LLM calls")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ensure_env()
    out_dir = smoke_out(cfg.out_dir, cfg.limit)
    store = Store(out_dir)
    cells = build_cells(cfg.domains, sorted(cfg.criteria), cfg.task_indices, cfg.seed, store)
    if cfg.limit:
        cells = cells[:cfg.limit]
        print(f"*** limit {cfg.limit}: SMOKE TEST on {len(cells)} cell(s), "
              f"not a measurement ***")
        if out_dir != cfg.out_dir:
            print(f"    writing to {out_dir} so the real run is not contaminated")

    judges = dict(cfg.judges)
    draws = cfg.filter_retries + 1
    print(f"config: {a.config}")
    print(f"{len(cells)} task(s) -> at most "
          f"{sum(len(c.levels) for c in cells) * draws} simulations, "
          f"{len(cells)} instruction calls, "
          f"{len(cells) * len(judges) * draws} judge calls")
    if cfg.filter_retries:
        print(f"*** filter_retries {cfg.filter_retries}: a rejected pair is re-drawn up to "
              f"{cfg.filter_retries} more time(s). P1 is reused, so a re-draw is the same "
              f"datapoint redrawn.")
    print(f"out: {out_dir}")

    if a.dry_run:
        return _dry_run(cells, store, judges, cfg.filter_retries)

    if not cfg.warm:
        pass
    else:
        # Once, before the pool. Never from inside a task: `warm` is a serial loop with a
        # 30s sleep, so N workers warming would serialise the pool behind it.
        print("pre-warming:")
        if warm([m for m, _ in AGENTS.values()] + [USER, cfg.generator]):
            print("a deployment did not come up; re-run, or raise its replica count")
            return 1

    # Identifies this invocation in `runs` and in `usage_checkpoints.json`, so a checkpoint is
    # never counted beside the completed record it preceded.
    run_id = uuid.uuid4().hex[:12]
    # Assigned before the pool, not after it: the periodic checkpoint inside the loop reads it,
    # and while it was assigned below, the 100th completed task raised `UnboundLocalError` --
    # which the loop's `except KeyboardInterrupt` does not catch, so it escaped `main` and took
    # the run down before `pipeline_run.json`, `write_dataset` or `publish_stage_usage` ran. A
    # checkpoint meant to bound a kill to 100 cells instead guaranteed one at cell 100.
    run_cfg = cfg.as_record
    meter = UsageMeter().start()
    limits = Limits.of(cfg.sim_concurrency, cfg.serverless_concurrency)
    stages = real_stages(meter, cfg.generator, judges, cfg.seed,
                         cfg.sim_attempts, cfg.sim_backoff, cfg.criteria)

    outcomes: list[Outcome] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Bare executor, not a `with`: Ctrl-C must keep the outcomes already collected, and the
    # per-phase artifacts are already on disk so a re-run resumes.
    ex = ThreadPoolExecutor(max_workers=cfg.workers)
    try:
        futures = {ex.submit(run_task, c, stages, store, judges, limits,
                             filter_retries=cfg.filter_retries): c
                   for c in cells}
        for n, fut in enumerate(as_completed(futures), 1):
            outcomes.append(fut.result())
            if n % 10 == 0 or n == len(futures):
                print(f"  --- {n}/{len(futures)} tasks done")
            # Checkpoint the meter so a kill costs at most this window instead of the whole
            # invocation. Every 100 rather than every 10: the write is small, but it takes the
            # store lock that the workers' own index writes need.
            if n % 100 == 0:
                store.write_checkpoint(run_id, run_record(meter, run_cfg, n, run_id))
        ex.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\ninterrupted; every finished phase is on disk. Re-run to resume.")
        ex.shutdown(wait=False, cancel_futures=True)
    meter.stop()

    # Configuration travels with the numbers: a pipeline clock without its concurrency is
    # not reproducible, and concurrency also moves result stability, since temperature 0 is
    # not bitwise reproducible on Fireworks and MoE routing depends on batch composition.
    #
    # **A resume must not erase what the data cost.** Tokens are a fact about an invocation,
    # but "what did this dataset cost to produce" is the sum over every invocation that built
    # it -- and a no-op resume reports 0. Overwriting wholesale replaced 158,718 real tokens
    # with 0 in the smoke run. So each invocation appends to `runs`, and `cumulative` sums
    # them. Cumulative wall clock is labelled as a sum of separate sessions, not as one
    # elapsed figure, because that is what it is.
    #
    # `summary()` is computed once and threaded through. It walks every record to build a
    # ~20-key aggregate, and this used to call it four times over the same list.
    usage = meter.summary()
    this_run = run_record(meter, run_cfg, len(outcomes), run_id)

    prior = Store._read(store.out / "pipeline_run.json", {})
    runs = list(prior.get("runs") or [])
    runs += recover_checkpoints(store, runs, run_id)
    runs += [this_run]
    cum_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
                "n_calls", "n_error")
    cumulative = {k: sum((r.get("usage") or {}).get(k) or 0 for r in runs)
                  for k in cum_keys}
    cumulative["session_wall_s"] = round(
        sum((r.get("usage") or {}).get("wall_s") or 0 for r in runs), 1)
    cumulative["n_invocations"] = len(runs)
    # Accumulated across invocations for the same reason the token totals are: a resume that
    # redoes nothing has an empty meter, and writing its per-stage and per-datapoint views
    # over the real ones erases what the data cost. Observed on three of four run
    # directories before this: `per_datapoint` empty, `stage_service_s` `{}`.
    cumulative["stage_usage"] = _merge_stage_usage(runs)
    cumulative["stage_service_s"] = {
        role: round(sum((r.get("stage_service_s") or {}).get(role) or 0 for r in runs), 1)
        for role in {k for r in runs for k in (r.get("stage_service_s") or {})}}
    cumulative["per_datapoint"] = {**(prior.get("cumulative") or {}).get("per_datapoint", {}),
                                   **meter.item_stage_matrix()}
    # Summed across invocations: a directory built partly before the patch and partly after
    # still contains untimed calls, so its trajectory service time is still a floor. Taking
    # only the latest invocation would clear the caveat while the data it describes remains.
    cumulative["untimed_calls"] = {
        role: sum((r.get("untimed_calls") or {}).get(role) or 0 for r in runs)
        for role in {k for r in runs for k in (r.get("untimed_calls") or {})}}

    # Built once, then both printed and written. `outcomes` is deliberately NOT duplicated in
    # here: `pair_index.json` is the single record of per-cell state, and a second copy that
    # only covers the latest invocation would disagree with it after any partial re-run.
    blob = {
        **this_run,
        "kept": sum(1 for o in outcomes if o.state == "kept"),
        "rejected": sum(1 for o in outcomes if o.state == "rejected"),
        "failed": sum(1 for o in outcomes if o.state == "failed"),
        "per_datapoint": meter.item_stage_matrix(),
        "strata": strata(outcomes),
        # What the dataset cost, across every invocation that built it.
        "cumulative": cumulative,
        "runs": runs,
    }
    print_report(blob, outcomes)
    Store._write(store.out / "pipeline_run.json", blob)
    # This invocation's record is now in `runs`, so its checkpoint would be a duplicate.
    store.clear_checkpoint(run_id)

    # **Put each method stage's usage next to that stage's own output.** A run whose cost is
    # recorded only in the run blob invites the reading that the stage was never measured --
    # which is how an earlier version of this project ended up telling a reader to re-spend
    # money on numbers already sitting in the same directory.
    #
    # The two shapes differ -- `instruction_usage` NESTED under `usage.summary`, `judge_usage`
    # FLAT under `usage` -- because each matches the file it is folded into. Left as-is rather
    # than unified: changing either shape changes a file format on disk.
    #
    # Written from `cumulative`, never from this invocation: a resume with an empty meter would
    # otherwise publish zeros over the real numbers.
    store.publish_stage_usage(cumulative, kept_pairs=blob["kept"] + blob["rejected"])

    # **`dataset.json` is the flat concatenation of the pair files, and it is the input to
    # `build_eval_dataset.py`.** Written at the end of every run, including a resumed one, so a
    # run directory is self-describing without re-walking the pair files by hand. Note what it
    # is *not*: at `filter_retries > 0` it holds every draw, not the winning ones, which is why
    # nothing points an evaluator at it directly -- see `write_dataset`.
    store.write_dataset()
    n_pairs, n_cells = store.n_pairs(), store.n_cells()
    print(f"wrote {store.out}/dataset.json ({n_pairs} pairs) for the report tools")
    if n_pairs != n_cells:
        # Only reachable under filter_retries. Said out loud because "52 pairs" over 36
        # datapoints is not 52 datapoints, and every accuracy statistic downstream has to
        # cluster by cell rather than by pair.
        print(f"  NOTE {n_pairs} pairs over {n_cells} cells -- re-drawn cells contribute "
              f"more than one. Dedupe via pair_index.json before an evaluator sweep,")
        print(f"  and cluster by cell for any accuracy figure: draws from one cell share a "
              f"task, a criterion and their steering text.")
    print(f"\nwrote {store.out}/pipeline_run.json")
    if len(runs) > 1:
        print(f"  cumulative over {len(runs)} invocation(s): "
              f"{cumulative['total_tokens']:,} tokens, {cumulative['n_calls']} calls "
              f"-- this run added {usage['total_tokens']:,}")
    return 0


def _dry_run(cells: list[Cell], store: Store, judges: dict[str, str],
             filter_retries: int = 0) -> int:
    """The plan, and what resume would skip. Zero LLM calls.

    The shared files are read **once**, not once per cell: they grow with the run, and
    re-parsing them per cell made a "plan only" command take tens of seconds on a large
    directory -- with each pair file parsed twice on top of that.

    Two ways this used to disagree with the actual run, both found by pointing it at the
    finished 36-pair directory, and both in the direction that overstates the work:

    - **A cell rejected at P4 was listed as owing P5.** The cascade stops at the first veto, so
      judge2 never runs on that pair -- it is not pending, it is moot. The finished run planned
      as "P5 x9" for nine calls that can never happen.
    - **Retries were not modelled at all.** At `filter_retries: 2` over that same directory the
      real work is 22+ simulations re-drawing the 11 rejects, which the plan showed nowhere.

    A plan a spend decision is made from has to be right about both, so terminal state now
    comes from `pair_index.json` -- the run's own record of what it concluded -- rather than
    being inferred from which artifacts happen to exist.
    """
    # The **records**, not just their keys: resume rejects a record whose body is empty
    # ("An empty body is a failed generation, not a completed phase"), and a key-membership
    # test does not. That made the plan disagree with the run in both directions -- a blank
    # instruction planned as "skips P1" and then spent the generator call.
    instr_records = Store._read(store.instr_path, {}).get("records", {})
    votes_by_pair: dict[str, set] = {}
    for v in Store._read(store.votes_path, {}).get("votes", []):
        if v.get("side") is not None:
            votes_by_pair.setdefault(v.get("id"), set()).add(v.get("judge"))
    index = store.index()

    print(f"\n{'cell':<44} {'gap':<11} {'agent':<20} would do")
    print("-" * 92)
    todo = collections.Counter()
    redraws = 0
    for c in cells:
        state = (index.get(c.key) or {}).get("state")
        attempt = int((index.get(c.key) or {}).get("attempt") or 1)

        if state == "kept":
            print(f"{c.key:<44} {'-'.join(c.gap):<11} {c.agent:<20} "
                  f"nothing, complete (kept"
                  f"{f' on draw {attempt}' if attempt > 1 else ''})")
            continue

        if state == "rejected":
            # Terminal at retries=0. At retries>0 the remaining draws are the work, and the
            # count is an UPPER bound: a draw that passes ends the cell early.
            left = max(0, filter_retries - (attempt - 1))
            if not left:
                print(f"{c.key:<44} {'-'.join(c.gap):<11} {c.agent:<20} "
                      f"nothing, complete (rejected)")
                continue
            redraws += left
            todo["P2"] += left * len(c.levels)
            todo["P3"] += left
            todo["P4"] += left
            print(f"{c.key:<44} {'-'.join(c.gap):<11} {c.agent:<20} "
                  f"REJECTED, up to {left} re-draw(s): "
                  f"{left * len(c.levels)} sims + judges")
            continue

        done, pending = [], []
        pair = store.pair_for(c)          # once, not twice
        bodies = (instr_records.get(c.key) or {}).get("instructions") or {}
        have_p1 = bool(bodies) and all((bodies.get(lvl) or "").strip() for lvl in c.levels)
        (done if have_p1 else pending).append("P1")
        for lvl in c.levels:
            (done if store.has_trajectory(c, lvl) else pending).append(f"P2:{lvl}")
        (done if pair else pending).append("P3")
        prior = votes_by_pair.get(pair["id"], set()) if pair else set()
        for phase, judge in zip(("P4", "P5"), judges):
            (done if judge in prior else pending).append(f"{phase}:{judge}")
        for item in pending:
            todo[item.split(":")[0]] += 1
        skip = f"  (resume skips {len(done)})" if done else ""
        print(f"{c.key:<44} {'-'.join(c.gap):<11} {c.agent:<20} "
              f"{' '.join(pending) if pending else 'nothing, complete'}{skip}")

    print(f"\nwould run: " + (", ".join(f"{k} x{v}" for k, v in sorted(todo.items()))
                              or "nothing"))
    if redraws:
        print(f"  {redraws} re-draw(s) across the rejected cells -- an UPPER bound, since a "
              f"draw that passes ends that cell.")
        print(f"  Measured recovery is ~55% per draw, so expect roughly "
              f"{round(redraws * 0.55)} of them to be spent.")
    print("no LLM calls made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
