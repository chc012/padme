"""The τ³ coupling is three functions wide, asserted rather than claimed.

**Why this is a test.** "Narrow boundary" is the load-bearing claim behind every statement in
the docs that another substrate could be added, and it is the kind of claim that decays
silently: one `from tau2...` added to the generator for a quick fix, and the boundary is four
functions wide with nothing anywhere saying so. The check is cheap and the drift is invisible
otherwise.

Deliberately *not* checked here: that a second substrate would work. It would not, without
also writing its own trajectory conversion. See `src/metaeval/sources/base.py`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from src.metaeval.sources import base, tau2_source

ROOT = Path(__file__).resolve().parents[1]


def test_tau2_source_satisfies_the_substrate_protocol():
    """Structural conformance, by name. `runtime_checkable` checks no more than that, which is
    why the signature test below exists as well."""
    assert isinstance(tau2_source, base.Substrate)


def test_the_signatures_agree_where_the_pipeline_relies_on_them():
    """The pipeline passes `run_steered` its arguments by keyword, so a renamed or dropped
    keyword is a `TypeError` at call time -- during generation, after the substrate has already
    been warmed and paid for. Compared here instead.

    `self` is dropped from the protocol side: `tau2_source` satisfies this as a module of
    functions, not as an instance, which `base.py` explains.
    """
    for name in ("domain_tasks", "task_description", "run_steered"):
        want = list(inspect.signature(getattr(base.Substrate, name)).parameters)[1:]
        got = list(inspect.signature(getattr(tau2_source, name)).parameters)
        assert want == got, f"{name} drifted from the Substrate protocol: {want} vs {got}"


def test_only_the_three_boundary_modules_import_tau3():
    """The boundary itself, as an allowlist rather than as a single file.

    Three modules touch τ³ and each one has to: `tau2_source` runs the simulations,
    `schema/convert.py` turns τ³'s simulation object into our `Trajectory`, and
    `steering/agent.py` registers a steered agent in τ³'s own registry. A substrate owns its
    conversion and its agent wiring -- `sources/base.py` says so -- so pretending the coupling
    is one file wide would be a nicer sentence and a false one.

    What this pins is that the list does not grow. Everything else in `src/metaeval`, and every
    tool, must reach τ³ through these three, so this greps the tree rather than trusting the
    layout.

    Import *strings* rather than import machinery, because the imports that matter are the lazy
    ones -- all three do their `from tau2...` inside functions to keep the schema importable
    without τ³ installed, and a lazy import elsewhere would be equally invisible to an
    import-time check.
    """
    allowed = {
        "src/metaeval/sources/tau2_source.py",   # runs the simulations
        "src/metaeval/schema/convert.py",        # τ³ simulation -> our Trajectory
        "src/metaeval/steering/agent.py",        # registers the steered agent with τ³
    }
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in allowed:
            continue
        text = path.read_text()
        if "import tau2" in text or "from tau2" in text:
            offenders.append(rel)
    assert not offenders, (
        f"{offenders} import τ³ directly. The substrate boundary is the three modules in "
        f"`allowed`; route it through one of them so the coupling stays where it is documented.")


def test_no_tool_imports_tau3_at_all():
    """Stronger than the allowlist, and separate because it is the part a reader checks.

    A tool that imports τ³ cannot run without it, so `tools/build_eval_dataset.py` and
    `tools/run_evaluators.py` -- the entire evaluator half of the repo -- would acquire a
    dependency the docs promise they do not have. `run_pipeline.py` needs τ³ and still must not
    import it: it goes through `sources/`, which is what keeps the substrate swappable in
    principle and what makes the import-time cost of τ³ land in one place.
    """
    offenders = [p.name for p in sorted((ROOT / "tools").glob("*.py"))
                 if "import tau2" in p.read_text() or "from tau2" in p.read_text()]
    assert not offenders, (
        f"{offenders} import τ³ directly; tools must go through src/metaeval/sources/")
