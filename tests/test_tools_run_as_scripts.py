"""Every tool must survive `python tools/<name>.py`.

**The gap this closes.** The suite imports tools as modules, which puts the repo root on
`sys.path` via pytest's `pythonpath` setting -- so a `from src.metaeval... import ...` passes
every test while being fatal in the only way anyone actually runs these:

    $ python tools/run_pipeline.py --config config/pipeline.yaml
    ModuleNotFoundError: No module named 'src'

That shipped once. Three tools had no `sys.path.insert`, so a shared import added to a helper
module broke them as scripts while the suite stayed green. `--help` is enough to catch it:
argparse exits before doing any work, but only after the module has fully imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = sorted(p for p in (ROOT / "tools").glob("*.py") if p.name != "__init__.py")

# Import-time failures. A tool may legitimately exit non-zero on `--help` (a few have required
# positional args and argparse errors first), but never with one of these.
IMPORT_ERRORS = ("ModuleNotFoundError", "ImportError", "SyntaxError", "NameError",
                 "AttributeError")


@pytest.mark.parametrize("tool", TOOLS, ids=lambda p: p.stem)
def test_the_tool_imports_when_run_as_a_script(tool: Path):
    r = subprocess.run([sys.executable, str(tool), "--help"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    combined = r.stdout + r.stderr
    bad = [e for e in IMPORT_ERRORS if e in combined]
    assert not bad, f"{tool.name} fails to import as a script: {bad}\n{combined[-600:]}"


def test_every_tool_puts_the_repo_root_on_sys_path():
    """The structural version of the check above, which `--help` alone can miss.

    A `from src.metaeval... import ...` inside a *function* is invisible to an import-time
    check of the module: `--help` returns 0 and the tool still dies on the first real call. So
    every tool is required to do the insert at module level, whether or not its imports are
    lazy today -- which also means adding a lazy import later cannot break it.
    """
    offenders = [p.name for p in TOOLS if "sys.path.insert" not in p.read_text()]
    assert not offenders, (
        f"{offenders} never put the repo root on sys.path, so a `from src.metaeval` import "
        f"raises ModuleNotFoundError when run as `python tools/<name>.py`")


def test_the_six_shipped_tools_are_all_here():
    """The parametrisation walks `tools/`, so it would pass on an empty directory. This repo
    ships exactly two pipelines -- generation, then the evaluator sweep -- plus three read-only
    reporters, and this names their entry points, so a tool dropped by accident fails rather
    than silently reduces coverage."""
    assert {p.stem for p in TOOLS} == {
        "run_pipeline",        # generation: trajectories -> pairs -> filtered dataset
        "build_eval_dataset",  # the blind/answers split the sweep reads
        "run_evaluators",      # the sweep itself
        "report",              # derived statistics from saved results, no calls
        "report_human",        # the human validation subset, no calls
        "score_one",           # one evaluator on one pair, for debugging a row
    }
