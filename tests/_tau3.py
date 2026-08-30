"""`needs_tau3` -- skip a test that cannot run without τ³ installed.

The generation pipeline runs on upstream tau2-bench plus patches/tau3.patch (`scripts/setup_tau3.sh`), which is a
separate ~1GB checkout and is **not** a dependency of the evaluator sweep. A reviewer who only
wants to reproduce the evaluator numbers should be able to run `pytest` and see green.

So the tests that touch tau3's agent registry are skipped rather than failed when τ³ is
absent. Two things this deliberately does not do:

- **It does not skip on a broken tau3.** Only a *missing* module skips; an `ImportError` from
  inside an installed tau3 still fails, because that is a real breakage of the pipeline.
- **It does not guard the whole file.** The steering and pipeline suites are mostly pure --
  criterion text, gap assignment, resume bookkeeping, config validation -- and those keep
  running. Only the registration path is τ³-dependent.
"""

from __future__ import annotations

import importlib.util

import pytest

TAU3_INSTALLED = importlib.util.find_spec("tau2") is not None

needs_tau3 = pytest.mark.skipif(
    not TAU3_INSTALLED,
    reason="τ³ is not installed; run `bash scripts/setup_tau3.sh` first")
