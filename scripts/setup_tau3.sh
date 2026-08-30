#!/usr/bin/env bash
#
# Install the tau3-bench substrate into this repo's venv.
#
# The substrate is upstream tau2-bench at a pinned commit, plus `patches/tau3.patch`.
# The patch is small (9 files, ~125 lines) and does two things that matter:
#
#   1. Disables tau2's own LLM-based judges by default. They grade task success with a
#      model call, which this work does not use and which would need an extra provider
#      key. Disabling is deliberately NOT "return no findings": an empty NL-assertion
#      list would score reward 1.0, so the evaluator drops NL_ASSERTION from the reward
#      basis and warns instead.
#   2. Relaxes the upper bound on `litellm`, which excluded two yanked releases and
#      every later one with them.
#
# It also adds a `TAU2_LLM_OVERRIDE` test hook and restores a dropped latency field on
# user-simulator messages. It changes nothing about the domains, tasks, policies, tools
# or the agent loop -- the substrate is upstream's, unchanged.
#
# Usage:
#   scripts/setup_tau3.sh [path-to-checkout]   # default ../tau2-bench
#   scripts/setup_tau3.sh --check              # verify only, install nothing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
PATCH="$REPO_ROOT/patches/tau3.patch"
CHECK_ONLY=0
TAU3_DIR=""

# Upstream tau2-bench, and the commit every published number was produced on top of.
# Applying patches/tau3.patch to this commit reproduces the substrate exactly.
UPSTREAM=https://github.com/sierra-research/tau2-bench.git
PINNED_SHA=668d3bcd135c02aa3438f987ef45735b7c163ee3

for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    *)       TAU3_DIR="$arg" ;;
  esac
done
TAU3_DIR="${TAU3_DIR:-$REPO_ROOT/../tau2-bench}"

[[ -x "$VENV_PY" ]] || { echo "error: no venv at $REPO_ROOT/.venv" >&2; exit 1; }

verify() {
  echo "--- verifying"
  "$VENV_PY" - <<'PY'
import contextlib
import importlib.metadata as md
import io
import sys

lit = md.version("litellm")
print(f"  litellm            {lit}")
# 1.82.7 and 1.82.8 were yanked from PyPI; anything in that window means the
# install pulled us into it.
if (1, 82, 7) <= tuple(int(x) for x in lit.split(".")[:3]) <= (1, 82, 8):
    print("  FAIL: litellm is in the yanked 1.82.7/1.82.8 window", file=sys.stderr)
    sys.exit(1)

try:
    print(f"  tau2               {md.version('tau2')}")
except md.PackageNotFoundError:
    print("  FAIL: tau2 not installed", file=sys.stderr)
    sys.exit(1)

buf = io.StringIO()
with contextlib.redirect_stderr(buf):          # the registry logs on import
    try:
        from tau2.config import DISABLE_LLM_JUDGES, LLM_OVERRIDE
    except ImportError:
        print(
            "  FAIL: tau2 is installed but patches/tau3.patch was never applied\n"
            "        (tau2.config has no DISABLE_LLM_JUDGES). Re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    from tau2.registry import registry

doms = sorted(registry.get_domains())
print(f"  domains            {len(doms)}: {', '.join(doms)}")
assert "banking_knowledge" in doms, "banking_knowledge missing (rank-bm25 not installed?)"
print(f"  LLM judges         {'DISABLED' if DISABLE_LLM_JUDGES else 'enabled'}")
print(f"  TAU2_LLM_OVERRIDE  {LLM_OVERRIDE or '(unset)'}")
if not DISABLE_LLM_JUDGES:
    print("  WARN: LLM judges are enabled; tau2's own judges will run and will "
          "need provider keys", file=sys.stderr)
print("  OK")
PY
}

# Is the patch already in the tree? Checked by content, not by a marker file, so a
# half-applied or hand-edited checkout is not mistaken for a clean one.
patch_applied() {
  git -C "$TAU3_DIR" apply --reverse --check "$PATCH" 2>/dev/null
}

apply_patch() {
  if patch_applied; then
    echo "--- patch already applied"
    return
  fi
  echo "--- applying patches/tau3.patch"
  if ! git -C "$TAU3_DIR" apply --check "$PATCH" 2>/dev/null; then
    cat >&2 <<EOS
error: patches/tau3.patch does not apply to this checkout, and is not already applied.

  Almost always this means the checkout is not at the pinned commit. Reset it:

    git -C "$TAU3_DIR" fetch origin $PINNED_SHA
    git -C "$TAU3_DIR" checkout --force $PINNED_SHA

  If you have local edits you want to keep, apply the patch by hand -- it is 9 files.
EOS
    exit 1
  fi
  git -C "$TAU3_DIR" apply "$PATCH"
}

check_pin() {
  local head
  head="$(git -C "$TAU3_DIR" rev-parse HEAD)"
  echo "--- tau3 checkout: $TAU3_DIR"
  echo "    ${head:0:7} (pinned ${PINNED_SHA:0:7})"
  if [[ "$head" != "$PINNED_SHA" ]]; then
    cat >&2 <<EOS

  ****************************************************************************
  WARNING: this checkout is NOT the pinned commit.

    expected  ${PINNED_SHA:0:7}
    found     ${head:0:7}

  Every published number was produced on the pinned commit. A different commit may
  change tasks, policies or the agent loop, and nothing downstream will say so --
  the pipeline will run and report accuracies over a different substrate.

    git -C "$TAU3_DIR" checkout $PINNED_SHA

  ****************************************************************************

EOS
  fi
}

if [[ $CHECK_ONLY -eq 1 ]]; then
  # `&&` here would take the whole line non-zero under `set -e` when the checkout is
  # absent, and --check would exit without ever running verify.
  if [[ -d "$TAU3_DIR/.git" ]]; then
    check_pin
    if patch_applied; then echo "    patch applied"; else echo "    patch NOT applied"; fi
  fi
  verify
  exit 0
fi

[[ -d "$TAU3_DIR" ]] || {
  cat >&2 <<EOS
error: no tau3 checkout at $TAU3_DIR

  git clone $UPSTREAM "$TAU3_DIR"
  git -C "$TAU3_DIR" checkout $PINNED_SHA
  bash scripts/setup_tau3.sh
EOS
  exit 1
}

check_pin
apply_patch

# The patched pin permits this repo's litellm, so let pip resolve normally. The
# knowledge extra pulls rank-bm25, needed for the banking_knowledge domain.
echo "--- installing tau3 (editable, with knowledge extra)"
"$VENV_PY" -m pip install -q -e "${TAU3_DIR}[knowledge]"

verify

cat <<'EOS'

--- next
  python tools/run_pipeline.py --config config/smoke.yaml --dry-run   # no calls, prints the plan
  python tools/run_pipeline.py --config config/smoke.yaml             # 3 cells, minutes, cents
EOS
