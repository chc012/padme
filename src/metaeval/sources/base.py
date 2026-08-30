"""What a substrate has to provide, written down as a type.

`tau2_source` is the only implementation, and this file exists to make the boundary's *width*
checkable rather than merely claimed. What the generator needs from a substrate is three
functions, named below; `tests/test_sources_boundary.py` pins both that list and the (larger,
three-module) list of files allowed to import τ³ at all, so a fourth one is a change someone
has to make deliberately.

**A Protocol rather than an ABC, and no adapter class.** `tau2_source` is a module of
functions, which is the right shape for it: the task catalogue is process-global and cached,
the τ³ agent registry is process-global, and there is no per-instance state for a class to
hold. Wrapping that in a class to satisfy an ABC would add an object whose only job is to
forward three calls -- so the Protocol is structural, `tau2_source` satisfies it as a module,
and `tests/test_sources_boundary.py` asserts so.

**This is a documentation and typing device, not a plugin system.** Nothing dispatches on it,
and adding a second substrate is more than implementing three methods: `run_steered` returns
our `Trajectory`, which means the new substrate also owns the conversion (`schema/convert.py`
does it for τ³), and it has to decide what "the same task twice" means in its own terms. The
Protocol says what the generator needs. It does not say that supplying it is cheap.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from ..schema.models import Trajectory


@runtime_checkable
class Substrate(Protocol):
    """The three things the steering pipeline asks of the environment it runs in.

    Named for what the caller needs, not for how τ³ happens to spell it. `runtime_checkable`
    so a test can assert conformance -- with the usual caveat that it checks *names*, not
    signatures, which is why the test also inspects the signatures.
    """

    def domain_tasks(self, domain: str) -> list:
        """Every task in a domain, on the split this substrate samples from.

        The pipeline needs the count to plan cells (`task_indices`, `limit`) and needs the
        catalogue to be stable across a run: cell N must mean the same task on the retry as it
        did on the first draw, or a re-draw silently compares two different tasks.
        """
        ...

    def task_description(self, task: Any) -> str:
        """The task as a plain string for the instruction generator.

        **The whole substrate-to-generator boundary.** Handed over verbatim: cleaning it means
        writing substrate-specific string surgery, which is exactly the brittleness this
        interface exists to avoid. The generator never passes it to the agent, and it is not
        ground truth -- so scaffolding in it is harmless noise.
        """
        ...

    def run_steered(
        self,
        domain: str,
        task: Any,
        criterion: str,
        level: str,
        agent_key: str,
        *,
        seed: int = 42,
        max_steps: Optional[int] = None,
        instruction: Optional[str] = None,
        instruction_tag: str = "",
    ) -> Trajectory:
        """One conversation, run with `instruction` appended to the agent's system prompt.

        Returns our `Trajectory`, so a substrate owns its own conversion. Two calls that
        differ only in `instruction` are what a pair is, which puts two requirements on any
        implementation:

        - **Everything else must be held fixed** -- task, agent model, user simulator, seed.
          Not to make the two runs identical apart from the steer (they are not; identical
          inputs to τ³ produce 22-52 messages), but to make the steered axis the largest
          *systematic* difference rather than merely one of several.
        - **A failure must raise, not degrade.** A substrate that returns a truncated
          conversation on error hands the pipeline a pair whose worse side is worse for the
          wrong reason, and nothing downstream can tell.
        """
        ...
