"""A tau3 agent that appends a steering instruction to its system prompt.

Registered as `steered:<criterion>:<level>` -- one agent name per pair, nine in
total. That keeps the instruction in a closure rather than in module state, so
concurrent simulations on different criteria cannot read each other's steering.
`TextRunConfig.agent` is a plain `str` with no registry validation, so the name
passes straight through.

Everything else about the agent is tau3's `LLMAgent`: same tool handling, same
message loop, same generation path. Only `system_prompt` differs, which is what
makes the steered and unsteered runs comparable.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from .criteria import CRITERIA, LEVELS, get_criterion

AGENT_PREFIX = "steered"

# tau3's registry is global, mutable process state with no removal, and
# `register_agent_factory` raises on a duplicate name. Both registration functions below
# therefore do a check-then-register, which is a race as soon as two threads run
# simulations concurrently: both see the name absent, both register, the loser raises and
# kills its simulation. The window is small and the cost of losing it is one trajectory,
# ~40k tokens -- a two-level group is ~97k, which is the figure `run_steered_resilient` prices
# -- so the pair is made atomic rather than left to chance.
#
# One lock for the module, not one per name: registrations are microseconds of dict work
# with no I/O inside the critical section, so there is nothing to gain from finer grain.
_REGISTRY_LOCK = threading.Lock()


def agent_name(criterion: str, level: str, tag: str = "") -> str:
    """Registry name for a steered agent.

    `tag` distinguishes per-data-point instruction sets from the static
    hand-written baseline. Instructions are generated per group, so the same
    (criterion, level) carries different text in different groups and cannot share
    one registry entry.
    """
    base = f"{AGENT_PREFIX}:{criterion}:{level}"
    return f"{base}:{tag}" if tag else base


def register_instruction(instruction: str, name: str) -> str:
    """Register an agent carrying exactly this instruction. Idempotent by name.

    The instruction lives in the factory's closure, so two groups running
    concurrently on the same criterion cannot read each other's steering. tau3's
    registry has no removal, but a full run registers ~36 names, which is nothing.

    The caller owns uniqueness of `name`: registering a *different* instruction
    under a name already taken would silently run the old one, so that is an error
    rather than an overwrite.

    **Thread-safe, and safe to call again for the same name.** A simulation retry re-runs
    this with the identical instruction and takes the early return, which is what lets
    `run_steered_resilient` re-run a failed simulation without tripping tau3's
    raise-on-duplicate. The lock is what makes the check and the register one step; see
    `_REGISTRY_LOCK`.
    """
    from tau2.registry import registry

    with _REGISTRY_LOCK:
        existing = registry.get_agent_factory(name)
        if existing is not None:
            prior = getattr(existing, "steering_instruction", None)
            if prior is not None and prior != instruction:
                raise ValueError(
                    f"agent name {name!r} is already registered with a different "
                    f"instruction; use a distinct tag rather than reusing the name"
                )
            return name

        factory = _make_instruction_factory(instruction)
        registry.register_agent_factory(factory, name=name)
    return name


def _make_class(instruction: str):
    from tau2.agent.llm_agent import LLMAgent

    class SteeredLLMAgent(LLMAgent):
        """`LLMAgent` with the steering instruction appended to its prompt.

        Appended *after* tau3's `<policy>` block, so the instruction's reference
        to "the <policy> above" resolves. Subclassing rather than reformatting
        `SYSTEM_PROMPT` keeps us on tau3's own prompt whenever it changes.
        """

        steering_instruction = instruction

        @property
        def system_prompt(self) -> str:
            return f"{super().system_prompt}\n\n{self.steering_instruction}"

    return SteeredLLMAgent


def _make_instruction_factory(instruction: str):
    """A tau3 agent factory bound to one exact instruction."""

    def factory(
        tools: Any,
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        # `build_agent` passes these to every factory; irrelevant for a
        # half-duplex text agent. Accepted and ignored, with **kwargs so a new
        # tau3 argument does not break registration.
        task: Any = None,
        audio_native_config: Any = None,
        audio_taps_dir: Any = None,
        **kwargs: Any,
    ):
        cls = _make_class(instruction)
        return cls(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )

    # Recorded on the factory so register_instruction can detect a name reused
    # with different steering.
    factory.steering_instruction = instruction
    return factory


def _make_factory(criterion: str, level: str):
    """Factory for the static hand-written baseline set."""
    factory = _make_instruction_factory(get_criterion(criterion).instruction(level))
    factory.__name__ = f"steered_agent_{criterion}_{level}"
    return factory


def register_steered_agents() -> list[str]:
    """Register one steered agent per (criterion, level) pair that has a baseline
    instruction -- six with the shipped criteria. Idempotent.

    tau3's `register_agent_factory` raises on a duplicate name, so re-registration
    is skipped rather than allowed to fail -- this gets called from a CLI, a test,
    and a notebook, and the second call must not be an error. Under the same lock as
    `register_instruction`, since `run_steered` calls this one on the baseline path and
    that one on the generated path, and a pool can have both in flight at once.
    """
    from tau2.registry import registry

    names = []
    with _REGISTRY_LOCK:
        for criterion, spec in CRITERIA.items():
            # Only the legacy hand-written set can be registered statically. Criteria added
            # after 2026-08-11 carry no baseline; their instructions come from the generator
            # per data point and are registered on the fly by `register_instruction`.
            if not spec.instructions:
                continue
            for level in LEVELS:
                name = agent_name(criterion, level)
                names.append(name)
                if registry.get_agent_factory(name) is not None:
                    continue
                registry.register_agent_factory(
                    _make_factory(criterion, level), name=name
                )
    return names
