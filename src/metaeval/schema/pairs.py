"""Turn a group of three steered trajectories into the three pairs it yields.

The group-of-three design: one (domain, task, criterion), the same agent, user
simulator and seed, run three times under a bad / ok / good steering
instruction. The three pairwise comparisons that fall out are (bad, ok),
(ok, good) and (bad, good) -- two adjacent gaps and one wide one, so gap size
varies within the dataset rather than across datasets.

The three pairs of a group **share trajectories**, so they are not independent
observations. Cluster by `group_id` in any statistics.
"""

from __future__ import annotations

import random
import uuid

from .models import LEVEL_ORDER, Level, PairwiseEntry, Trajectory
from .render import render_entry

# Every unordered pair of distinct levels. Two adjacent, one wide.
PAIR_LEVELS: list[tuple[Level, Level]] = [("bad", "ok"), ("ok", "good"), ("bad", "good")]


def pairs_from_group(
    trajectories: dict[str, Trajectory],
    *,
    criterion_name: str,
    criterion_description: str,
    group_id: str | None = None,
    seed: int | None = None,
    render: bool = True,
    pair_levels: list[tuple[Level, Level]] | None = None,
    **render_kwargs,
) -> list[PairwiseEntry]:
    """Build the three pairs from one group.

    `trajectories` maps level -> trajectory. By default all three levels are required
    and all three pairs are built; a missing level is an error rather than a silently
    smaller group, because a dataset quietly short on wide gaps would skew the
    accuracy-versus-gap result the design exists to measure.

    `pair_levels` narrows that to specific pairs, which is how the **one gap per cell**
    design works: the gap is assigned at generation time (`steering/gaps.py`), so only the
    two levels it needs are ever simulated -- two simulations per cell instead of three, a
    3:2 saving on the stage that dominates both cost and wall-clock. Only the levels those
    pairs reference are then required.

    Which trajectory becomes `response_1` is randomised per pair. Without that, the better
    side would always sit in the same slot and an evaluator could score above chance by
    position alone.

    A group whose sides do not share one task context is **refused**, not built. The pair
    records a single `context`, so building it would mean storing one side's task and
    policy as if both had run under it -- and every downstream number would then be
    computed over a comparison that is not between two performances of one task.
    """
    wanted = pair_levels if pair_levels is not None else PAIR_LEVELS
    need = {lvl for pair in wanted for lvl in pair}
    missing = [lvl for lvl in LEVEL_ORDER if lvl in need and lvl not in trajectories]
    if missing:
        raise ValueError(
            f"group is incomplete: missing level(s) {missing}. "
            f"Have {sorted(trajectories)}."
        )

    keys = {lvl: trajectories[lvl].context.context_key() for lvl in sorted(need)}
    if len(set(keys.values())) > 1:
        raise ValueError(
            "group was not held fixed: the levels disagree on task context, so there is "
            f"no single task being compared. Context keys by level: {keys}."
        )

    gid = group_id or f"group-{uuid.uuid4()}"
    entries: list[PairwiseEntry] = []

    for worse_level, better_level in wanted:
        worse, better = trajectories[worse_level], trajectories[better_level]
        entry_id = str(uuid.uuid4())
        # Seeded off the entry id so the layout is reproducible from the stored
        # record alone, exactly as the steering generator does it.
        rng = random.Random(f"{seed}:{entry_id}" if seed is not None else entry_id)
        better_first = rng.random() > 0.5

        if better_first:
            t1, t2 = better, worse
            levels = (better_level, worse_level)
            correct = 1
        else:
            t1, t2 = worse, better
            levels = (worse_level, better_level)
            correct = 2

        entry = PairwiseEntry(
            id=entry_id,
            criterion_name=criterion_name,
            criterion_description=criterion_description,
            positive_hint=_instruction(better),
            negative_hint=_instruction(worse),
            correct_response=correct,
            group_id=gid,
            levels=levels,
            # Checked above, so by this line both sides genuinely share it.
            context=t1.context,
            trajectory_1=t1,
            trajectory_2=t2,
        )
        if render:
            render_entry(entry, **render_kwargs)
        entries.append(entry)

    return entries


def _instruction(traj: Trajectory) -> str:
    steering = traj.provenance.steering
    return steering.instruction if steering is not None else ""
