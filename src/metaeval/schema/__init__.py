"""Canonical schema for trajectory-based preference pairs.

Import from here, not from the submodules. `convert` is the exception: it pulls
in tau2, so it is imported lazily by the helpers below rather than at package
import time -- reading a stored dataset must not require tau3 to be installed.

    from src.metaeval.schema import PairwiseEntry, load_pairs, save_pairs
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    LEVEL_ORDER,
    SCHEMA_VERSION,
    JudgeVote,
    KnowledgeBaseInfo,
    Level,
    Outcome,
    PairwiseEntry,
    Provenance,
    RetrievedDoc,
    RewardRecord,
    SteeringRecord,
    TaskContext,
    ToolCallRecord,
    ToolResult,
    ToolSpec,
    Trajectory,
    TrajectoryStats,
    Turn,
)
from .pairs import PAIR_LEVELS, pairs_from_group
from .render import (
    render_context,
    render_entry,
    render_task_metadata,
    render_trajectory,
)

__all__ = [
    "LEVEL_ORDER",
    "PAIR_LEVELS",
    "SCHEMA_VERSION",
    "JudgeVote",
    "KnowledgeBaseInfo",
    "Level",
    "Outcome",
    "PairwiseEntry",
    "Provenance",
    "RetrievedDoc",
    "RewardRecord",
    "SteeringRecord",
    "TaskContext",
    "ToolCallRecord",
    "ToolResult",
    "ToolSpec",
    "Trajectory",
    "TrajectoryStats",
    "Turn",
    "load_pairs",
    "load_trajectory",
    "pairs_from_group",
    "render_context",
    "render_entry",
    "render_task_metadata",
    "render_trajectory",
    "save_pairs",
    "save_trajectory",
]


def save_pairs(entries: list[PairwiseEntry], path: str | Path) -> None:
    """Write a dataset.

    A plain JSON list of objects, each leading with the flat pair fields, so a consumer
    that only wants `prompt`/`response_1`/`response_2`/`correct_response` can read the file
    with `json.load` and never import this schema.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([e.model_dump(mode="json") for e in entries], indent=2))


def load_pairs(path: str | Path) -> list[PairwiseEntry]:
    return [PairwiseEntry.model_validate(d) for d in json.loads(Path(path).read_text())]


def save_trajectory(traj: Trajectory, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(traj.model_dump_json(indent=2))


def load_trajectory(path: str | Path) -> Trajectory:
    return Trajectory.model_validate_json(Path(path).read_text())
