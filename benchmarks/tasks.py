"""
Task registry — one TaskSpec per benchmark environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

_CONFIGS = os.path.join(os.path.dirname(__file__), "configs")


def _config(name: str) -> str:
    return os.path.join(_CONFIGS, name)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_id: str
    variants: dict
    solve_threshold: float
    reward_label: str = "reward"
    reward_max: float | None = None
    register: Callable[[], None] = field(default=lambda: None, compare=False, hash=False)
    substrate: Callable[[], tuple] | None = field(default=None, compare=False, hash=False)


def _register_voxel_forage() -> None:
    import neat3p.benchmarks.envs  # noqa: F401


def _voxel_forage_substrate():
    from neat3p.benchmarks.substrates import voxel_forage_substrate

    return voxel_forage_substrate()


CARTPOLE = TaskSpec(
    name="cartpole",
    env_id="CartPole-v1",
    variants={},
    solve_threshold=475.0,
    reward_label="reward",
    reward_max=500.0,
)

LUNARLANDER = TaskSpec(
    name="lunarlander",
    env_id="LunarLander-v3",
    variants={},
    solve_threshold=200.0,
    reward_label="reward",
)

VOXEL_FORAGE = TaskSpec(
    name="voxel_forage",
    env_id="VoxelForage-v0",
    variants={
        "scent": "VoxelForage-v0",
        "noscent": "VoxelForage-NoScent-v0",
        "shaped": "VoxelForage-Shaped-v0",
    },
    solve_threshold=110.0,
    reward_label="food",
    reward_max=120.0,
    register=_register_voxel_forage,
    substrate=_voxel_forage_substrate,
)

TASKS: dict[str, TaskSpec] = {t.name: t for t in (CARTPOLE, LUNARLANDER, VOXEL_FORAGE)}
