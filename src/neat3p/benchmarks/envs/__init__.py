"""neat3p benchmark environments. Importing this package registers them with Gymnasium."""

from gymnasium.envs.registration import register

from .voxel_forage import VoxelForageEnv

register(
    id="VoxelForage-v0",
    entry_point="neat3p.benchmarks.envs.voxel_forage:VoxelForageEnv",
    max_episode_steps=None,  # the env handles truncation via its own max_steps
    kwargs={"scent": True},
)

# Sparse variant: food is only visible inside the perception patch (no scent gradient).
register(
    id="VoxelForage-NoScent-v0",
    entry_point="neat3p.benchmarks.envs.voxel_forage:VoxelForageEnv",
    max_episode_steps=None,
    kwargs={"scent": False},
)

__all__ = ["VoxelForageEnv"]
