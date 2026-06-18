"""neat3p benchmark environments. Importing this package registers them with Gymnasium."""

from gymnasium.envs.registration import register

from .voxel_forage import VoxelForageEnv

register(
    id="VoxelForage-v0",
    entry_point="neat3p.gym_envs.voxel_forage:VoxelForageEnv",
    max_episode_steps=None,  # the env handles truncation via its own max_steps
    kwargs={"scent": True},
)

# Sparse variant: food is only visible inside the perception patch (no scent gradient).
register(
    id="VoxelForage-NoScent-v0",
    entry_point="neat3p.gym_envs.voxel_forage:VoxelForageEnv",
    max_episode_steps=None,
    kwargs={"scent": False},
)

# Shaped-reward variants: dense potential-based shaping (reward += 0.1 * scent[pos]) so even
# non-eating genomes get a gradient toward food. The eat reward stays dominant. v1 leaves the
# sparse v0 untouched.
_SHAPING = 0.1
register(
    id="VoxelForage-Shaped-v0",
    entry_point="neat3p.gym_envs.voxel_forage:VoxelForageEnv",
    max_episode_steps=None,
    kwargs={"scent": True, "reward_shaping": _SHAPING},
)
register(
    id="VoxelForage-NoScent-Shaped-v0",
    entry_point="neat3p.gym_envs.voxel_forage:VoxelForageEnv",
    max_episode_steps=None,
    kwargs={"scent": False, "reward_shaping": _SHAPING},
)

__all__ = ["VoxelForageEnv"]
