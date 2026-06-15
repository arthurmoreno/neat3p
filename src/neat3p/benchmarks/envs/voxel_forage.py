"""
VoxelForage — the central neat3p gate environment.

A small, fast, deterministic 3D voxel foraging task whose observation/action contract
mirrors the life-simulation game in miniature:

    (C, D, H, W) egocentric voxel patch  +  scalar player attributes  ->  discrete action

so every NEAT network (RecurrentNet, FeatureAttention, HyperNEAT, Adaptive, …) is scored
on the *same* fixed-shape problem and the numbers are directly comparable.

Task
----
A 3D grid holds food (energy ↑), hazards (energy ↓) and walls. The agent carries an
``energy`` scalar that decays each step; eating food restores it, energy ≤ 0 is death.
A diffused **scent** field (one of the observation channels) points toward food, so the
optimal policy is reactive — climb the scent gradient, avoid hazards — which converges
in minutes. Reward = energy gained from food (doing nothing scores 0; no reward-hacking).

Observation (flattened to 1-D, fixed length)
    voxel patch : (C, D, H, W) egocentric window, default C=3 channels
                  ch0 = obstacle, ch2 = hazard, and ch1 depends on ``scent``:
                    scent=True  -> diffused scent field (food detectable from a distance)
                    scent=False -> raw food presence (food visible only inside the patch)
    scalars     : [energy_norm, z_norm]
Action : Discrete(7) = {idle, +x, -x, +y, -y, +z (up), -z (down)}

The ``scent`` flag is the headline experiment knob: scent=True is a *shaped* task a
reactive net can climb; scent=False is a *sparse* task that needs exploration / memory.
Both share the same observation shape, so scores are directly comparable.
"""

from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Action deltas, index = action id.
_ACTION_DELTAS = [
    (0, 0, 0),   # 0 idle
    (1, 0, 0),   # 1 +x
    (-1, 0, 0),  # 2 -x
    (0, 1, 0),   # 3 +y
    (0, -1, 0),  # 4 -y
    (0, 0, 1),   # 5 +z (up)
    (0, 0, -1),  # 6 -z (down)
]


class VoxelForageEnv(gym.Env):
    """3D voxel foraging gate. See module docstring for the contract."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        size=(8, 8, 3),          # grid (W, H, D)
        n_food: int = 6,
        n_hazard: int = 3,
        n_wall: int = 6,
        xy_radius: int = 1,
        z_radius: int = 1,
        energy_start: float = 60.0,
        energy_decay: float = 1.0,
        food_value: float = 20.0,
        hazard_damage: float = 15.0,
        scent_scale: float = 2.5,
        max_steps: int = 160,
        scent: bool = True,
    ):
        super().__init__()
        self.use_scent = scent
        self.W, self.H, self.D = size
        self.n_food = n_food
        self.n_hazard = n_hazard
        self.n_wall = n_wall
        self.xr, self.yr, self.zr = xy_radius, xy_radius, z_radius
        self.energy_start = energy_start
        self.energy_decay = energy_decay
        self.food_value = food_value
        self.hazard_damage = hazard_damage
        self.scent_scale = scent_scale
        self.max_steps = max_steps

        self.n_channels = 3  # obstacle, scent, hazard
        self.patch_shape = (self.n_channels, 2 * self.zr + 1, 2 * self.yr + 1, 2 * self.xr + 1)
        self.n_scalars = 2  # energy_norm, z_norm
        self.obs_dim = int(np.prod(self.patch_shape)) + self.n_scalars

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(_ACTION_DELTAS))

        self.total_food_energy = self.n_food * self.food_value
        self.render_mode = render_mode
        self._screen = None
        self._clock = None

    # ── world generation ────────────────────────────────────────────────────
    def _rand_empty_cell(self):
        while True:
            x = int(self.np_random.integers(self.W))
            y = int(self.np_random.integers(self.H))
            z = int(self.np_random.integers(self.D))
            if not (self.wall[x, y, z] or self.food[x, y, z] or self.hazard[x, y, z]) and (x, y, z) != tuple(self.pos):
                return x, y, z

    def _recompute_scent(self):
        """Diffused scent field: sum of exp(-dist/scale) over remaining food, normalised to [0,1]."""
        self.scent = np.zeros((self.W, self.H, self.D), dtype=np.float32)
        food_cells = np.argwhere(self.food)
        if len(food_cells) == 0:
            return
        xs, ys, zs = np.meshgrid(
            np.arange(self.W), np.arange(self.H), np.arange(self.D), indexing="ij"
        )
        for fx, fy, fz in food_cells:
            dist = np.sqrt((xs - fx) ** 2 + (ys - fy) ** 2 + (zs - fz) ** 2)
            self.scent += np.exp(-dist / self.scent_scale)
        m = float(self.scent.max())
        if m > 0:
            self.scent /= m

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.wall = np.zeros((self.W, self.H, self.D), dtype=bool)
        self.food = np.zeros((self.W, self.H, self.D), dtype=bool)
        self.hazard = np.zeros((self.W, self.H, self.D), dtype=bool)
        self.pos = np.array([self.W // 2, self.H // 2, self.D // 2], dtype=int)

        for _ in range(self.n_wall):
            x, y, z = self._rand_empty_cell()
            self.wall[x, y, z] = True
        for _ in range(self.n_food):
            x, y, z = self._rand_empty_cell()
            self.food[x, y, z] = True
        for _ in range(self.n_hazard):
            x, y, z = self._rand_empty_cell()
            self.hazard[x, y, z] = True

        self.energy = float(self.energy_start)
        self.steps = 0
        self.collected = 0.0
        self._recompute_scent()

        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    # ── observation ─────────────────────────────────────────────────────────
    def _obs(self):
        patch = np.zeros(self.patch_shape, dtype=np.float32)
        ax, ay, az = self.pos
        for k, dz in enumerate(range(-self.zr, self.zr + 1)):
            for j, dy in enumerate(range(-self.yr, self.yr + 1)):
                for i, dx in enumerate(range(-self.xr, self.xr + 1)):
                    vx, vy, vz = ax + dx, ay + dy, az + dz
                    if not (0 <= vx < self.W and 0 <= vy < self.H and 0 <= vz < self.D):
                        patch[0, k, j, i] = 1.0  # out of bounds = obstacle
                        continue
                    if self.wall[vx, vy, vz]:
                        patch[0, k, j, i] = 1.0
                    # ch1: scent gradient (visible from afar) or raw food presence (local only)
                    if self.use_scent:
                        patch[1, k, j, i] = self.scent[vx, vy, vz]
                    elif self.food[vx, vy, vz]:
                        patch[1, k, j, i] = 1.0
                    if self.hazard[vx, vy, vz]:
                        patch[2, k, j, i] = 1.0
        scalars = np.array(
            [np.clip(self.energy / self.energy_start, 0.0, 1.0), az / max(1, self.D - 1)],
            dtype=np.float32,
        )
        return np.concatenate([patch.reshape(-1), scalars]).astype(np.float32)

    def _info(self):
        return {"energy": self.energy, "collected": self.collected, "steps": self.steps}

    # ── dynamics ────────────────────────────────────────────────────────────
    def step(self, action):
        dx, dy, dz = _ACTION_DELTAS[int(action)]
        nx, ny, nz = self.pos[0] + dx, self.pos[1] + dy, self.pos[2] + dz

        moved = (
            0 <= nx < self.W and 0 <= ny < self.H and 0 <= nz < self.D and not self.wall[nx, ny, nz]
        )
        if moved:
            self.pos = np.array([nx, ny, nz], dtype=int)

        reward = 0.0
        x, y, z = self.pos
        if self.food[x, y, z]:
            self.food[x, y, z] = False
            self.energy += self.food_value
            self.collected += self.food_value
            reward = self.food_value
            self._recompute_scent()
        if self.hazard[x, y, z]:
            self.energy -= self.hazard_damage

        self.energy -= self.energy_decay
        self.steps += 1

        terminated = self.energy <= 0.0 or not self.food.any()
        truncated = self.steps >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info()

    # ── rendering (simple pygame top-down of the agent's z-plane) ────────────
    def render(self):
        if self.render_mode is None:
            return None
        import pygame

        cell = 48
        margin_top = 40
        w_px, h_px = self.W * cell, self.H * cell + margin_top
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((w_px, h_px))
                pygame.display.set_caption("VoxelForage")
            else:
                self._screen = pygame.Surface((w_px, h_px))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 18)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        surf = self._screen
        surf.fill((18, 18, 22))
        az = int(self.pos[2])
        for x in range(self.W):
            for y in range(self.H):
                rect = pygame.Rect(x * cell, margin_top + y * cell, cell - 2, cell - 2)
                scent = float(self.scent[x, y, az]) if self.use_scent else 0.0
                color = (int(25 + 90 * scent), int(25 + 50 * scent), int(35 + 30 * scent))
                if self.wall[x, y, az]:
                    color = (90, 90, 100)
                elif self.food[x, y, az]:
                    color = (60, 200, 90)
                elif self.hazard[x, y, az]:
                    color = (210, 70, 60)
                pygame.draw.rect(surf, color, rect)
        ax, ay = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(
            surf, (70, 140, 240),
            (ax * cell + cell // 2 - 1, margin_top + ay * cell + cell // 2 - 1), cell // 3
        )
        hud = f"z={az} energy={self.energy:5.1f} food={self.collected:5.0f}/{self.total_food_energy:.0f} step={self.steps}"
        surf.blit(self._font, (6, 10)) if False else surf.blit(self._font.render(hud, True, (230, 230, 230)), (6, 10))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        import numpy as _np

        return _np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._screen = None
