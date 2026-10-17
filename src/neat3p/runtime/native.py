"""Torch-free brain net with the same contract as ``neat3p.nn.composite.NEATRecurrentNet``."""

from __future__ import annotations

import numpy as np

from neat3p.runtime.compiled import CompiledRecurrentNet, compile_recurrent
from neat3p.runtime.layout import build_recurrent_layout


def _as_row(state, state_dim: int) -> np.ndarray:
    if hasattr(state, "detach"):  # CPU torch tensor, without importing torch
        state = state.detach().cpu().numpy()
    arr = np.asarray(state)
    if arr.ndim == 2 and arr.shape[0] != 1:
        raise ValueError(f"NativeRecurrentNet runs one creature at a time, got batch of {arr.shape[0]}")
    arr = arr.reshape(-1)
    if arr.shape[0] != state_dim:
        raise ValueError(f"expected state of length {state_dim}, got {arr.shape[0]}")
    return arr


class NativeRecurrentNet:
    """A genome compiled into the native evaluator, behind forward / get_action / reset.

    Only batch size 1 is supported. ``seed`` seeds the epsilon-greedy exploration; with
    ``explore_rate=0`` the net is deterministic.
    """

    def __init__(
        self,
        genome,
        state_dim: int,
        action_dim: int,
        config,
        *,
        use_current_activs: bool = True,
        explore_rate: float = 0.125,
        seed: int | None = None,
        dtype: str = "float64",
        device_alias: str | None = None,  # accepted and ignored: callers written for torch pass a device
    ):
        self.genome = genome
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_rate = explore_rate
        self.layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=use_current_activs)
        if self.layout.n_inputs != state_dim:
            raise ValueError(f"genome expects {self.layout.n_inputs} inputs, state_dim is {state_dim}")
        if self.layout.n_outputs != action_dim:
            raise ValueError(f"genome has {self.layout.n_outputs} outputs, action_dim is {action_dim}")
        self.net: CompiledRecurrentNet = compile_recurrent(self.layout, dtype=dtype)
        self._rng = np.random.default_rng(seed)

    def forward(self, state) -> np.ndarray:
        """(1, state_dim) or (state_dim,) -> (1, action_dim) float64."""
        return self.net.activate(_as_row(state, self.state_dim))[None, :]

    __call__ = forward

    def reset(self, batch_size: int = 1) -> None:
        if batch_size != 1:
            raise ValueError("NativeRecurrentNet supports batch_size=1 only")
        self.net.reset()

    def get_action(self, state) -> np.ndarray:
        """Epsilon-greedy action index, shape (1, 1) int64; one random draw per call."""
        if self._rng.random() < self.explore_rate:
            return np.array([[self._rng.integers(self.action_dim)]], dtype=np.int64)
        return np.argmax(self.forward(state), axis=1, keepdims=True).astype(np.int64)
