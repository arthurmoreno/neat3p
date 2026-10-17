"""Torch-free layout of a recurrent NEAT phenotype.

A line-for-line copy of the index logic in ``neat3p.nn.phenotypes.recurrent_net.RecurrentNet.create``,
quirks included, so the native evaluator reproduces the torch phenotype exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from neat3p.graphs import required_for_output

BlockName = Literal[
    "input_to_hidden",
    "hidden_to_hidden",
    "output_to_hidden",
    "input_to_output",
    "hidden_to_output",
    "output_to_output",
]

BLOCK_NAMES: tuple[BlockName, ...] = (
    "input_to_hidden",
    "hidden_to_hidden",
    "output_to_hidden",
    "input_to_output",
    "hidden_to_output",
    "output_to_output",
)


@dataclass(frozen=True)
class CooBlock:
    """Sparse weights of one block: ``matrix[rows[i], cols[i]] = weights[i]``."""

    rows: np.ndarray  # int64, target index
    cols: np.ndarray  # int64, source index
    weights: np.ndarray  # float64


@dataclass(frozen=True)
class RecurrentLayout:
    n_inputs: int
    n_hidden: int
    n_outputs: int
    input_keys: tuple[int, ...]
    hidden_keys: tuple[int, ...]
    output_keys: tuple[int, ...]
    blocks: Mapping[str, CooBlock]
    hidden_biases: np.ndarray
    hidden_responses: np.ndarray
    output_biases: np.ndarray
    output_responses: np.ndarray
    use_current_activs: bool
    n_internal_steps: int

    def block_shape(self, name: str) -> tuple[int, int]:
        src, dst = name.split("_to_")
        sizes = {"input": self.n_inputs, "hidden": self.n_hidden, "output": self.n_outputs}
        return sizes[dst], sizes[src]

    def to_dense(self) -> dict[str, np.ndarray]:
        """Dense float64 matrices with the shapes torch's ``RecurrentNet`` builds."""
        dense: dict[str, np.ndarray] = {}
        for name in BLOCK_NAMES:
            if self.n_hidden == 0 and "hidden" in name:
                continue
            mat = np.zeros(self.block_shape(name), dtype=np.float64)
            blk = self.blocks[name]
            if len(blk.rows):
                mat[blk.rows, blk.cols] = blk.weights
            dense[name] = mat
        return dense


def build_recurrent_layout(
    genome,
    genome_config,
    *,
    use_current_activs: bool,
    prune_empty: bool = False,
    n_internal_steps: int = 1,
) -> RecurrentLayout:
    """Build the layout of ``genome``; ``genome_config`` is ``config.genome_config``."""
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

    nonempty: set[int] = set()
    if prune_empty:
        nonempty = {conn.key[1] for conn in genome.connections.values() if conn.enabled}.union(
            set(genome_config.input_keys)
        )

    input_keys = list(genome_config.input_keys)
    output_keys = list(genome_config.output_keys)
    output_set = set(output_keys)
    # Every non-output node is a hidden node, required or not.
    hidden_keys = [k for k in genome.nodes.keys() if k not in output_set]

    hidden_responses = np.array([genome.nodes[k].response for k in hidden_keys], dtype=np.float64)
    output_responses = np.array([genome.nodes[k].response for k in output_keys], dtype=np.float64)
    hidden_biases = np.array([genome.nodes[k].bias for k in hidden_keys], dtype=np.float64)
    output_biases = np.array([genome.nodes[k].bias for k in output_keys], dtype=np.float64)

    if prune_empty:
        for i, key in enumerate(output_keys):
            if key not in nonempty:
                output_biases[i] = 0.0

    input_idx = {k: i for i, k in enumerate(input_keys)}
    hidden_idx = {k: i for i, k in enumerate(hidden_keys)}
    output_idx = {k: i for i, k in enumerate(output_keys)}

    coo: dict[str, tuple[list[int], list[int], list[float]]] = {name: ([], [], []) for name in BLOCK_NAMES}

    for conn in genome.connections.values():
        if not conn.enabled:
            continue

        i_key, o_key = conn.key
        if o_key not in required and i_key not in required:
            continue
        if prune_empty and i_key not in nonempty:
            continue

        if i_key in input_idx and o_key in hidden_idx:
            name, src, dst = "input_to_hidden", input_idx[i_key], hidden_idx[o_key]
        elif i_key in hidden_idx and o_key in hidden_idx:
            name, src, dst = "hidden_to_hidden", hidden_idx[i_key], hidden_idx[o_key]
        elif i_key in output_idx and o_key in hidden_idx:
            name, src, dst = "output_to_hidden", output_idx[i_key], hidden_idx[o_key]
        elif i_key in input_idx and o_key in output_idx:
            name, src, dst = "input_to_output", input_idx[i_key], output_idx[o_key]
        elif i_key in hidden_idx and o_key in output_idx:
            name, src, dst = "hidden_to_output", hidden_idx[i_key], output_idx[o_key]
        elif i_key in output_idx and o_key in output_idx:
            name, src, dst = "output_to_output", output_idx[i_key], output_idx[o_key]
        else:
            raise ValueError("Invalid connection from key {} to key {}".format(i_key, o_key))

        rows, cols, weights = coo[name]
        rows.append(dst)
        cols.append(src)
        weights.append(conn.weight)

    blocks = {
        name: CooBlock(
            rows=np.array(rows, dtype=np.int64),
            cols=np.array(cols, dtype=np.int64),
            weights=np.array(weights, dtype=np.float64),
        )
        for name, (rows, cols, weights) in coo.items()
    }

    return RecurrentLayout(
        n_inputs=len(input_keys),
        n_hidden=len(hidden_keys),
        n_outputs=len(output_keys),
        input_keys=tuple(input_keys),
        hidden_keys=tuple(hidden_keys),
        output_keys=tuple(output_keys),
        blocks=MappingProxyType(blocks),
        hidden_biases=hidden_biases,
        hidden_responses=hidden_responses,
        output_biases=output_biases,
        output_responses=output_responses,
        use_current_activs=use_current_activs,
        n_internal_steps=n_internal_steps,
    )
