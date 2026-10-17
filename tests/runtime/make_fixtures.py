"""Generate the torch-parity fixtures for ``neat3p.runtime`` (needs torch; never run by the test suite).

Usage (from ``neat3p/``, in the ``neat3p`` conda env)::

    conda run --no-capture-output -n neat3p python tests/runtime/make_fixtures.py

Writes ``fixtures/parity_cases.json`` (genomes + metadata) and ``fixtures/parity_arrays.npz``
(inputs, torch outputs, torch dense matrices). Both are committed so the suite runs without torch.
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

import neat3p
from neat3p.genes import DefaultConnectionGene, DefaultNodeGene
from neat3p.genome import DefaultGenome

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
CASES_JSON = os.path.join(FIXTURES, "parity_cases.json")
ARRAYS_NPZ = os.path.join(FIXTURES, "parity_arrays.npz")

CONFIG_PATHS = {
    "lifesim_recurrent": os.path.join(HERE, "configs", "lifesim_recurrent.cfg"),
    "stress": os.path.join(HERE, "configs", "stress.cfg"),
    "edge": os.path.join(HERE, "configs", "edge.cfg"),
    "cartpole": os.path.join(HERE, "..", "..", "benchmarks", "configs", "cartpole.cfg"),
}

TICKS = 64
RESET_AT = (40, 48, 56)
DENSE_NAMES = (
    "input_to_hidden",
    "hidden_to_hidden",
    "output_to_hidden",
    "input_to_output",
    "hidden_to_output",
    "output_to_output",
)


def load_config(name: str):
    return neat3p.Config(
        neat3p.DefaultGenome,
        neat3p.DefaultReproduction,
        neat3p.DefaultSpeciesSet,
        neat3p.DefaultStagnation,
        CONFIG_PATHS[name],
    )


def make_inputs(n_inputs: int, seed: int) -> np.ndarray:
    """(TICKS, n_inputs) float32: normal rows, then zeros / +-1e3 / NaN / inf tails, one per reset segment."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(TICKS, n_inputs)).astype(np.float32)
    x[40:42] = 0.0
    x[42:46] = rng.choice([-1e3, 1e3], size=(4, n_inputs)).astype(np.float32)
    x[52, 0] = np.nan
    x[60, 0] = np.inf
    x[61, min(1, n_inputs - 1)] = -np.inf
    return x


def _node(key, bias, response, activation="sigmoid"):
    return DefaultNodeGene(key=key, bias=bias, response=response, activation=activation, aggregation="sum")


def _conn(i, o, weight, enabled=True):
    return DefaultConnectionGene(key=(i, o), weight=weight, enabled=enabled)


def _genome(key, nodes, conns):
    g = DefaultGenome(key=key)
    for n in nodes:
        g.nodes[n.key] = n
    for c in conns:
        g.connections[c.key] = c
    return g


def edge_genomes() -> list[tuple[str, DefaultGenome]]:
    """Hand-built genomes on edge.cfg (inputs -1..-3, outputs 0..1) covering the quirks of RecurrentNet.create."""
    out0, out1 = _node(0, 0.1, 1.0), _node(1, -0.2, 1.0)
    h = lambda k, b=0.3, r=1.0, a="sin": _node(k, b, r, a)  # noqa: E731
    cases = {
        "no_hidden": ([out0, out1], [_conn(-1, 0, 0.7), _conn(-2, 1, -1.3), _conn(-3, 0, 2.0), _conn(-1, 1, 0.5)]),
        "isolated_hidden": ([out0, out1, h(2, 0.9)], [_conn(-1, 0, 1.1), _conn(-2, 1, 0.4)]),
        "dead_branch": (
            [out0, out1, h(2), h(3)],
            [_conn(-1, 2, 1.5), _conn(2, 3, 1.0), _conn(-2, 0, 0.8), _conn(-3, 1, -0.6)],
        ),
        "output_into_dead_node": (
            [out0, out1, h(2)],
            [_conn(-1, 0, 1.2), _conn(-2, 1, 0.9), _conn(0, 2, 2.0), _conn(2, 2, 1.0)],
        ),
        "plain_hidden": (
            [out0, out1, h(2), h(3, 0.1, 1.5)],
            [_conn(-1, 2, 0.9), _conn(-2, 3, -0.4), _conn(2, 0, 1.7), _conn(3, 1, 0.8), _conn(2, 3, 0.6)],
        ),
        "self_loop": (
            [out0, out1, h(2)],
            [_conn(-1, 2, 0.8), _conn(2, 2, 1.4), _conn(2, 0, 1.0), _conn(-3, 1, 0.5)],
        ),
        "hidden_cycle": (
            [out0, out1, h(2), h(3)],
            [_conn(-1, 2, 1.0), _conn(2, 3, 1.2), _conn(3, 2, -0.9), _conn(3, 1, 0.7), _conn(2, 0, 0.3)],
        ),
        "output_to_output": (
            [out0, out1],
            [_conn(-1, 0, 1.0), _conn(-2, 1, 1.0), _conn(0, 1, 1.5), _conn(1, 0, -1.5), _conn(1, 1, 0.7)],
        ),
        "output_to_hidden": (
            [out0, out1, h(2)],
            [_conn(-1, 0, 1.0), _conn(0, 2, 1.3), _conn(2, 1, 0.9), _conn(-3, 1, -0.2)],
        ),
        "disabled_connections": (
            [out0, out1, h(2)],
            [_conn(-1, 0, 1.0), _conn(-2, 2, 2.0, enabled=False), _conn(2, 1, 1.0), _conn(-3, 1, 0.5, enabled=False)],
        ),
        "all_disabled": (
            [out0, out1, h(2)],
            [_conn(-1, 0, 1.0, enabled=False), _conn(-2, 2, 2.0, enabled=False), _conn(2, 1, 1.0, enabled=False)],
        ),
        "extreme_values": (
            [_node(0, 30.0, 30.0), _node(1, -30.0, -30.0), h(2, 30.0, -30.0, "gauss")],
            [_conn(-1, 2, 30.0), _conn(2, 0, -30.0), _conn(-2, 1, 30.0), _conn(2, 1, 30.0), _conn(-3, 0, -30.0)],
        ),
    }
    return [(f"edge_{name}", _genome(100 + i, nodes, conns)) for i, (name, (nodes, conns)) in enumerate(cases.items())]


def random_genomes(config_name: str, fresh: int, mutated: int, mutations: tuple[int, int], seed: int):
    config = load_config(config_name)
    gc = config.genome_config
    cases = []
    for i in range(fresh + mutated):
        random.seed(seed * 1000 + i)
        np.random.seed(seed * 1000 + i)
        g = DefaultGenome(key=i)
        g.configure_new(gc)
        kind = "fresh"
        if i >= fresh:
            kind = "mutated"
            rng = random.Random(seed * 77 + i)
            for _ in range(rng.randint(*mutations)):
                g.mutate(gc)
        cases.append((f"{config_name}_{kind}_{i}", g))
    return cases


def build_cases() -> list[tuple[str, str, DefaultGenome]]:
    cases = []
    for name, g in random_genomes("lifesim_recurrent", 6, 10, (60, 200), seed=11):
        cases.append((name, "lifesim_recurrent", g))
    for name, g in random_genomes("stress", 6, 14, (30, 120), seed=22):
        cases.append((name, "stress", g))
    for name, g in random_genomes("cartpole", 4, 6, (20, 80), seed=33):
        cases.append((name, "cartpole", g))
    for name, g in edge_genomes():
        cases.append((name, "edge", g))
    return cases


def torch_reference(genome, config, inputs: np.ndarray, use_current_activs: bool):
    """Run torch's RecurrentNet (float64, CPU) over ``inputs``, resetting at RESET_AT. Returns outputs and the net."""
    import torch

    from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

    net = RecurrentNet.create(genome, config, use_current_activs=use_current_activs, device="cpu")
    outs = []
    for t in range(inputs.shape[0]):
        if t in RESET_AT:
            net.reset(batch_size=1)
        outs.append(net.activate(torch.from_numpy(inputs[t : t + 1])).numpy()[0].copy())
    return np.stack(outs), net


def case_arrays(name: str, genome, config, seed: int) -> dict[str, np.ndarray]:
    gc = config.genome_config
    inputs = make_inputs(len(gc.input_keys), seed)
    arrays = {f"{name}/inputs": inputs}
    net = None
    for flag, tag in ((True, "cur"), (False, "prev")):
        outs, net = torch_reference(genome, config, inputs, use_current_activs=flag)
        arrays[f"{name}/out_{tag}"] = outs
    for dense in DENSE_NAMES:
        if hasattr(net, dense):
            arrays[f"{name}/dense/{dense}"] = getattr(net, dense).numpy().copy()
    for attr in ("output_biases", "output_responses", "hidden_biases", "hidden_responses"):
        if hasattr(net, attr):
            arrays[f"{name}/{attr}"] = getattr(net, attr).numpy().copy()
    return arrays


def main() -> int:
    import torch

    os.makedirs(FIXTURES, exist_ok=True)
    meta, arrays = [], {}
    configs = {}
    for idx, (name, config_name, genome) in enumerate(build_cases()):
        config = configs.setdefault(config_name, load_config(config_name))
        arrays.update(case_arrays(name, genome, config, seed=idx))
        meta.append({"name": name, "config": config_name, "genome": json.loads(genome.model_dump_json())})
    with open(CASES_JSON, "w") as f:
        json.dump({"torch_version": torch.__version__, "ticks": TICKS, "reset_at": list(RESET_AT), "cases": meta}, f)
    np.savez_compressed(ARRAYS_NPZ, **arrays)
    print(f"{len(meta)} cases, {len(arrays)} arrays -> {FIXTURES} (torch {torch.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
