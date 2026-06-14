"""
One-time fixture generator — run from life-sim-312 conda env.

Generates deterministic outputs from the current aievolution network
implementations and saves them as .npy files for parity testing.

Usage:
    conda run --no-capture-output -n life-sim-312 \
        python neat3p/tests/nn/fixtures/generate_fixtures.py
"""

import os

import numpy as np
import torch

from neat3p.nn.modules.activations import sigmoid_activation
from neat3p.nn.phenotypes.cppn import create_cppn
from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

import neat3p

_HERE = os.path.dirname(__file__)
_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "xor.cfg")


def _make_config():
    return neat3p.Config(
        neat3p.DefaultGenome,
        neat3p.DefaultReproduction,
        neat3p.DefaultSpeciesSet,
        neat3p.DefaultStagnation,
        _CFG_PATH,
    )


def generate_recurrent_net_fixture():
    """Fixed genome → RecurrentNet → fixed input → output snapshot."""
    import random as _random

    torch.manual_seed(0)
    np.random.seed(0)
    _random.seed(0)

    config = _make_config()

    # Deterministic population: take the first genome (key 1)
    pop = neat3p.Population(config)

    # Assign a dummy fitness so we can pick a genome
    genome_id, genome = next(iter(sorted(pop.population.items())))

    net = RecurrentNet.create(genome, config, batch_size=2, activation=sigmoid_activation)

    fixed_input = [[0.25, 0.75], [1.0, 0.0]]
    out = net.activate(fixed_input)  # moves to CUDA internally
    out_cpu = out.cpu().numpy()

    save_path = os.path.join(_HERE, "recurrent_net_output.npy")
    np.save(save_path, out_cpu)
    print(f"Saved RecurrentNet fixture → {save_path}  shape={out_cpu.shape}")
    return out_cpu


def generate_cppn_fixture():
    """Fixed genome → create_cppn → node activations snapshot."""
    import random as _random

    torch.manual_seed(0)
    np.random.seed(0)
    _random.seed(0)

    # CPPN needs a config with appropriate leaf names.
    # We reuse the XOR config (2-input, 1-output) as a minimal genome.
    config = _make_config()
    pop = neat3p.Population(config)
    genome_id, genome = next(iter(sorted(pop.population.items())))

    nodes = create_cppn(
        genome,
        config,
        leaf_names=["x", "y"],
        node_names=["out"],
    )
    out_node = nodes[0]

    x = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float32, device="cuda")
    y = torch.tensor([0.1, 0.9, 0.5], dtype=torch.float32, device="cuda")
    result = out_node(x=x, y=y)
    result_cpu = result.cpu().numpy()

    save_path = os.path.join(_HERE, "cppn_output.npy")
    np.save(save_path, result_cpu)
    print(f"Saved CPPN fixture → {save_path}  shape={result_cpu.shape}")
    return result_cpu


if __name__ == "__main__":
    print("Generating parity fixtures from neat3p.nn …")
    generate_recurrent_net_fixture()
    generate_cppn_fixture()
    print("Done.")
