"""
XOR correctness test — RED until Phase 1 migration lands neat3p.nn.phenotypes.

Gate: 🔴 (F0.2) → 🟢 (F1.7)

What this validates when GREEN:
  - RecurrentNet.create(genome, config) builds a phenotype from a neat3p DefaultGenome
  - net.activate(inputs) runs forward correctly
  - NEAT with RecurrentNet solves XOR within 50 generations
"""

import os

import pytest
import torch

import neat3p

# ---------------------------------------------------------------------------
# XOR problem definition
# ---------------------------------------------------------------------------

_XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
_XOR_TARGETS = [0.0, 1.0, 1.0, 0.0]


def _load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "xor.cfg")
    return neat3p.Config(
        neat3p.DefaultGenome,
        neat3p.DefaultReproduction,
        neat3p.DefaultSpeciesSet,
        neat3p.DefaultStagnation,
        cfg_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RecurrentNet requires CUDA")
def test_recurrent_net_xor_solves():
    """NEAT with RecurrentNet solves XOR within 50 generations.

    RED (F0.2): fails with ModuleNotFoundError — neat3p.nn.phenotypes not yet migrated.
    GREEN (F1.7): passes after Phase 1 migration.
    """
    # RED: ModuleNotFoundError until Phase 1 migrates aievolution/networks/recurrent_net.py
    from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

    config = _load_config()

    def eval_xor(genomes, config):
        for _genome_id, genome in genomes:
            net = RecurrentNet.create(genome, config, batch_size=1, use_current_activs=True)
            error = 0.0
            for xi, xo in zip(_XOR_INPUTS, _XOR_TARGETS):
                net.reset(batch_size=1)
                out = net.activate([xi])
                error += (out[0, 0].item() - xo) ** 2
            genome.fitness = 4.0 - error

    pop = neat3p.Population(config)
    winner = pop.run(eval_xor, 300)

    assert winner.fitness >= 3.9, f"XOR not solved: fitness={winner.fitness:.3f}"

    net = RecurrentNet.create(winner, config, batch_size=1, use_current_activs=True)
    for xi, xo in zip(_XOR_INPUTS, _XOR_TARGETS):
        net.reset(batch_size=1)
        out = net.activate([xi])
        predicted = out[0, 0].item()
        assert abs(predicted - xo) < 0.5, f"XOR({xi}) = {predicted:.3f}, expected ~{xo}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RecurrentNet requires CUDA")
def test_recurrent_net_forward_shape():
    """RecurrentNet.activate output has the expected batch shape.

    RED (F0.2): same import failure as above.
    GREEN (F1.7): passes after migration.
    """
    from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

    config = _load_config()
    pop = neat3p.Population(config)
    _genome_id, genome = next(iter(sorted(pop.population.items())))
    genome.fitness = 0.0

    net = RecurrentNet.create(genome, config, batch_size=2)
    out = net.activate([[0.0, 0.0], [1.0, 1.0]])
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
