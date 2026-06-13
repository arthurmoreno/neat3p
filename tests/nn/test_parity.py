"""
Characterization / parity test — gate for Phase 1 migration.

Gate: 🛡 (F0.4 generates fixtures) → 🛡 (F1.6 re-points imports + asserts parity)

Phase 0:  fixtures in tests/nn/fixtures/ are generated from aievolution.networks.*
          by running:  python tests/nn/fixtures/generate_fixtures.py
          This test is SKIPPED until neat3p.nn.phenotypes exists (Phase 1).

Phase 1.6: remove the skip, change imports to neat3p.nn.phenotypes, assert
           byte-identical outputs vs the stored fixtures.

If parity FAILS after Phase 1 migration, the move changed numeric behaviour —
stop and fix before merging.
"""

import os

import numpy as np
import pytest
import torch

# ── fixture paths ─────────────────────────────────────────────────────────────

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_RNN_OUTPUT = os.path.join(_FIXTURES, "recurrent_net_output.npy")
_CPPN_OUTPUT = os.path.join(_FIXTURES, "cppn_output.npy")

# ── Phase-0 skip guard ────────────────────────────────────────────────────────


def _nn_migrated():
    try:
        from neat3p.nn.phenotypes.recurrent_net import RecurrentNet  # noqa: F401

        return True
    except ImportError:
        return False


_SKIP_REASON = "neat3p.nn.phenotypes not yet migrated (Phase 1 — F1.6 removes this skip)"

# ── helpers ───────────────────────────────────────────────────────────────────

import neat3p


def _load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "xor.cfg")
    return neat3p.Config(
        neat3p.DefaultGenome,
        neat3p.DefaultReproduction,
        neat3p.DefaultSpeciesSet,
        neat3p.DefaultStagnation,
        cfg_path,
    )


def _seed_all(seed=0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _first_genome(config):
    """Return the first (lowest-key) genome from a seeded fresh population."""
    _seed_all(0)
    pop = neat3p.Population(config)
    gid, genome = next(iter(sorted(pop.population.items())))
    return genome


# ── parity tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _nn_migrated(), reason=_SKIP_REASON)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="RecurrentNet requires CUDA")
def test_recurrent_net_parity():
    """
    RecurrentNet output from neat3p.nn is byte-identical to the pre-migration
    aievolution.networks baseline stored in fixtures/recurrent_net_output.npy.

    Fixed inputs: [[0.25, 0.75], [1.0, 0.0]]  (batch_size=2, 2-input XOR genome)
    Genome: first genome of a fresh population seeded with seed=0.
    """
    from neat3p.nn.modules.activations import sigmoid_activation
    from neat3p.nn.phenotypes.recurrent_net import RecurrentNet  # Phase 1+ import

    config = _load_config()
    genome = _first_genome(config)

    net = RecurrentNet.create(genome, config, batch_size=2, activation=sigmoid_activation)
    out = net.activate([[0.25, 0.75], [1.0, 0.0]])
    out_cpu = out.cpu().numpy()

    expected = np.load(_RNN_OUTPUT)
    np.testing.assert_array_equal(
        out_cpu, expected, err_msg="RecurrentNet output does not match pre-migration baseline"
    )


@pytest.mark.skipif(not _nn_migrated(), reason=_SKIP_REASON)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPPN requires CUDA")
def test_cppn_parity():
    """
    create_cppn output from neat3p.nn is byte-identical to the pre-migration
    aievolution.networks.cppn baseline stored in fixtures/cppn_output.npy.
    """
    from neat3p.nn.phenotypes.cppn import create_cppn  # Phase 1+ import

    config = _load_config()
    genome = _first_genome(config)

    nodes = create_cppn(genome, config, leaf_names=["x", "y"], node_names=["out"])
    out_node = nodes[0]

    x = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float32, device="cuda")
    y = torch.tensor([0.1, 0.9, 0.5], dtype=torch.float32, device="cuda")
    result = out_node(x=x, y=y)
    result_cpu = result.cpu().numpy()

    expected = np.load(_CPPN_OUTPUT)
    np.testing.assert_array_equal(result_cpu, expected, err_msg="CPPN output does not match pre-migration baseline")


def test_fixtures_exist():
    """Fixtures must be present — generated once by generate_fixtures.py."""
    assert os.path.exists(_RNN_OUTPUT), f"Missing: {_RNN_OUTPUT}"
    assert os.path.exists(_CPPN_OUTPUT), f"Missing: {_CPPN_OUTPUT}"
