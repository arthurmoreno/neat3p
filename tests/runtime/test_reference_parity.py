"""G2: the numpy reference evaluator equals torch over 64 ticks (resets, zeros, +-1e3, NaN and inf included)."""

import _parity as P
import numpy as np
import pytest

from neat3p.runtime import ReferenceRecurrentNet, build_recurrent_layout

CASES = P.case_names()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("use_current_activs", [True, False])
def test_reference_matches_torch(case, use_current_activs):
    genome, config, fx, reset_at = P.load_case(case)
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=use_current_activs)
    net = ReferenceRecurrentNet(layout)
    got = P.run_sequence(lambda x: net.activate(x[None, :])[0], net.reset, fx["inputs"], reset_at)
    want = fx["out_cur" if use_current_activs else "out_prev"]

    assert P.max_abs_err(got, want) <= P.ATOL64
    P.assert_actions_match(got, want)


def test_reference_flags_actually_differ():
    """Guards the fixtures: use_current_activs must change outputs for a brain with a hidden path."""
    genome, config, fx, _ = P.load_case("lifesim_recurrent_mutated_10")
    assert not np.array_equal(fx["out_cur"], fx["out_prev"])


def test_reference_ignores_evolved_activation_genes():
    """The live brain applies sigmoid(5x) to every node, whatever the activation gene says."""
    genome, config, fx, reset_at = P.load_case("edge_plain_hidden")
    assert {n.activation for n in genome.nodes.values()} != {"sigmoid"}
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    net = ReferenceRecurrentNet(layout)
    got = P.run_sequence(lambda x: net.activate(x[None, :])[0], net.reset, fx["inputs"], reset_at)
    assert P.max_abs_err(got, fx["out_cur"]) <= P.ATOL64
