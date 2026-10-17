"""G1: the torch-free layout builds exactly the matrices torch's RecurrentNet.create builds."""

import copy
import os
import subprocess
import sys

import _parity as P
import numpy as np
import pytest

from neat3p.genes import DefaultConnectionGene
from neat3p.runtime import build_recurrent_layout

CASES = P.case_names()
DENSE = (
    "input_to_hidden",
    "hidden_to_hidden",
    "output_to_hidden",
    "input_to_output",
    "hidden_to_output",
    "output_to_output",
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("use_current_activs", [True, False])
def test_dense_matrices_equal_torch(case, use_current_activs):
    genome, config, fx, _ = P.load_case(case)
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=use_current_activs)
    dense = layout.to_dense()

    assert set(dense) == {n for n in DENSE if f"dense/{n}" in fx}
    for name, mat in dense.items():
        assert np.array_equal(mat, fx[f"dense/{name}"]), name
    assert np.array_equal(layout.output_biases, fx["output_biases"])
    assert np.array_equal(layout.output_responses, fx["output_responses"])
    if layout.n_hidden:
        assert np.array_equal(layout.hidden_biases, fx["hidden_biases"])
        assert np.array_equal(layout.hidden_responses, fx["hidden_responses"])
    else:
        assert "hidden_biases" not in fx


@pytest.mark.parametrize("case", CASES)
def test_coo_indices_in_range_and_unique(case):
    genome, config, _, _ = P.load_case(case)
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    for name, blk in layout.blocks.items():
        rows, cols = layout.block_shape(name)
        assert blk.rows.dtype == np.int64 and blk.cols.dtype == np.int64 and blk.weights.dtype == np.float64
        assert len(blk.rows) == len(blk.cols) == len(blk.weights)
        if len(blk.rows):
            assert 0 <= blk.rows.min() and blk.rows.max() < rows
            assert 0 <= blk.cols.min() and blk.cols.max() < cols
            assert len(set(zip(blk.rows.tolist(), blk.cols.tolist()))) == len(blk.rows)


def test_layout_is_independent_of_the_genome_after_building():
    genome, config, _, _ = P.load_case("edge_plain_hidden")
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    before = {k: v.copy() for k, v in layout.to_dense().items()}
    for conn in genome.connections.values():
        conn.weight = 99.0
    for node in genome.nodes.values():
        node.bias = 99.0
    after = layout.to_dense()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert 99.0 not in layout.output_biases


def test_quirks_are_replicated():
    """Hidden = every non-output node; a connection survives if either end is required."""
    genome, config, _, _ = P.load_case("edge_dead_branch")
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    assert layout.hidden_keys == (2, 3)  # both hidden nodes exist even though neither is required
    assert all(len(b.rows) == 0 for n, b in layout.blocks.items() if "hidden" in n)  # their links are dropped

    genome, config, _, _ = P.load_case("edge_output_into_dead_node")
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    assert len(layout.blocks["output_to_hidden"].rows) == 1  # output -> dead node is kept (output end is required)
    assert len(layout.blocks["hidden_to_hidden"].rows) == 0  # the dead node's self-loop has no required end


def test_unknown_node_raises_value_error():
    genome, config, _, _ = P.load_case("edge_plain_hidden")
    broken = copy.deepcopy(genome)
    broken.connections[(7, 0)] = DefaultConnectionGene(key=(7, 0), weight=1.0, enabled=True)
    with pytest.raises(ValueError, match="Invalid connection from key 7 to key 0"):
        build_recurrent_layout(broken, config.genome_config, use_current_activs=True)


def test_runtime_import_does_not_pull_torch():
    import neat3p

    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(neat3p.__file__))}
    code = "import sys; sys.modules['torch']=None; import neat3p, neat3p.runtime"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_prune_empty_matches_torch_semantics():
    pytest.importorskip("torch")
    from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

    for case in ("edge_dead_branch", "edge_no_hidden", "lifesim_recurrent_mutated_10"):
        genome, config, _, _ = P.load_case(case)
        net = RecurrentNet.create(genome, config, prune_empty=True, device="cpu")
        layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=False, prune_empty=True)
        assert np.array_equal(layout.output_biases, net.output_biases.numpy())
        for name, mat in layout.to_dense().items():
            assert np.array_equal(mat, getattr(net, name).numpy()), name
