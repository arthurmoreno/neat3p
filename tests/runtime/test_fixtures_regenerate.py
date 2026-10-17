"""The committed fixtures can be reproduced (guards fixture drift); needs torch, skipped without it."""

import json

import _parity as P
import make_fixtures as mf
import numpy as np
import pytest

pytest.importorskip("torch")


@pytest.mark.parametrize(
    "case_name", ["lifesim_recurrent_mutated_10", "stress_mutated_12", "cartpole_fresh_1", "edge_hidden_cycle"]
)
def test_regenerated_case_equals_committed(case_name):
    built = {name: (cfg, g) for name, cfg, g in mf.build_cases()}
    config_name, genome = built[case_name]
    index = [n for n, _, _ in mf.build_cases()].index(case_name)

    committed_genome, _, committed, _ = P.load_case(case_name)
    assert json.loads(genome.model_dump_json()) == json.loads(committed_genome.model_dump_json())

    arrays = mf.case_arrays(case_name, genome, mf.load_config(config_name), seed=index)
    prefix = case_name + "/"
    assert {k[len(prefix) :] for k in arrays} == set(committed)
    for key, value in arrays.items():
        want = committed[key[len(prefix) :]]
        assert P.max_abs_err(np.asarray(value, dtype=np.float64), np.asarray(want, dtype=np.float64)) <= 1e-12, key
