"""A4: the NativeRecurrentNet brain contract, and the torch-free guarantee."""

import os
import subprocess
import sys

import _parity as P
import numpy as np
import pytest

from neat3p.runtime import NativeRecurrentNet


def _net(case="cartpole_mutated_8", **kw):
    genome, config, fx, _ = P.load_case(case)
    gc = config.genome_config
    return NativeRecurrentNet(genome, len(gc.input_keys), len(gc.output_keys), config, **kw), fx


def test_forward_shape_dtype_and_call_alias():
    net, fx = _net()
    out = net.forward(fx["inputs"][0][None, :])
    assert out.shape == (1, net.action_dim) and out.dtype == np.float64
    net.reset()
    assert np.array_equal(net(fx["inputs"][0]), out)  # __call__ is forward; 1-D input accepted


def test_forward_matches_stored_torch_outputs():
    net, fx = _net("lifesim_recurrent_mutated_10")
    for t in range(40):
        got = net.forward(fx["inputs"][t][None, :])[0]
        assert np.max(np.abs(got - fx["out_cur"][t])) <= P.ATOL64


def test_get_action_is_argmax_when_not_exploring():
    net, fx = _net(explore_rate=0.0)
    ref, _ = _net(explore_rate=0.0)
    for t in range(30):
        action = net.get_action(fx["inputs"][t])
        assert action.shape == (1, 1) and action.dtype == np.int64
        assert action[0, 0] == int(np.argmax(ref.forward(fx["inputs"][t])))


def test_exploration_rate_and_seed_reproducibility():
    n = 10_000
    net, fx = _net(explore_rate=0.125, seed=7)
    x = fx["inputs"][0]
    greedy = int(np.argmax(_net(explore_rate=0.0)[0].forward(x)))
    a = [int(net.get_action(x)[0, 0]) for _ in range(n)]
    # An exploring draw lands on the greedy action with probability 1/action_dim, so count only a lower bound.
    deviating = sum(1 for v in a if v != greedy)
    expected = 0.125 * n * (1 - 1 / net.action_dim)
    assert abs(deviating - expected) < 0.02 * n
    net2, _ = _net(explore_rate=0.125, seed=7)
    assert a == [int(net2.get_action(x)[0, 0]) for _ in range(n)]


def test_actions_stay_in_range_when_always_exploring():
    net, fx = _net(explore_rate=1.0, seed=3)
    seen = {int(net.get_action(fx["inputs"][0])[0, 0]) for _ in range(500)}
    assert seen == set(range(net.action_dim))


def test_reset_clears_state_and_rejects_batches():
    net, fx = _net("stress_mutated_12", explore_rate=0.0)
    first = net.forward(fx["inputs"][0]).copy()
    net.forward(fx["inputs"][1])
    net.reset()
    assert np.array_equal(net.forward(fx["inputs"][0]), first)
    with pytest.raises(ValueError, match="batch_size=1"):
        net.reset(batch_size=2)
    with pytest.raises(ValueError, match="one creature at a time"):
        net.forward(np.zeros((2, net.state_dim)))
    with pytest.raises(ValueError, match="expected state of length"):
        net.forward(np.zeros(net.state_dim + 1))


def test_dimension_mismatch_is_rejected_at_construction():
    genome, config, _, _ = P.load_case("cartpole_fresh_0")
    with pytest.raises(ValueError, match="inputs"):
        NativeRecurrentNet(genome, 5, 2, config)
    with pytest.raises(ValueError, match="outputs"):
        NativeRecurrentNet(genome, 4, 3, config)


def test_device_alias_is_ignored_and_dtype_is_selectable():
    net, fx = _net(device_alias="cuda:0", dtype="float32")
    assert net.net.dtype == "float32"
    assert net.forward(fx["inputs"][0]).shape == (1, net.action_dim)


def test_accepts_a_cpu_torch_tensor_without_numpy_conversion_by_caller():
    torch = pytest.importorskip("torch")
    net, fx = _net(explore_rate=0.0)
    ref, _ = _net(explore_rate=0.0)
    t = torch.from_numpy(fx["inputs"][0]).unsqueeze(0)
    assert np.array_equal(net.forward(t), ref.forward(fx["inputs"][0]))


def test_argmax_matches_torch_including_nan_rows():
    torch = pytest.importorskip("torch")
    mismatches = 0
    for case in P.case_names():
        _, _, fx, _ = P.load_case(case)
        out = fx["out_cur"]
        mine = np.argmax(out, axis=1)
        theirs = torch.argmax(torch.from_numpy(out), dim=1).numpy()
        mismatches += int((mine != theirs).sum())
    assert mismatches == 0


def test_torch_free_import_and_run():
    """G7-A: neat3p and neat3p.runtime import and step a brain with torch made unimportable."""
    import neat3p

    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(neat3p.__file__))}
    code = f"""
import sys
sys.modules['torch'] = None
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
import numpy as np, _parity as P
import neat3p, neat3p.runtime
genome, config, fx, _ = P.load_case('lifesim_recurrent_mutated_10')
net = neat3p.runtime.NativeRecurrentNet(genome, 90, 3, config, explore_rate=0.0)
out = net.forward(fx['inputs'][0])
assert np.max(np.abs(out[0] - fx['out_cur'][0])) <= 1e-12
loaded = [m for m, v in sys.modules.items() if v is not None and m.split('.')[0] in ('torch', 'torchvision')]
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
