"""G3: the native Eigen evaluator equals the stored torch outputs (and runs with torch blocked)."""

import _parity as P
import numpy as np
import pytest

from neat3p.runtime import build_recurrent_layout, compile_recurrent

CASES = P.case_names()


def _native_run(genome, config, fx, reset_at, use_current_activs, dtype="float64"):
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=use_current_activs)
    net = compile_recurrent(layout, dtype=dtype)
    return P.run_sequence(net.activate, net.reset, fx["inputs"], reset_at)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("use_current_activs", [True, False])
def test_native_matches_torch(case, use_current_activs):
    genome, config, fx, reset_at = P.load_case(case)
    got = _native_run(genome, config, fx, reset_at, use_current_activs)
    want = fx["out_cur" if use_current_activs else "out_prev"]

    assert P.max_abs_err(got, want) <= P.ATOL64
    P.assert_actions_match(got, want)


@pytest.mark.parametrize("case", ["lifesim_recurrent_mutated_10", "stress_mutated_12", "cartpole_mutated_8"])
def test_native_matches_numpy_reference_bitwise_close(case):
    from neat3p.runtime import ReferenceRecurrentNet

    genome, config, fx, reset_at = P.load_case(case)
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    ref = ReferenceRecurrentNet(layout)
    want = P.run_sequence(lambda x: ref.activate(x[None, :])[0], ref.reset, fx["inputs"], reset_at)
    got = _native_run(genome, config, fx, reset_at, True)
    assert P.max_abs_err(got, want) <= 1e-13


@pytest.mark.parametrize("case", ["lifesim_recurrent_mutated_10", "stress_mutated_12", "cartpole_fresh_1"])
def test_native_float32_is_close_to_float64(case):
    genome, config, fx, reset_at = P.load_case(case)
    got = _native_run(genome, config, fx, reset_at, True, dtype="float32")
    want = fx["out_cur"]
    finite = np.isfinite(want)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.max(np.abs(got[finite] - want[finite])) <= 1e-3  # smoke only: float32 is not a gate


def test_state_persists_and_reset_clears_it():
    genome, config, fx, _ = P.load_case("stress_mutated_12")
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    net = compile_recurrent(layout)
    x = fx["inputs"][0]
    first = net.activate(x).copy()
    second = net.activate(x).copy()
    assert not np.array_equal(first, second)  # recurrent state feeds back
    assert np.any(net.output_state != 0)
    net.reset()
    assert not np.any(net.output_state) and not np.any(net.hidden_state)
    assert np.array_equal(net.activate(x), first)


def test_activate_accepts_row_vector_and_float32():
    genome, config, fx, _ = P.load_case("cartpole_fresh_0")
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    a, b = compile_recurrent(layout), compile_recurrent(layout)
    x32 = fx["inputs"][0]
    assert x32.dtype == np.float32
    assert np.array_equal(a.activate(x32), b.activate(x32[None, :]))


def test_activate_returns_fresh_float64_arrays():
    genome, config, fx, _ = P.load_case("cartpole_fresh_0")
    net = compile_recurrent(build_recurrent_layout(genome, config.genome_config, use_current_activs=True))
    out1 = net.activate(fx["inputs"][0])
    keep = out1.copy()
    net.activate(fx["inputs"][1])
    assert out1.dtype == np.float64 and out1.shape == (net.n_outputs,)
    assert np.array_equal(out1, keep)  # not a view of the net's state


def test_errors_are_value_errors():
    genome, config, fx, _ = P.load_case("cartpole_fresh_0")
    layout = build_recurrent_layout(genome, config.genome_config, use_current_activs=True)
    net = compile_recurrent(layout)
    with pytest.raises(ValueError, match="expected input of length"):
        net.activate(np.zeros(net.n_inputs + 1))
    with pytest.raises(ValueError):
        net.activate(np.zeros((2, net.n_inputs)))
    with pytest.raises(ValueError, match="unknown dtype"):
        compile_recurrent(layout, dtype="float16")

    from neat3p import _neat3p

    z = np.zeros
    with pytest.raises(ValueError, match="bad input_to_hidden shape"):
        _neat3p.RecurrentKernelF64(
            z((1, 2)), z((0, 0)), z((0, 2)), z((2, 2)), z((2, 0)), z((2, 2)), z(0), z(0), z(2), z(2), True
        )
    with pytest.raises(ValueError, match="n_internal_steps"):
        _neat3p.RecurrentKernelF64(
            z((0, 2)), z((0, 0)), z((0, 2)), z((2, 2)), z((2, 0)), z((2, 2)), z(0), z(0), z(2), z(2), True, 0
        )


def test_native_runs_with_torch_blocked():
    import os
    import subprocess
    import sys

    import neat3p

    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(neat3p.__file__))}
    code = (
        "import sys; sys.modules['torch']=None\n"
        "import numpy as np, neat3p\n"
        "from neat3p.runtime import CompiledRecurrentNet\n"
        "assert 'torch' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
