"""G4/G5 (CartPole): the native evaluator drives the benchmark harness exactly like the torch brain.

Needs the ``[bench]`` extra (torch is the oracle here); skipped otherwise.
"""

import os

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.artifacts import load_winner, save_winner, select_action  # noqa: E402
from benchmarks.models import MODELS  # noqa: E402
from benchmarks.runners.gym_eval import _seed_for, run_neat_gym  # noqa: E402
from neat3p.nn.composite import NEATRecurrentNet  # noqa: E402
from neat3p.runtime import NativeRecurrentNet  # noqa: E402

CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../benchmarks/configs/cartpole.cfg"))
TORCH_CPU = {"device_alias": "cpu"}


def _evolve(net_class, kwargs, seed, generations=4, episodes=3):
    return run_neat_gym(
        "CartPole-v1", CFG, generations, episodes, seed=seed, net_class=net_class, net_kwargs=kwargs, verbose=False
    )


def _rollout_actions(net, forward, seed):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    net.reset(batch_size=1)
    actions, total, done = [], 0.0, False
    while not done:
        out = forward(net, obs)
        action = int(out.argmax(dim=1).item()) if hasattr(out, "detach") else int(np.argmax(out, axis=1)[0])
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    env.close()
    return actions, total


def _torch_forward(net, obs):
    return net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))


def test_native_adapter_is_registered_and_shares_the_recurrent_config():
    from benchmarks.tasks import CARTPOLE

    assert "recurrent_net_native" in MODELS
    assert MODELS["recurrent_net_native"].config_path(CARTPOLE) == MODELS["recurrent_net"].config_path(CARTPOLE)


@pytest.mark.parametrize("seed", [42, 7])
def test_champion_replay_gives_identical_actions_and_returns(seed):
    """G4: a torch-evolved champion replayed natively takes the same action at every step."""
    result = _evolve(NEATRecurrentNet, TORCH_CPU, seed)
    gc = result.config.genome_config
    state_dim, action_dim = len(gc.input_keys), len(gc.output_keys)

    torch_net = NEATRecurrentNet(result.winner, state_dim, action_dim, result.config, device_alias="cpu")
    native_net = NativeRecurrentNet(result.winner, state_dim, action_dim, result.config)

    steps = 0
    for i in range(20):
        want = _rollout_actions(torch_net, _torch_forward, _seed_for(seed + 1, i))
        got = _rollout_actions(native_net, _torch_forward, _seed_for(seed + 1, i))
        assert got == want, f"episode {i} diverged"
        steps += len(want[0])
    assert steps > 200  # the replay really ran non-trivial episodes


@pytest.mark.parametrize("seed", [42, 7])
def test_native_evolution_matches_torch_generation_by_generation(seed):
    """G5 (short): same seed -> same population statistics every generation, torch vs native."""
    torch_run = _evolve(NEATRecurrentNet, TORCH_CPU, seed, generations=5)
    native_run = _evolve(NativeRecurrentNet, {}, seed, generations=5)
    assert native_run.generation_stats == torch_run.generation_stats
    assert native_run.winner.fitness == torch_run.winner.fitness


def test_native_winner_artifact_roundtrip(tmp_path):
    result = _evolve(NativeRecurrentNet, {}, seed=42, generations=2)
    path = save_winner(str(tmp_path), "recurrent_net_native", result, "CartPole-v1", 42, CFG)
    pkg, net, style = load_winner(path, device="cpu")
    assert style == "module" and isinstance(net, NativeRecurrentNet)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=1)
    net.reset(batch_size=1)
    assert select_action(net, obs, style) in (0, 1)
    env.close()
