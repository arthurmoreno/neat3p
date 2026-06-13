"""
CartPole-v1 benchmark — RED until Phase 2 harness + migration are complete.

Gate: 🔴 (F0.3) → 🟢 (F2.2)

Solve criterion: mean episode reward ≥ 475 over SOLVE_EPISODES rollouts,
achieved within MAX_GENERATIONS NEAT generations.

What this validates when GREEN:
  - neat3p.nn.phenotypes.recurrent_net.RecurrentNet drives a CartPole agent
  - neat3p.benchmarks.runners.gym_eval.run_neat_gym provides the eval harness
  - NEAT (memoryless, feed_forward=True) reliably solves CartPole-v1
  - Run is seeded so CI is reproducible
"""

import os

import pytest

# ---------------------------------------------------------------------------
# Constants — budget tuned empirically in F2.2
# ---------------------------------------------------------------------------

SEED = 42
MAX_GENERATIONS = 300
EVAL_EPISODES = 10  # episodes per genome during training — 5 was too noisy (lucky winners fail 100-ep eval)
SOLVE_EPISODES = 100  # episodes for the final solve assertion
SOLVE_THRESHOLD = 475.0

_CFG_PATH = os.path.join(os.path.dirname(__file__), "configs", "cartpole.cfg")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cartpole_solves():
    """NEAT solves CartPole-v1 ≥ 475 mean reward within budget (seeded).

    RED (F0.3): fails with ModuleNotFoundError on both imports below —
                neat3p.nn.phenotypes (Phase 1) and neat3p.benchmarks.runners
                (Phase 2) do not exist yet.
    GREEN (F2.2): passes after Phase 1 migration + Phase 2 harness.
    """
    # RED: ModuleNotFoundError — migrated in Phase 1
    # RED: ModuleNotFoundError — built in Phase 2
    from neat3p.benchmarks.runners.gym_eval import run_neat_gym
    from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

    result = run_neat_gym(
        env_id="CartPole-v1",
        config_path=_CFG_PATH,
        max_generations=MAX_GENERATIONS,
        episodes_per_genome=EVAL_EPISODES,
        seed=SEED,
        net_class=RecurrentNet,
    )

    mean_reward = result.evaluate(n_episodes=SOLVE_EPISODES, seed=SEED + 1)
    assert mean_reward >= SOLVE_THRESHOLD, f"CartPole not solved: mean_reward={mean_reward:.1f} < {SOLVE_THRESHOLD}"
