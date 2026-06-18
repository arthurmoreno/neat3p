"""
CartPole-v1 benchmark — plain NEAT RecurrentNet controller (NEATRecurrentNet).

Solve criterion: mean episode reward ≥ 475 over SOLVE_EPISODES rollouts,
achieved within MAX_GENERATIONS NEAT generations.

What this validates:
  - neat3p.nn.composite.NEATRecurrentNet (the shipping game-contract brain net)
    drives a CartPole agent — i.e. we test the class the game would plug in, not the
    raw RecurrentNet phenotype.
  - benchmarks.runners.gym_eval.run_neat_gym provides the eval harness.
  - NEAT (memoryless, feed_forward=True) reliably solves CartPole-v1.
  - Run is seeded so CI is reproducible.
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

_CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../benchmarks/configs/cartpole.cfg"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cartpole_recurrent_net_solves():
    """NEATRecurrentNet solves CartPole-v1 ≥ 475 mean reward within budget (seeded)."""
    from benchmarks.runners.gym_eval import run_neat_gym
    from neat3p.nn.composite import NEATRecurrentNet

    result = run_neat_gym(
        env_id="CartPole-v1",
        config_path=_CFG_PATH,
        max_generations=MAX_GENERATIONS,
        episodes_per_genome=EVAL_EPISODES,
        seed=SEED,
        net_class=NEATRecurrentNet,
    )

    mean_reward = result.evaluate(n_episodes=SOLVE_EPISODES, seed=SEED + 1)
    assert mean_reward >= SOLVE_THRESHOLD, f"CartPole not solved: mean_reward={mean_reward:.1f} < {SOLVE_THRESHOLD}"
