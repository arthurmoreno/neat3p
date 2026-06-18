"""
CartPole-v1 benchmark with NEATNetWithFeatureAttention — Phase 1 (frozen encoder).

Gate: 🔴 → 🟢

Phase 1: encoder + attention initialised randomly and kept frozen; NEAT evolves
the RecurrentNet controller through the fixed front-end. This validates the full
  obs → SimpleEncoder → FeatureAttention → RecurrentNet → action
pipeline end-to-end before any encoder training is introduced.

feature_dim=4: no compression (4→4) so NEAT gets a learned linear projection of
the full CartPole state. Compressing 4→1 (the default //10 ratio) would lose
cart velocity and pole angular velocity with random weights, making the task
unsolvable in Phase 1.
"""

import os

import pytest

pytestmark = pytest.mark.skip(reason="too slow — skipped until attention benchmark is prioritised")

SEED = 42
MAX_GENERATIONS = 300
EVAL_EPISODES = 10
SOLVE_EPISODES = 100
SOLVE_THRESHOLD = 475.0

_CFG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../benchmarks/configs/cartpole.cfg"))


@pytest.mark.slow
def test_cartpole_feature_attention_solves():
    """NEATNetWithFeatureAttention (frozen encoder, feature_dim=4) solves CartPole-v1.

    Phase 1: encoder + attention are random and frozen; only the NEAT genome evolves.
    GREEN when the full encoder→attention→RecurrentNet pipeline works and NEAT can
    adapt through the fixed front-end to achieve ≥ 475 mean reward over 100 episodes.
    """
    from benchmarks.runners.gym_eval import run_neat_gym
    from neat3p.nn.composite import NEATNetWithFeatureAttention

    result = run_neat_gym(
        env_id="CartPole-v1",
        config_path=_CFG_PATH,
        max_generations=MAX_GENERATIONS,
        episodes_per_genome=EVAL_EPISODES,
        seed=SEED,
        net_class=NEATNetWithFeatureAttention,
        net_kwargs={"feature_dim": 4},
    )

    mean_reward = result.evaluate(n_episodes=SOLVE_EPISODES, seed=SEED + 1)
    assert mean_reward >= SOLVE_THRESHOLD, (
        f"CartPole not solved with NEATNetWithFeatureAttention: mean_reward={mean_reward:.1f} < {SOLVE_THRESHOLD}"
    )
