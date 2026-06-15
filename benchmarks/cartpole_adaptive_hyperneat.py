#!/usr/bin/env python3
"""
CartPole-v1 Adaptive HyperNEAT benchmark.

The CPPN genome maps substrate geometry coordinates to network weights and
Hebbian learning rates; AdaptiveNet applies per-step delta_w updates.

Substrate (CartPole):
  inputs  — 4 nodes on a horizontal line at y= 1.0
  hidden  — 8 nodes on a horizontal line at y= 0.0
  outputs — 2 nodes (left/right action) at y=-1.0

Usage:
    python benchmarks/cartpole_adaptive_hyperneat.py
    python benchmarks/cartpole_adaptive_hyperneat.py --seed 1 --generations 500
    python benchmarks/cartpole_adaptive_hyperneat.py --no-render
"""

import os

import gymnasium as gym
import numpy as np

from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.nn.composite import AdaptiveNet

_CFG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "configs/cartpole_adaptive_hyperneat.cfg")
)

BENCHMARK_NAME = "cartpole_adaptive_hyperneat"
SOLVE_THRESHOLD = 475.0

# Fixed substrate geometry for CartPole-v1
# 4 observation dims spread evenly across x=[-1, 1] at y=1
_INPUT_COORDS = [(-1.0, 1.0), (-0.33, 1.0), (0.33, 1.0), (1.0, 1.0)]
# 8 hidden nodes spread across x=[-1, 1] at y=0
_HIDDEN_COORDS = [(-1.0 + 2.0 * i / 7, 0.0) for i in range(8)]
# 2 output nodes (left=0, right=1) at y=-1
_OUTPUT_COORDS = [(-0.5, -1.0), (0.5, -1.0)]


class CartPoleAdaptiveNet:
    """Thin wrapper adapting AdaptiveNet to the gym_eval recurrent-style interface.

    gym_eval calls:
        net = CartPoleAdaptiveNet.create(genome, config, batch_size=1, use_current_activs=True)
        net.reset(batch_size=1)          # batch_size arg is ignored
        out = net.activate([obs_list])   # returns tensor (1, 2)
    """

    def __init__(self, net: AdaptiveNet) -> None:
        self._net = net

    @classmethod
    def create(
        cls,
        genome,
        config,
        batch_size: int = 1,
        use_current_activs: bool = True,
    ) -> "CartPoleAdaptiveNet":
        net = AdaptiveNet.create(
            genome,
            config,
            input_coords=_INPUT_COORDS,
            hidden_coords=_HIDDEN_COORDS,
            output_coords=_OUTPUT_COORDS,
            batch_size=batch_size,
        )
        return cls(net)

    def activate(self, inputs):
        return self._net.activate(inputs)

    def reset(self, batch_size=None):
        self._net.reset()


def run_benchmark(
    seed: int,
    generations: int = 500,
    episodes_per_genome: int = 5,
    eval_episodes: int = 100,
    verbose: bool = True,
) -> dict:
    """Run one CartPole Adaptive HyperNEAT trial. Returns a stats dict."""
    result = run_neat_gym(
        env_id="CartPole-v1",
        config_path=_CFG,
        max_generations=generations,
        episodes_per_genome=episodes_per_genome,
        seed=seed,
        net_class=CartPoleAdaptiveNet,
        verbose=verbose,
    )

    rewards = result.evaluate_rewards(n_episodes=eval_episodes, seed=seed + 1)
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)

    return {
        "benchmark_name": BENCHMARK_NAME,
        "seed": seed,
        "solve_generation": solve_gen,
        "total_generations": len(gen_stats),
        "winner_fitness": float(result.winner.fitness),
        "final_mean_reward": float(np.mean(rewards)),
        "final_std_reward": float(np.std(rewards)),
        "wall_time_seconds": result.wall_time_seconds,
        "training_rss_mb": result.training_rss_mb,
        "peak_gpu_mb": result.peak_gpu_mb,
        "winner_nodes": result.winner_nodes,
        "winner_connections": result.winner_connections,
        "generation_stats": gen_stats,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--episodes-per-genome", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to render after training")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    print(
        f"Training CartPole-v1 with Adaptive HyperNEAT "
        f"(seed={args.seed}, max_generations={args.generations})..."
    )

    result = run_neat_gym(
        env_id="CartPole-v1",
        config_path=_CFG,
        max_generations=args.generations,
        episodes_per_genome=args.episodes_per_genome,
        seed=args.seed,
        net_class=CartPoleAdaptiveNet,
        verbose=True,
    )
    rewards = result.evaluate_rewards(n_episodes=100, seed=args.seed + 1)
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)

    print(f"\nWinner fitness       : {result.winner.fitness:.1f}")
    print(f"Mean reward (100 ep) : {float(np.mean(rewards)):.1f} ± {float(np.std(rewards)):.1f}")
    print(f"Solve generation     : {solve_gen}")
    print(f"Wall time            : {result.wall_time_seconds:.1f}s")
    print(f"Winner nodes/conns   : {result.winner_nodes} / {result.winner_connections}")

    if not args.no_render:
        print(f"\nRendering {args.episodes} episode(s) — close the window between runs...")
        env = gym.make("CartPole-v1", render_mode="human")
        net = CartPoleAdaptiveNet.create(result.winner, result.config, batch_size=1)
        for ep in range(args.episodes):
            obs, _ = env.reset()
            net.reset()
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                out = net.activate([obs.tolist()])
                action = int(out[0].argmax().item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
            print(f"  Episode {ep + 1}: reward={total:.0f}")
        env.close()


if __name__ == "__main__":
    main()
