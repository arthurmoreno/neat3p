#!/usr/bin/env python3
"""
VoxelForage gate — plain HyperNEAT (fixed-weight CPPN substrate net).

A CPPN paints a fixed weight matrix over the voxel substrate (the 83-input observation laid
out in 2-D, see neat3p.benchmarks.substrates) -> hidden -> 7 actions. Uses the shipping
``HyperNEATNet`` unchanged; ``VoxelForageHyperNEATNet`` only bakes the substrate coordinates.

Usage:
    python benchmarks/voxel_forage_hyper_neat.py
    python benchmarks/voxel_forage_hyper_neat.py --no-scent
    python benchmarks/voxel_forage_hyper_neat.py --render
"""

import os

import gymnasium as gym
import numpy as np
import torch

import neat3p.benchmarks.envs  # noqa: F401  — registers VoxelForage-v0
from neat3p.benchmarks.artifacts import save_winner
from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.benchmarks.substrates import voxel_forage_substrate
from neat3p.nn.composite import HyperNEATNet

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/voxel_forage_hyper_neat.cfg"))
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

BENCHMARK_NAME = "voxel_forage_hyper_neat"
ENV_ID = "VoxelForage-v0"
SOLVE_THRESHOLD = 110.0

_INPUT, _HIDDEN, _OUTPUT = voxel_forage_substrate()
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class VoxelForageHyperNEATNet:
    """Bakes the voxel substrate into the shipping HyperNEATNet (recurrent-style interface)."""

    def __init__(self, net: HyperNEATNet) -> None:
        self._net = net

    @classmethod
    def create(cls, genome, config, batch_size: int = 1, use_current_activs: bool = True, device: str = _DEVICE):
        net = HyperNEATNet.create(genome, config, _INPUT, _HIDDEN, _OUTPUT, batch_size=batch_size, device=device)
        return cls(net)

    def activate(self, inputs):
        return self._net.activate(inputs)

    def reset(self, batch_size=None):
        self._net.reset(batch_size)


def run_benchmark(
    seed: int,
    generations: int = 60,
    episodes_per_genome: int = 2,
    eval_episodes: int = 20,
    verbose: bool = True,
    env_id: str = ENV_ID,
    save_dir: str = None,
) -> dict:
    result = run_neat_gym(
        env_id=env_id, config_path=_CFG, max_generations=generations,
        episodes_per_genome=episodes_per_genome, seed=seed,
        net_class=VoxelForageHyperNEATNet, verbose=verbose,
    )
    rewards = result.evaluate_rewards(n_episodes=eval_episodes, seed=seed + 1)
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)
    winner_path = save_winner(save_dir or _OUTPUT_DIR, "hyper_neat", result, env_id, seed, _CFG)
    if verbose:
        print(f"  saved winner -> {winner_path}")
    return {
        "benchmark_name": BENCHMARK_NAME,
        "env_id": env_id,
        "winner_path": winner_path,
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


def _watch(winner, config, episodes, env_id):
    env = gym.make(env_id, render_mode="human")
    net = VoxelForageHyperNEATNet.create(winner, config)
    for ep in range(episodes):
        obs, _ = env.reset()
        net.reset(batch_size=1)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            out = net.activate([obs.tolist()])
            action = int(out[0].argmax().item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        print(f"  Episode {ep + 1}: collected={total:.0f} steps={info['steps']}")
    env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--episodes-per-genome", type=int, default=2)
    parser.add_argument("--no-scent", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env_id = "VoxelForage-NoScent-v0" if args.no_scent else ENV_ID
    print(f"Training {env_id} with plain HyperNEAT (seed={args.seed}, gens={args.generations})...")

    result = run_neat_gym(
        env_id=env_id, config_path=_CFG, max_generations=args.generations,
        episodes_per_genome=args.episodes_per_genome, seed=args.seed,
        net_class=VoxelForageHyperNEATNet, verbose=True,
    )
    rewards = result.evaluate_rewards(n_episodes=20, seed=args.seed + 1)
    print(f"\nWinner fitness (food): {result.winner.fitness:.1f} / 120")
    print(f"Mean collected (20 ep): {float(np.mean(rewards)):.1f} ± {float(np.std(rewards)):.1f}")
    print(f"Wall time            : {result.wall_time_seconds:.1f}s")
    print(f"Winner nodes/conns   : {result.winner_nodes} / {result.winner_connections}")

    if args.render:
        print(f"\nRendering {args.episodes} episode(s)...")
        _watch(result.winner, result.config, args.episodes, env_id)


if __name__ == "__main__":
    main()
