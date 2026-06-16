#!/usr/bin/env python3
"""
VoxelForage gate — plain NEAT RecurrentNet controller.

VoxelForage is the central neat3p benchmark: a 3D voxel + scalar-attributes -> discrete
action task that mirrors the game's contract, so every network is scored on the same
fixed-shape problem and the numbers are comparable. This file runs the shipping
``NEATRecurrentNet`` brain on it (gym_eval nn.Module path) — the first net on the gate.

Usage:
    python benchmarks/voxel_forage_recurrent_net.py                 # train (scent, headless)
    python benchmarks/voxel_forage_recurrent_net.py --render        # train + watch winner
    python benchmarks/voxel_forage_recurrent_net.py --no-scent      # sparse mode (harder)
    python benchmarks/voxel_forage_recurrent_net.py --generations 80
"""

import os

import gymnasium as gym
import numpy as np
import torch

import neat3p.benchmarks.envs  # noqa: F401  — registers VoxelForage-v0
from neat3p.benchmarks.artifacts import save_winner
from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.nn.composite import NEATRecurrentNet

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/voxel_forage.cfg"))
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

BENCHMARK_NAME = "voxel_forage_recurrent_net"
ENV_ID = "VoxelForage-v0"
SOLVE_THRESHOLD = 110.0  # food energy collected (total available = 120)


def run_benchmark(
    seed: int,
    generations: int = 120,
    episodes_per_genome: int = 5,
    eval_episodes: int = 50,
    verbose: bool = True,
    env_id: str = ENV_ID,
    save_dir: str = None,
    progress: bool = False,
    progress_desc: str = "",
    progress_position: int = 0,
    eval_strategy: str = "per_generation",
    validation_episodes: int = 0,
) -> dict:
    """Run one VoxelForage NEAT trial. Returns a suite-compatible stats dict."""
    result = run_neat_gym(
        env_id=env_id,
        config_path=_CFG,
        max_generations=generations,
        episodes_per_genome=episodes_per_genome,
        seed=seed,
        net_class=NEATRecurrentNet,
        verbose=verbose,
        progress=progress,
        progress_desc=progress_desc,
        progress_position=progress_position,
        eval_strategy=eval_strategy,
        validation_episodes=validation_episodes,
    )

    rewards = result.evaluate_rewards(n_episodes=eval_episodes, seed=seed + 1)
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)

    winner_path = save_winner(save_dir or _OUTPUT_DIR, "recurrent_net", result, env_id, seed, _CFG)
    if verbose:
        print(f"  saved winner -> {winner_path}")

    return {
        "benchmark_name": BENCHMARK_NAME,
        "env_id": env_id,
        "winner_path": winner_path,
        "validation_stats": result.validation_stats,
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


def _watch(winner, config, episodes: int, env_id: str = ENV_ID):
    env = gym.make(env_id, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    net = NEATRecurrentNet(winner, state_dim, action_dim, config)
    for ep in range(episodes):
        obs, _ = env.reset()
        net.reset(batch_size=1)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = net(obs_t)
            action = int(out.argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        print(f"  Episode {ep + 1}: collected={total:.0f}  steps={info['steps']}")
    env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--episodes-per-genome", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to render after training")
    parser.add_argument("--render", action="store_true", help="Watch the winner in a pygame window")
    parser.add_argument("--no-scent", action="store_true", help="Sparse mode: food visible only inside the patch")
    args = parser.parse_args()

    env_id = "VoxelForage-NoScent-v0" if args.no_scent else ENV_ID

    print(f"Training on {env_id} with NEATRecurrentNet (seed={args.seed}, max_generations={args.generations})...")

    result = run_neat_gym(
        env_id=env_id,
        config_path=_CFG,
        max_generations=args.generations,
        episodes_per_genome=args.episodes_per_genome,
        seed=args.seed,
        net_class=NEATRecurrentNet,
        verbose=True,
    )
    rewards = result.evaluate_rewards(n_episodes=50, seed=args.seed + 1)
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)

    print(f"\nWinner fitness (food): {result.winner.fitness:.1f} / 120")
    print(f"Mean collected (50 ep): {float(np.mean(rewards)):.1f} ± {float(np.std(rewards)):.1f}")
    print(f"Solve generation     : {solve_gen}")
    print(f"Wall time            : {result.wall_time_seconds:.1f}s")
    print(f"Winner nodes/conns   : {result.winner_nodes} / {result.winner_connections}")

    if args.render:
        print(f"\nRendering {args.episodes} episode(s) — close the window to continue...")
        _watch(result.winner, result.config, args.episodes, env_id=env_id)


if __name__ == "__main__":
    main()
