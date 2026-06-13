#!/usr/bin/env python3
"""
CartPole-v1 NEAT benchmark — train then watch.

Usage:
    python benchmarks/cartpole.py             # train + render winner
    python benchmarks/cartpole.py --episodes 3  # watch for 3 episodes
    python benchmarks/cartpole.py --no-render   # train only (headless)
"""

import argparse
import os

import gymnasium as gym

from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

_CFG = os.path.join(os.path.dirname(__file__), "../tests/benchmarks/configs/cartpole.cfg")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--generations", type=int, default=300)
parser.add_argument("--episodes-per-genome", type=int, default=5)
parser.add_argument("--episodes", type=int, default=3, help="Episodes to render after training")
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()

print(f"Training on CartPole-v1 (seed={args.seed}, max_generations={args.generations})...")

result = run_neat_gym(
    env_id="CartPole-v1",
    config_path=os.path.abspath(_CFG),
    max_generations=args.generations,
    episodes_per_genome=args.episodes_per_genome,
    seed=args.seed,
    net_class=RecurrentNet,
)

mean = result.evaluate(n_episodes=100, seed=args.seed + 1)
print(f"\nWinner fitness: {result.winner.fitness:.1f}")
print(f"Mean reward over 100 episodes: {mean:.1f}")

if not args.no_render:
    print(f"\nRendering {args.episodes} episode(s) — close the window between runs...")
    env = gym.make("CartPole-v1", render_mode="human")
    net = RecurrentNet.create(result.winner, result.config, batch_size=1, use_current_activs=True)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        net.reset(batch_size=1)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            out = net.activate([obs.tolist()])
            action = int(out[0].argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
        print(f"  Episode {ep + 1}: reward={total:.0f}")
    env.close()
