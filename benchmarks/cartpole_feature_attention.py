#!/usr/bin/env python3
"""
CartPole-v1 benchmark with NEATNetWithFeatureAttention — train then watch.

Phase 1 (frozen encoder): encoder + attention are random and frozen;
NEAT evolves the RecurrentNet controller through the fixed front-end.

Usage:
    python benchmarks/cartpole_feature_attention.py
    python benchmarks/cartpole_feature_attention.py --episodes 5
    python benchmarks/cartpole_feature_attention.py --no-render
"""

import argparse
import os

import gymnasium as gym
import torch

from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.nn.composite import NEATNetWithFeatureAttention

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tests/benchmarks/configs/cartpole.cfg"))

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--generations", type=int, default=300)
parser.add_argument("--episodes-per-genome", type=int, default=10)
parser.add_argument(
    "--feature-dim", type=int, default=4, help="Encoder output dim. Default 4 = no compression (full CartPole state)."
)
parser.add_argument("--episodes", type=int, default=3, help="Episodes to render after training")
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()

print("Training on CartPole-v1 with NEATNetWithFeatureAttention")
print(f"  seed={args.seed}  max_generations={args.generations}  feature_dim={args.feature_dim}")
print(f"  pipeline: obs(4) → SimpleEncoder({args.feature_dim}) → FeatureAttention → RecurrentNet → action")
print()

result = run_neat_gym(
    env_id="CartPole-v1",
    config_path=_CFG,
    max_generations=args.generations,
    episodes_per_genome=args.episodes_per_genome,
    seed=args.seed,
    net_class=NEATNetWithFeatureAttention,
    net_kwargs={"feature_dim": args.feature_dim},
)

mean = result.evaluate(n_episodes=100, seed=args.seed + 1)
print(f"\nWinner fitness : {result.winner.fitness:.1f}")
print(f"Mean reward (100 ep): {mean:.1f}")

if not args.no_render:
    print(f"\nRendering {args.episodes} episode(s)...")
    env = gym.make("CartPole-v1", render_mode="human")
    net = NEATNetWithFeatureAttention(
        result.winner,
        result.state_dim,
        result.action_dim,
        result.config,
        feature_dim=args.feature_dim,
    )
    for ep in range(args.episodes):
        obs, _ = env.reset()
        net.reset(batch_size=1)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = net(obs_t)
            action = int(out.argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
        print(f"  Episode {ep + 1}: reward={total:.0f}")
    env.close()
