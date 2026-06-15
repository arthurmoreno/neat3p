#!/usr/bin/env python3
"""
Replay a saved VoxelForage winner — load any trained network and watch it forage.

Winners are saved automatically by every benchmark / suite run into benchmarks/output/
(e.g. feature_attention_scent_seed42.pkl). Point this at one to watch it perform.

Usage:
    python benchmarks/voxel_forage_replay.py                                   # list saved winners
    python benchmarks/voxel_forage_replay.py benchmarks/output/feature_attention_scent_seed42.pkl
    python benchmarks/voxel_forage_replay.py <winner.pkl> --episodes 5
    python benchmarks/voxel_forage_replay.py <winner.pkl> --no-render          # numbers only, no window
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch

import neat3p.benchmarks.envs  # noqa: F401  — registers VoxelForage-v0
from neat3p.benchmarks.artifacts import list_winners, load_winner, reset_net, select_action

_OUTPUT = os.path.join(os.path.dirname(__file__), "output")
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("winner", nargs="?", default=None, help="Path to a saved winner .pkl")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--env", default=None, help="Override the env id (default: the one it trained on)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed world seed (default: random each episode)")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if args.winner is None:
        saved = list_winners(_OUTPUT)
        if not saved:
            print(f"No saved winners in {_OUTPUT}. Run a benchmark or the suite first.")
            return
        print("Saved winners (pass one as an argument):")
        for p in saved:
            print(f"  {p}")
        return

    pkg, net, style = load_winner(args.winner, device=_DEVICE)
    env_id = args.env or pkg["env_id"]
    print(
        f"Replaying {pkg['kind']}  (trained on {pkg['env_id']}, seed={pkg['seed']}, "
        f"train_fitness={pkg['fitness']:.1f}/120)  →  env={env_id}"
    )

    render_mode = None if args.no_render else "human"
    env = gym.make(env_id, render_mode=render_mode)
    total_food = env.unwrapped.total_food_energy

    collected = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        reset_net(net, style)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = select_action(net, obs, style)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        win = "🏆 ALL FOOD" if total >= total_food else ("starved" if info["energy"] <= 0 else "time up")
        collected.append(total)
        print(f"  Episode {ep + 1}: collected={total:.0f}/{total_food:.0f}  steps={info['steps']}  ({win})")
    env.close()
    print(f"\nMean collected: {np.mean(collected):.1f} ± {np.std(collected):.1f} over {args.episodes} episodes")


if __name__ == "__main__":
    main()
