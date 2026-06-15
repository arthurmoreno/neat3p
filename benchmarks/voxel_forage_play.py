#!/usr/bin/env python3
"""
Play VoxelForage yourself (keyboard). Turn-based: one key press = one env step.

Controls:
    Arrow keys / WASD : move in the x-y plane (W/Up = -y, S/Down = +y, A/Left = -x, D/Right = +x)
    Q / PageUp        : move up   (+z)
    E / PageDown      : move down (-z)
    Space             : idle (wait one step — energy still decays!)
    R                 : reset a fresh world
    Esc / window close: quit

Goal: collect food (green) to refill energy before it hits 0. Avoid hazards (red), walls
are gray, the agent is the blue dot. In scent mode the background brightens toward food.
The HUD (top-left) shows the z-plane you're on, energy, food collected, and step count.

Usage:
    python benchmarks/voxel_forage_play.py
    python benchmarks/voxel_forage_play.py --no-scent      # food only visible in the patch
    python benchmarks/voxel_forage_play.py --seed 7
"""

import argparse

import pygame

import gymnasium as gym
import neat3p.benchmarks.envs  # noqa: F401  — registers VoxelForage-v0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-scent", action="store_true")
    args = parser.parse_args()

    env_id = "VoxelForage-NoScent-v0" if args.no_scent else "VoxelForage-v0"
    env = gym.make(env_id, render_mode="human")

    keymap = {
        pygame.K_RIGHT: 1, pygame.K_d: 1,   # +x
        pygame.K_LEFT: 2, pygame.K_a: 2,    # -x
        pygame.K_DOWN: 3, pygame.K_s: 3,    # +y (down on screen)
        pygame.K_UP: 4, pygame.K_w: 4,      # -y (up on screen)
        pygame.K_q: 5, pygame.K_PAGEUP: 5,  # +z (up a layer)
        pygame.K_e: 6, pygame.K_PAGEDOWN: 6,  # -z (down a layer)
        pygame.K_SPACE: 0,                  # idle
    }

    obs, info = env.reset(seed=args.seed)
    print(__doc__)
    print(f"Playing {env_id}.  Total food available = {env.unwrapped.total_food_energy:.0f}.")
    print("Move with arrows/WASD, Q/E for up/down, Space to wait, R to reset, Esc to quit.\n")

    total = 0.0
    done = False
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            break
        if event.type != pygame.KEYDOWN:
            continue
        if event.key == pygame.K_ESCAPE:
            break
        if event.key == pygame.K_r:
            obs, info = env.reset()
            total = 0.0
            done = False
            print("--- new world ---")
            continue
        if done or event.key not in keymap:
            continue

        obs, reward, terminated, truncated, info = env.step(keymap[event.key])
        total += float(reward)
        if reward > 0:
            print(f"  +{reward:.0f} food!  total={total:.0f}  energy={info['energy']:.0f}")
        if terminated or truncated:
            done = True
            cause = "starved" if info["energy"] <= 0 else ("all food eaten" if truncated is False else "time up")
            print(f"\nEpisode over ({cause}): collected={total:.0f}/{env.unwrapped.total_food_energy:.0f} "
                  f"in {info['steps']} steps.  Press R to play again, Esc to quit.\n")

    env.close()


if __name__ == "__main__":
    main()
