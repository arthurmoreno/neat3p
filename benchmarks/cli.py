"""
Unified CLI for neat3p benchmarks.

Subcommands:
    train   — train one (task × model) combination
    suite   — run all models on a task and build a comparison report
    replay  — watch a saved winner .pkl perform
    play    — play VoxelForage yourself (keyboard)

Usage::

    python -m benchmarks train  --task voxel_forage --model recurrent_net --seed 42
    python -m benchmarks suite  --task voxel_forage --runs 2 --generations 15
    python -m benchmarks replay benchmarks/output/recurrent_net_scent_seed42.pkl
    python -m benchmarks play   --task voxel_forage [--no-scent]
"""

from __future__ import annotations

import argparse
import os
import time

from benchmarks.models import MODELS
from benchmarks.tasks import TASKS

_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "output")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> None:
    from benchmarks.runner import run_benchmark

    result = run_benchmark(
        args.task,
        args.model,
        seed=args.seed,
        generations=args.generations,
        episodes_per_genome=args.episodes_per_genome,
        eval_episodes=args.eval_episodes,
        variant=args.variant,
        save_dir=args.save_dir or _DEFAULT_OUTPUT,
        verbose=not args.quiet,
        eval_strategy=args.eval_strategy,
        validation_episodes=args.validation_episodes,
        pretrain_episodes=args.pretrain_episodes,
        pretrain_epochs=args.pretrain_epochs,
    )

    task = TASKS[args.task]
    label = task.reward_label
    print(f"\n{'─' * 56}")
    print(
        f"  Winner fitness ({label}): {result['winner_fitness']:.1f}"
        + (f" / {task.reward_max:.0f}" if task.reward_max else "")
    )
    print(
        f"  Mean {label} ({result['total_generations']} gens) : "
        f"{result['final_mean_reward']:.1f} ± {result['final_std_reward']:.1f}"
    )
    print(f"  Solve generation : {result['solve_generation']}")
    print(f"  Wall time        : {result['wall_time_seconds']:.1f}s")
    print(f"  Winner nodes/conns: {result['winner_nodes']} / {result['winner_connections']}")
    if args.render:
        _replay_winner(result["winner_path"], episodes=args.episodes, render=True)


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------


def _cmd_suite(args: argparse.Namespace) -> None:
    from tqdm.auto import tqdm as _tqdm

    from benchmarks.report import build_report
    from benchmarks.runner import run_benchmark

    task_name = args.task
    models = args.models or list(MODELS.keys())
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(42, 42 + args.runs))
    output = args.output
    if not os.path.isabs(output):
        output = os.path.join(_DEFAULT_OUTPUT, output)

    print("=" * 64)
    print(f"neat3p suite — task={task_name}  models={models}  seeds={seeds}")
    print(f"  budget: {args.generations} gens × {args.episodes_per_genome} eps/genome")
    print(f"  output: {output}")
    print("=" * 64)

    all_results: list[dict] = []
    total_runs = len(models) * len(seeds)
    outer = _tqdm(total=total_runs, desc="Suite", position=0, leave=True)
    t_suite = time.perf_counter()

    for model_name in models:
        outer.write(f"── {model_name} ──")
        for seed in seeds:
            t0 = time.perf_counter()
            result = run_benchmark(
                task_name,
                model_name,
                seed=seed,
                generations=args.generations,
                episodes_per_genome=args.episodes_per_genome,
                eval_episodes=args.eval_episodes,
                variant=args.variant,
                save_dir=_DEFAULT_OUTPUT,
                verbose=False,
                progress=True,
                progress_desc=f"{model_name} s{seed}",
                progress_position=1,
                eval_strategy=args.eval_strategy,
                validation_episodes=args.validation_episodes,
                pretrain_episodes=args.pretrain_episodes,
                pretrain_epochs=args.pretrain_epochs,
            )
            elapsed = time.perf_counter() - t0
            solved = result["solve_generation"] is not None
            outer.write(
                f"  ✓ {model_name} s{seed}  {'SOLVED' if solved else 'done'} "
                f"| best={result['winner_fitness']:.0f} "
                f"| eval={result['final_mean_reward']:.1f} "
                f"| {elapsed:.0f}s"
            )
            outer.update(1)
            all_results.append(result)

    outer.close()
    total = time.perf_counter() - t_suite
    print(f"\n{'=' * 64}\nDone in {total:.0f}s. Building report...")
    build_report(all_results, list(MODELS.keys()), output, format=args.format)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def _resolve_env_id(env: str | None, variant: str, pkg_env_id: str) -> str:
    """Resolve the env to replay in.

    --env may be a task key in TASKS (resolved through its variants/default env_id)
    or a literal gym env id. When omitted, fall back to the env baked into the pkl.
    """
    if env is None:
        return pkg_env_id
    task = TASKS.get(env)
    if task is None:
        return env  # treat as a literal gym env id
    return task.variants.get(variant, task.env_id) if task.variants else task.env_id


def _replay_winner(
    winner_path: str, episodes: int = 3, render: bool = True, env: str | None = None, variant: str = "scent"
) -> None:
    import gymnasium as gym
    import numpy as np
    import torch

    import neat3p.benchmarks.envs  # noqa: F401
    from neat3p.benchmarks.artifacts import load_winner, reset_net, select_action

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pkg, net, style = load_winner(winner_path, device=device)
    env_id = _resolve_env_id(env, variant, pkg["env_id"])
    override = " (override)" if env_id != pkg["env_id"] else ""
    print(
        f"Replaying {pkg['kind']}  env={env_id}{override}  seed={pkg['seed']}  train_fitness={pkg['fitness']:.1f}"
    )

    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    total_food = getattr(getattr(env, "unwrapped", env), "total_food_energy", None)

    collected = []
    for ep in range(episodes):
        obs, _ = env.reset()
        reset_net(net, style)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = select_action(net, obs, style)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        collected.append(total)
        suffix = f"/{total_food:.0f}" if total_food else ""
        print(f"  Episode {ep + 1}: collected={total:.0f}{suffix}  steps={info.get('steps', '?')}")
    env.close()
    print(f"\nMean: {np.mean(collected):.1f} ± {np.std(collected):.1f} over {episodes} episodes")


def _cmd_replay(args: argparse.Namespace) -> None:
    from neat3p.benchmarks.artifacts import list_winners

    if args.winner is None:
        saved = list_winners(_DEFAULT_OUTPUT)
        if not saved:
            print(f"No saved winners in {_DEFAULT_OUTPUT}. Run 'train' or 'suite' first.")
            return
        print("Saved winners:")
        for p in saved:
            print(f"  {p}")
        return
    _replay_winner(
        args.winner, episodes=args.episodes, render=not args.no_render, env=args.env, variant=args.variant
    )


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------


def _cmd_play(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import pygame

    import neat3p.benchmarks.envs  # noqa: F401

    task_name = args.task
    if task_name != "voxel_forage":
        print(f"play only supports voxel_forage; got '{task_name}'.")
        return

    env_id = "VoxelForage-NoScent-v0" if args.no_scent else "VoxelForage-v0"
    env = gym.make(env_id, render_mode="human")

    keymap = {
        pygame.K_RIGHT: 1,
        pygame.K_d: 1,
        pygame.K_LEFT: 2,
        pygame.K_a: 2,
        pygame.K_DOWN: 3,
        pygame.K_s: 3,
        pygame.K_UP: 4,
        pygame.K_w: 4,
        pygame.K_q: 5,
        pygame.K_PAGEUP: 5,
        pygame.K_e: 6,
        pygame.K_PAGEDOWN: 6,
        pygame.K_SPACE: 0,
    }
    obs, _ = env.reset(seed=args.seed)
    print(f"Playing {env_id}. Total food = {env.unwrapped.total_food_energy:.0f}.")
    print("Arrows/WASD to move, Q/E for up/down, Space to idle, R to reset, Esc to quit.\n")

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
            obs, _ = env.reset()
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
            cause = "starved" if info["energy"] <= 0 else ("all food eaten" if not truncated else "time up")
            print(
                f"\nEpisode over ({cause}): collected={total:.0f}/{env.unwrapped.total_food_energy:.0f} "
                f"in {info['steps']} steps. Press R to play again, Esc to quit.\n"
            )
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="neat3p benchmark CLI — train, suite, replay, play.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    p_train = sub.add_parser("train", help="Train one (task × model) combination.")
    p_train.add_argument("--task", required=True, choices=list(TASKS.keys()))
    p_train.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--generations", type=int, default=None)
    p_train.add_argument("--episodes-per-genome", type=int, default=4)
    p_train.add_argument("--eval-episodes", type=int, default=50)
    p_train.add_argument(
        "--variant", default="scent", help="Task variant key, e.g. 'scent'/'noscent' for voxel_forage."
    )
    p_train.add_argument("--save-dir", default=None)
    p_train.add_argument("--quiet", action="store_true")
    p_train.add_argument("--render", action="store_true", help="Watch winner after training.")
    p_train.add_argument("--episodes", type=int, default=3)
    p_train.add_argument("--eval-strategy", choices=["per_generation", "fixed", "random"], default="per_generation")
    p_train.add_argument("--validation-episodes", type=int, default=0)
    p_train.add_argument("--pretrain-episodes", type=int, default=250)
    p_train.add_argument("--pretrain-epochs", type=int, default=100)

    # ── suite ──
    p_suite = sub.add_parser("suite", help="Run all models on a task and build a report.")
    p_suite.add_argument("--task", required=True, choices=list(TASKS.keys()))
    p_suite.add_argument("--models", nargs="+", default=None, choices=list(MODELS.keys()))
    p_suite.add_argument("--runs", type=int, default=2)
    p_suite.add_argument("--seeds", default=None, help="Comma-separated explicit seeds.")
    p_suite.add_argument("--generations", type=int, default=15)
    p_suite.add_argument("--episodes-per-genome", type=int, default=2)
    p_suite.add_argument("--eval-episodes", type=int, default=20)
    p_suite.add_argument("--variant", default="scent")
    p_suite.add_argument("--eval-strategy", choices=["per_generation", "fixed", "random"], default="per_generation")
    p_suite.add_argument("--validation-episodes", type=int, default=10)
    p_suite.add_argument("--pretrain-episodes", type=int, default=250)
    p_suite.add_argument("--pretrain-epochs", type=int, default=100)
    p_suite.add_argument(
        "--output",
        default="suite_report.html",
        help="Output filename (resolved under benchmarks/output/ if not absolute).",
    )
    p_suite.add_argument("--format", choices=["html", "md", "both"], default="html")

    # ── replay ──
    p_replay = sub.add_parser("replay", help="Watch a saved winner .pkl perform.")
    p_replay.add_argument("winner", nargs="?", default=None, help="Path to saved winner .pkl")
    p_replay.add_argument("--episodes", type=int, default=3)
    p_replay.add_argument("--no-render", action="store_true")
    p_replay.add_argument(
        "--env",
        default=None,
        help="Override the replay env: a task key (e.g. 'voxel_forage') or a literal gym env id. "
        "Defaults to the env baked into the .pkl.",
    )
    p_replay.add_argument(
        "--variant", default="scent", help="Variant key when --env is a task with variants (e.g. 'scent'/'noscent')."
    )

    # ── play ──
    p_play = sub.add_parser("play", help="Play VoxelForage yourself (keyboard).")
    p_play.add_argument("--task", default="voxel_forage", choices=list(TASKS.keys()))
    p_play.add_argument("--no-scent", action="store_true")
    p_play.add_argument("--seed", type=int, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "suite":
        _cmd_suite(args)
    elif args.command == "replay":
        _cmd_replay(args)
    elif args.command == "play":
        _cmd_play(args)
