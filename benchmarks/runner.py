"""
Generic benchmark runner — one function over the (task × model) registry.

All per-(task × model) boilerplate (seeded eval, progress bars, save_winner,
stats-dict assembly) lives here once. Adding a new model or task requires only
a registry entry, not a new file.

Usage::

    from benchmarks.runner import run_benchmark
    result = run_benchmark("voxel_forage", "recurrent_net", seed=42, generations=120)
    # result["final_mean_reward"], result["winner_path"], ...
"""

from __future__ import annotations

import os

import numpy as np
import torch

from neat3p.benchmarks.artifacts import save_winner
from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from benchmarks.models import MODELS
from benchmarks.tasks import TASKS

_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "output")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def stats_dict(
    *,
    task_name: str,
    model_kind: str,
    result,
    rewards: list[float],
    env_id: str,
    solve_threshold: float,
    winner_path: str,
    seed: int,
) -> dict:
    """Assemble the canonical ~15-key stats dict from a GymEvalResult."""
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= solve_threshold), None)
    return {
        "benchmark_name": f"{task_name}_{model_kind}",
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


def run_benchmark(
    task_name: str,
    model_name: str,
    *,
    seed: int,
    generations: int | None = None,
    episodes_per_genome: int = 4,
    eval_episodes: int = 50,
    variant: str = "scent",
    save_dir: str | None = None,
    verbose: bool = True,
    progress: bool = False,
    progress_desc: str = "",
    progress_position: int = 0,
    eval_strategy: str = "per_generation",
    validation_episodes: int = 0,
    **tunables,
) -> dict:
    """Run one benchmark trial. Returns the canonical stats dict.

    task_name:  key in TASKS ("cartpole", "voxel_forage", …)
    model_name: key in MODELS ("recurrent_net", "feature_attention", …)
    variant:    task variant key (e.g. "scent" / "noscent" for voxel_forage; ignored if no variants)
    **tunables: model-specific knobs forwarded only to the adapter that declares them
    """
    task = TASKS[task_name]
    model = MODELS[model_name]

    task.register()

    env_id = task.variants.get(variant, task.env_id) if task.variants else task.env_id
    cfg_path = model.config_path(task)
    device = _device()

    if generations is None:
        generations = _default_generations(task_name, model_name)

    nb = model.build(task, env_id, seed, device, verbose, **model.accepts(tunables))

    result = run_neat_gym(
        env_id=env_id,
        config_path=cfg_path,
        max_generations=generations,
        episodes_per_genome=episodes_per_genome,
        seed=seed,
        net_class=nb.net_class,
        net_kwargs=nb.net_kwargs,
        verbose=verbose,
        progress=progress,
        progress_desc=progress_desc,
        progress_position=progress_position,
        eval_strategy=eval_strategy,
        validation_episodes=validation_episodes,
    )

    rewards = result.evaluate_rewards(n_episodes=eval_episodes, seed=seed + 1)

    winner_path = save_winner(
        save_dir or _DEFAULT_OUTPUT,
        model_name,
        result,
        env_id,
        seed,
        cfg_path,
        **nb.save_extras,
    )
    if verbose:
        print(f"  saved winner -> {winner_path}")

    return stats_dict(
        task_name=task_name,
        model_kind=model_name,
        result=result,
        rewards=rewards,
        env_id=env_id,
        solve_threshold=task.solve_threshold,
        winner_path=winner_path,
        seed=seed,
    )


def _default_generations(task_name: str, model_name: str) -> int:
    if task_name == "cartpole":
        return 300
    if model_name == "recurrent_net":
        return 120
    return 60
