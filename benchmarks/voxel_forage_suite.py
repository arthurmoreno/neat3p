#!/usr/bin/env python3
"""
VoxelForage comparison suite — run all networks on the central gate and build one HTML report.

Trains every network (recurrent_net, feature_attention, hyper_neat, adaptive_hyperneat) on the
SAME VoxelForage contract and compares them: convergence curves, fitness distributions, winner
complexity, memory, wall-time. Defaults are deliberately short so a full run stays well under
~20 min; every knob is configurable.

Usage:
    python benchmarks/voxel_forage_suite.py                          # all nets, short, scent
    python benchmarks/voxel_forage_suite.py --no-scent               # sparse (harder) variant
    python benchmarks/voxel_forage_suite.py --runs 3 --generations 40
    python benchmarks/voxel_forage_suite.py --networks recurrent_net hyper_neat
    python benchmarks/voxel_forage_suite.py --output my_report.html

Adding a network: drop a benchmarks/voxel_forage_<name>.py exposing
run_benchmark(seed, generations, episodes_per_genome, eval_episodes, verbose, env_id) and add it to REGISTRY.
"""

import argparse
import importlib.util
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

REGISTRY: dict[str, str] = {
    "recurrent_net": "voxel_forage_recurrent_net.py",
    "feature_attention": "voxel_forage_feature_attention.py",
    "hyper_neat": "voxel_forage_hyper_neat.py",
    "adaptive_hyperneat": "voxel_forage_adaptive_hyperneat.py",
}


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    import sys

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    spec.loader.exec_module(mod)
    return mod


# Reuse the report generator from the generic suite.
_suite = _load("neat3p_generic_suite", "suite.py")


def main():
    parser = argparse.ArgumentParser(description="VoxelForage network-comparison suite")
    parser.add_argument(
        "--networks", nargs="+", default=list(REGISTRY.keys()), choices=list(REGISTRY.keys()),
        help="Which networks to compare (default: all).",
    )
    parser.add_argument("--runs", type=int, default=2, help="Runs (seeds) per network (default: 2).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated explicit seeds; overrides --runs.")
    parser.add_argument("--generations", type=int, default=15, help="Max generations per run (default: 15).")
    parser.add_argument("--episodes-per-genome", type=int, default=2, help="Episodes per genome (default: 2).")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes for final winner eval (default: 20).")
    parser.add_argument("--no-scent", action="store_true", help="Use the sparse VoxelForage-NoScent-v0 variant.")
    parser.add_argument("--output", type=str, default="voxel_forage_report.html")
    args = parser.parse_args()

    env_id = "VoxelForage-NoScent-v0" if args.no_scent else "VoxelForage-v0"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(42, 42 + args.runs))

    print("=" * 64)
    print("VoxelForage comparison suite")
    print(f"  networks   : {', '.join(args.networks)}")
    print(f"  env        : {env_id}")
    print(f"  seeds      : {seeds}")
    print(f"  budget     : {args.generations} gens × {args.episodes_per_genome} eps/genome")
    print(f"  output     : {args.output}")
    print("=" * 64)

    all_results: list[dict] = []
    t_suite = time.perf_counter()

    for net in args.networks:
        mod = _load(f"voxel_{net}", REGISTRY[net])
        print(f"\n{'─' * 64}\nNetwork: {net}\n{'─' * 64}")
        for run_i, seed in enumerate(seeds):
            print(f"\n  Run {run_i + 1}/{len(seeds)}  seed={seed}")
            t0 = time.perf_counter()
            result = mod.run_benchmark(
                seed=seed,
                generations=args.generations,
                episodes_per_genome=args.episodes_per_genome,
                eval_episodes=args.eval_episodes,
                verbose=False,
                env_id=env_id,
            )
            elapsed = time.perf_counter() - t0
            solved = result["solve_generation"] is not None
            print(
                f"  ✓  {'SOLVED' if solved else 'done'} "
                f"| best_food={result['winner_fitness']:.1f}/120 "
                f"| eval={result['final_mean_reward']:.1f} "
                f"| {elapsed:.0f}s"
            )
            all_results.append(result)

    total = time.perf_counter() - t_suite
    print(f"\n{'=' * 64}\nSuite finished in {total:.0f}s. Building report...")
    benchmark_names = [r["benchmark_name"] for r in all_results]
    # preserve order, de-dup
    ordered = list(dict.fromkeys(benchmark_names))
    _suite.build_report(all_results, ordered, args.output)


if __name__ == "__main__":
    main()
