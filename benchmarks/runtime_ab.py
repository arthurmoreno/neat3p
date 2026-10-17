"""CartPole A/B: evolve with the torch phenotype (CPU) and the native phenotype, same seeds, and compare.

Usage (from ``neat3p/``)::

    python -m benchmarks.runtime_ab --seeds 42 43 44 45 46

With identical seeds the genome RNG stream is shared, so the two runs should evolve the same populations
generation by generation; the report says whether they did, how many generations each needed, and wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from benchmarks.runners.gym_eval import run_neat_gym

CFG = os.path.join(os.path.dirname(__file__), "configs", "cartpole.cfg")
SOLVE = 475.0


def _run(label, net_class, net_kwargs, seed, generations, episodes, solve_episodes, cfg=CFG):
    t0 = time.perf_counter()
    result = run_neat_gym(
        "CartPole-v1",
        cfg,
        generations,
        episodes,
        seed=seed,
        net_class=net_class,
        net_kwargs=net_kwargs,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    rewards = result.evaluate_rewards(n_episodes=solve_episodes, seed=seed + 1)
    return {
        "backend": label,
        "seed": seed,
        "generations": len(result.generation_stats),
        "best_per_generation": [round(g["best"], 3) for g in result.generation_stats],
        "mean_per_generation": [round(g["mean"], 3) for g in result.generation_stats],
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_rewards": rewards,
        "solved": float(np.mean(rewards)) >= SOLVE,
        "wall_seconds": wall,
        "nodes": result.winner_nodes,
        "connections": result.winner_connections,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--max-generations", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--solve-episodes", type=int, default=100)
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="unreachable fitness threshold: always run --max-generations (long structural-mutation comparison)",
    )
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "runtime_ab"))
    args = ap.parse_args()

    from neat3p.nn.composite import NEATRecurrentNet
    from neat3p.runtime import NativeRecurrentNet

    os.makedirs(args.out, exist_ok=True)
    cfg = CFG
    if args.no_early_stop:
        cfg = os.path.join(args.out, "cartpole_no_early_stop.cfg")
        with open(CFG) as src, open(cfg, "w") as dst:
            dst.write(src.read().replace("fitness_threshold     = 475.0", "fitness_threshold     = 1e9"))
    rows = []
    for seed in args.seeds:
        run_args = (args.max_generations, args.episodes, args.solve_episodes, cfg)
        torch_row = _run("torch_cpu", NEATRecurrentNet, {"device_alias": "cpu"}, seed, *run_args)
        native_row = _run("native", NativeRecurrentNet, {}, seed, *run_args)
        identical = (
            torch_row["best_per_generation"] == native_row["best_per_generation"]
            and torch_row["mean_per_generation"] == native_row["mean_per_generation"]
            and torch_row["eval_rewards"] == native_row["eval_rewards"]
        )
        rows.append({"seed": seed, "torch": torch_row, "native": native_row, "identical": identical})
        print(
            f"seed {seed}: torch gens={torch_row['generations']} solved={torch_row['solved']} "
            f"{torch_row['wall_seconds']:.1f}s | native gens={native_row['generations']} solved={native_row['solved']} "
            f"{native_row['wall_seconds']:.1f}s | identical={identical}",
            flush=True,
        )

    tag = "runtime_ab_fixed" if args.no_early_stop else "runtime_ab"
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)

    lines = [
        "| seed | torch gens | native gens | torch solved | native solved | identical stats "
        "| torch s | native s | speedup |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        t, n = r["torch"], r["native"]
        lines.append(
            f"| {r['seed']} | {t['generations']} | {n['generations']} | {t['solved']} | {n['solved']} "
            f"| {r['identical']} | {t['wall_seconds']:.1f} | {n['wall_seconds']:.1f} "
            f"| {t['wall_seconds'] / n['wall_seconds']:.2f}x |"
        )
    solved_all = all(r["native"]["solved"] for r in rows)
    lines.append("")
    lines.append(
        f"native solved on {sum(r['native']['solved'] for r in rows)}/{len(rows)} seeds; "
        f"identical to torch on {sum(r['identical'] for r in rows)}/{len(rows)}."
    )
    report = "\n".join(lines)
    with open(os.path.join(args.out, f"{tag}.md"), "w") as f:
        f.write(report + "\n")
    print(report)
    all_identical = all(r["identical"] for r in rows)
    return 0 if solved_all and all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
