#!/usr/bin/env python3
"""
Retina visual-discrimination benchmark — plain HyperNEAT.

The retina problem is the canonical task for showing a HyperNEAT substrate's
*spatial / modular* advantage: an 8-pixel retina is split into a LEFT block and a
RIGHT block, each block independently signals whether it "contains an object", and
the network must combine the two sides (default: LEFT AND RIGHT). Because the two
sides are spatially separated on the substrate, a CPPN that learns a geometric
weight pattern can wire each side's pixels to its own hidden units — the modularity
that direct encodings struggle to discover.

Retina layout (2 rows × 4 columns, pixel value ∈ {-1, +1}, +1 = lit):

        col0 col1 | col2 col3
    row0  0    1  |  2    3
    row1  4    5  |  6    7
        └ LEFT ┘   └ RIGHT ┘

Per-block "object" rule (fully specified here, geometric & nonlinear):
  • LEFT  object present  ⇔ a vertical bar exists   = (p0 & p4) OR (p1 & p5)
  • RIGHT object present  ⇔ a horizontal bar exists = (p2 & p3) OR (p6 & p7)
Target (objective="and"): +1 iff a left-object AND a right-object are both present.
("or" combines them with OR instead.) The network has a single tanh output; its
sign is the prediction. Fitness = classification accuracy over all 2**8 = 256
patterns, evaluated in a single batched forward pass per genome (very fast).

IMPORTANT: this benchmark uses the shipping ``HyperNEATNet`` class unchanged. The
only local code is ``RetinaHyperNEATNet`` — a thin wrapper that bakes the fixed
retina substrate coordinates, exactly the pattern the game will use to plug a
HyperNEAT brain in. The net contract (create / activate / reset) is untouched.

Usage:
    python benchmarks/retina_hyper_neat.py
    python benchmarks/retina_hyper_neat.py --seed 1 --generations 300
    python benchmarks/retina_hyper_neat.py --objective or
"""

import os
import time

import numpy as np
import torch

import neat3p
from neat3p.nn.composite import HyperNEATNet

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/retina_hyper_neat.cfg"))

BENCHMARK_NAME = "retina_hyper_neat"
SOLVE_THRESHOLD = 0.99  # fraction of the 256 patterns classified correctly

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Retina substrate geometry ────────────────────────────────────────────────
# 8 input pixels on a 2×4 grid; left block at x<0, right block at x>0.
# Index i = row*4 + col, so this order matches the pixel indices used below.
_INPUT_COORDS = [
    [-0.75, 0.5], [-0.25, 0.5], [0.25, 0.5], [0.75, 0.5],   # row 0 (p0..p3)
    [-0.75, -0.5], [-0.25, -0.5], [0.25, -0.5], [0.75, -0.5],  # row 1 (p4..p7)
]
# 6 hidden nodes, 3 on the left half (x=-0.5) and 3 on the right half (x=0.5),
# so the CPPN can route each retina side to its own hidden sub-population.
_HIDDEN_COORDS = [
    [-0.5, 0.4], [-0.5, 0.0], [-0.5, -0.4],
    [0.5, 0.4], [0.5, 0.0], [0.5, -0.4],
]
# 1 output node (the side-combiner) at the bottom centre.
_OUTPUT_COORDS = [[0.0, -1.0]]


class RetinaHyperNEATNet:
    """Thin wrapper that bakes the fixed retina substrate into ``HyperNEATNet``.

    Mirrors ``CartPoleAdaptiveNet``: it supplies the substrate coordinates and
    otherwise delegates straight to the shipping net. The net's create/activate/
    reset contract is unchanged — this is the same class the game will run.
    """

    def __init__(self, net: HyperNEATNet) -> None:
        self._net = net

    @classmethod
    def create(cls, genome, config, batch_size: int = 1, device: str = _DEVICE) -> "RetinaHyperNEATNet":
        net = HyperNEATNet.create(
            genome,
            config,
            input_coords=_INPUT_COORDS,
            hidden_coords=_HIDDEN_COORDS,
            output_coords=_OUTPUT_COORDS,
            batch_size=batch_size,
            device=device,
        )
        return cls(net)

    def activate(self, inputs):
        return self._net.activate(inputs)

    def reset(self, batch_size=None):
        self._net.reset(batch_size)


# ── Task definition: all 256 patterns + ground-truth targets ─────────────────


def _build_dataset(objective: str):
    """Return (patterns, targets): patterns (256, 8) ∈ {-1,+1}, targets (256,) ∈ {-1,+1}."""
    patterns = np.empty((256, 8), dtype=np.float32)
    targets = np.empty((256,), dtype=np.float32)
    for v in range(256):
        bits = [(v >> i) & 1 for i in range(8)]  # bits[0]=p0 ... bits[7]=p7
        p = bits  # 1 = lit
        left_object = (p[0] and p[4]) or (p[1] and p[5])      # vertical bar in left block
        right_object = (p[2] and p[3]) or (p[6] and p[7])     # horizontal bar in right block
        if objective == "or":
            present = left_object or right_object
        else:  # "and"
            present = left_object and right_object
        patterns[v] = [1.0 if b else -1.0 for b in bits]
        targets[v] = 1.0 if present else -1.0
    return patterns, targets


def _accuracy(net, patterns_list, targets_np) -> float:
    net.reset(batch_size=len(patterns_list))
    out = net.activate(patterns_list)          # (256, 1)
    preds = torch.sign(out[:, 0])
    preds = torch.where(preds == 0, torch.ones_like(preds), preds)  # sign(0) -> +1
    targets = torch.as_tensor(targets_np, device=preds.device)
    return float((preds == targets).float().mean().item())


def run_benchmark(
    seed: int,
    generations: int = 200,
    objective: str = "and",
    verbose: bool = True,
) -> dict:
    """Run one retina HyperNEAT trial. Returns a stats dict (suite-compatible)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random

    random.seed(seed)

    config = neat3p.Config(
        neat3p.DefaultGenome, neat3p.DefaultReproduction,
        neat3p.DefaultSpeciesSet, neat3p.DefaultStagnation, _CFG,
    )

    patterns_np, targets_np = _build_dataset(objective)
    patterns_list = patterns_np.tolist()

    def eval_genomes(genomes, cfg):
        for _gid, genome in genomes:
            net = RetinaHyperNEATNet.create(genome, cfg, batch_size=len(patterns_list), device=_DEVICE)
            genome.fitness = _accuracy(net, patterns_list, targets_np)

    pop = neat3p.Population(config)
    if verbose:
        pop.add_reporter(neat3p.StdOutReporter(True))
    stats = neat3p.StatisticsReporter()
    pop.add_reporter(stats)

    t0 = time.perf_counter()
    winner = pop.run(eval_genomes, generations)
    wall_time = time.perf_counter() - t0

    gen_stats = []
    for i, best_genome in enumerate(stats.most_fit_genomes):
        gen_stats.append({"generation": i, "best": float(best_genome.fitness)})
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)

    return {
        "benchmark_name": BENCHMARK_NAME,
        "seed": seed,
        "objective": objective,
        "solve_generation": solve_gen,
        "total_generations": len(gen_stats),
        "winner_fitness": float(winner.fitness),
        "final_accuracy": float(winner.fitness),
        "wall_time_seconds": wall_time,
        "winner_nodes": len(winner.nodes),
        "winner_connections": sum(1 for c in winner.connections.values() if c.enabled),
        "generation_stats": gen_stats,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--objective", choices=["and", "or"], default="and")
    args = parser.parse_args()

    print(
        f"Training Retina ({args.objective.upper()}) with plain HyperNEAT "
        f"(seed={args.seed}, max_generations={args.generations}, device={_DEVICE})..."
    )

    result = run_benchmark(
        seed=args.seed,
        generations=args.generations,
        objective=args.objective,
        verbose=True,
    )

    print(f"\nWinner accuracy      : {result['final_accuracy'] * 100:.1f}% ({result['final_accuracy'] * 256:.0f}/256)")
    print(f"Solve generation     : {result['solve_generation']}")
    print(f"Wall time            : {result['wall_time_seconds']:.1f}s")
    print(f"Winner nodes/conns   : {result['winner_nodes']} / {result['winner_connections']}")


if __name__ == "__main__":
    main()
