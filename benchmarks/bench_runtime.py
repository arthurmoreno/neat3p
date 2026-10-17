"""Per-call inference cost: torch CPU, torch CUDA (if present), the numpy reference, and the native evaluator.

Usage (from ``neat3p/``)::

    python -m benchmarks.bench_runtime            # prints a table and writes benchmarks/output/bench_runtime.md

The timed call is what the game does per creature: state array in, greedy action out (exploration off).
Brains are dense (full_direct) so every size is a worst case for its shape.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time

import numpy as np

import neat3p
from neat3p.genome import DefaultGenome
from neat3p.runtime import NativeRecurrentNet, ReferenceRecurrentNet, build_recurrent_layout

_LIFESIM_CFG = os.path.join(os.path.dirname(__file__), "..", "tests", "runtime", "configs", "lifesim_recurrent.cfg")

SIZES = [  # (label, n_inputs, n_outputs, n_hidden)
    ("cartpole 4-0-2", 4, 2, 0),
    ("cartpole 4-8-2", 4, 2, 8),
    ("lifesim 90-0-3", 90, 3, 0),
    ("lifesim 90-8-3", 90, 3, 8),
    ("lifesim 90-32-3", 90, 3, 32),
    ("lifesim 90-128-3", 90, 3, 128),
]


def _config(n_in: int, n_out: int, n_hidden: int, tmpdir: str):
    text = open(_LIFESIM_CFG).read()
    text = text.replace("num_inputs              = 90", f"num_inputs              = {n_in}")
    text = text.replace("num_hidden              = 0", f"num_hidden              = {n_hidden}")
    text = text.replace("num_outputs             = 3", f"num_outputs             = {n_out}")
    text = text.replace("initial_connection      = partial_nodirect 0.5", "initial_connection      = full_direct")
    path = os.path.join(tmpdir, f"bench_{n_in}_{n_hidden}_{n_out}.cfg")
    with open(path, "w") as f:
        f.write(text)
    return neat3p.Config(
        neat3p.DefaultGenome, neat3p.DefaultReproduction, neat3p.DefaultSpeciesSet, neat3p.DefaultStagnation, path
    )


def _time(fn, calls: int, rounds: int = 5) -> float:
    """Median microseconds per call over ``rounds`` rounds of ``calls`` calls."""
    for _ in range(200):
        fn()
    per_round = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        per_round.append((time.perf_counter() - t0) / calls * 1e6)
    return statistics.median(per_round)


def main() -> int:
    import tempfile

    import torch

    from neat3p.nn.composite import NEATRecurrentNet

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "bench_runtime.md"))
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    header = ["size", "torch cpu", "torch cpu 1 thread", "torch cuda", "numpy ref", "native", "native vs torch cpu"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, n_in, n_out, n_hid in SIZES:
            config = _config(n_in, n_out, n_hid, tmp)
            random.seed(5)
            genome = DefaultGenome(key=1)
            genome.configure_new(config.genome_config)
            x32 = np.random.RandomState(0).normal(size=(1, n_in)).astype(np.float32)
            xt = torch.from_numpy(x32)

            def torch_call(net):
                def call():
                    net.get_action(xt)

                return call

            results = {}
            net_cpu = NEATRecurrentNet(genome, n_in, n_out, config, device_alias="cpu")
            net_cpu.explore_rate = 0.0
            results["torch cpu"] = _time(lambda: int(net_cpu.get_action(xt)[0, 0]), args.calls)
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            results["torch cpu 1 thread"] = _time(lambda: int(net_cpu.get_action(xt)[0, 0]), args.calls)
            torch.set_num_threads(prev_threads)
            if has_cuda:
                net_gpu = NEATRecurrentNet(genome, n_in, n_out, config, device_alias="cuda")
                net_gpu.explore_rate = 0.0
                xg = xt.cuda()
                results["torch cuda"] = _time(lambda: int(net_gpu.get_action(xg).cpu()[0, 0]), args.calls)
            ref = ReferenceRecurrentNet(build_recurrent_layout(genome, config.genome_config, use_current_activs=True))
            results["numpy ref"] = _time(lambda: int(np.argmax(ref.activate(x32))), args.calls)
            native = NativeRecurrentNet(genome, n_in, n_out, config, explore_rate=0.0)
            results["native"] = _time(lambda: int(native.get_action(x32)[0, 0]), args.calls)
            rows.append((label, results))
            print(label, {k: round(v, 1) for k, v in results.items()}, flush=True)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, r in rows:
        cuda = f"{r['torch cuda']:.1f}" if "torch cuda" in r else "n/a"
        lines.append(
            f"| {label} | {r['torch cpu']:.1f} | {r['torch cpu 1 thread']:.1f} | {cuda} | {r['numpy ref']:.1f} "
            f"| {r['native']:.1f} | {r['torch cpu'] / r['native']:.1f}x |"
        )
    report = "microseconds per call (median of 5 rounds)\n\n" + "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(report)
    slower = [label for label, r in rows if r["native"] > r["torch cpu"]]
    if slower:
        print("native slower than torch cpu at:", slower)
    return 1 if slower else 0


if __name__ == "__main__":
    raise SystemExit(main())
