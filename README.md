# neat3p — *Neat-Python, Plus Plus*

[![tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)
[![license](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

<!-- More badges (coming soon): build · PyPI · Python version -->

![neat3p](assets/neat3p-cover.png)

**NEAT, amplified by C++.** A high-performance, C++-backed Python library for NEAT
(NeuroEvolution of Augmenting Topologies) and beyond.

neat3p pairs the flexibility of a Python API with a modern **C++20** core, bridged by
[nanobind](https://github.com/wjakob/nanobind). It ships the NEAT engine, a set of network
phenotypes (recurrent, feature-attention, HyperNEAT, adaptive), and a small reusable
**gym** (`VoxelForage`) for benchmarking — plus an out-of-package benchmark harness/CLI to
train, compare, and replay agents.

---

## Features

- **High-performance core** — genome / population / reproduction in C++20, exposed via nanobind.
- **Network phenotypes** — recurrent net, feature-attention, HyperNEAT, adaptive HyperNEAT.
- **Reusable gym** — `neat3p.gym_envs` registers the `VoxelForage` environments (a small 3D
  voxel-foraging task, with `scent` / `noscent` / `shaped` variants).
- **Benchmark suite** — `python -m benchmarks` for `train` / `suite` / `replay` / `play`, with
  seeded/held-out evaluation and HTML + Markdown reports.

## Repository layout

```
src/neat3p/            # the importable, pip-shipped library
  ...                  #   NEAT core (genome, population, reproduction, …) + the C++ extension
  nn/                  #   network phenotypes (recurrent / feature-attention / hyperneat / adaptive)
  gym_envs/            #   reusable gym: VoxelForage env + HyperNEAT substrates
benchmarks/            # research harness — NOT part of the pip package
  cli.py __main__.py   #   `python -m benchmarks` (train / suite / replay / play)
  tasks.py models.py   #   TaskSpec / ModelAdapter registries
  runner.py report.py  #   generic runner + HTML/Markdown reports
  runners/gym_eval.py  #   the NEAT × Gymnasium evaluation loop
  artifacts.py         #   winner save / load / replay
  configs/*.cfg        #   one NEAT config per (task, model)
  README.md            #   ← full benchmark CLI reference
tests/                 # pytest suite (core + benchmarks)
```

The library (`src/neat3p`) ships only the engine + phenotypes + the reusable `gym_envs`. The
benchmark harness lives in the top-level `benchmarks/` module (in-repo, not installed), so it
never bloats the deliverable. You can still run experiments standalone via `python -m benchmarks`.

---

## Setup

neat3p builds and runs in a dedicated **conda env on Python 3.12** (referenced as `neat3p` below,
which is also the Makefile's default `CONDA_ENV`).

```bash
conda create --name neat3p python=3.12
conda activate neat3p

# native deps for the C++ build
conda install --name neat3p -c conda-forge spdlog msgpack-c
# (nanobind is vendored via the build; install it only if building manually)
# conda install --name neat3p -c conda-forge nanobind

# python build tooling
conda run -n neat3p pip install --upgrade build scikit-build-core
```

## Build & install

The Makefile wraps the build/install in the conda env for you. Pass `EXTRAS` to pull optional
dependency groups onto the installed wheel:

| Extra | Pulls in | Use for |
|-------|----------|---------|
| `[nn]` | `torch` | the network phenotypes |
| `[bench]` | `torch`, `gymnasium[classic-control,box2d]`, `psutil`, `plotly`, `pandas`, `tqdm` | the benchmark suite + tests |

```bash
# build the wheel into dist/ (compiles the C++ extension via scikit-build / Ninja)
make build

# install the freshly built wheel (force-reinstall) WITH the benchmark extras
make install EXTRAS='[bench]'

# …or do both in one go
make build install EXTRAS='[bench]'
```

> Override the target env with `CONDA_ENV=myenv make build install EXTRAS='[bench]'`.

A bare `pip install .` also works for a plain library install (no extras), but the `make`
targets are the supported path for a full bench-capable environment.

## Testing

Run the full pytest suite (core + benchmarks) in the conda env:

```bash
make test          # == conda run -n neat3p pytest tests
```

### Fast functional checks

Quick end-to-end sanity runs through the benchmark CLI (all from the repo root):

```bash
# CartPole — must CONVERGE (basic sanity that the engine + phenotypes work)
conda run --no-capture-output -n neat3p python -m benchmarks train \
    --task cartpole --model recurrent_net --seed 42 \
    --generations 80 --episodes-per-genome 2 --eval-episodes 20 --quiet

conda run --no-capture-output -n neat3p python -m benchmarks train \
    --task cartpole --model feature_attention --seed 42 \
    --generations 80 --episodes-per-genome 2 --eval-episodes 20 \
    --pretrain-episodes 50 --pretrain-epochs 30 --quiet

# VoxelForage — SMOKE only (exercises the shaped + per_generation pipeline end-to-end;
# tiny generation budget, not trained to performance)
conda run --no-capture-output -n neat3p python -m benchmarks train \
    --task voxel_forage --model recurrent_net --variant shaped \
    --eval-strategy per_generation --seed 42 \
    --generations 5 --episodes-per-genome 4 --eval-episodes 10 \
    --validation-episodes 5 --quiet

conda run --no-capture-output -n neat3p python -m benchmarks train \
    --task voxel_forage --model feature_attention --variant shaped \
    --eval-strategy per_generation --seed 42 \
    --generations 5 --episodes-per-genome 4 --eval-episodes 10 \
    --validation-episodes 5 --pretrain-episodes 20 --pretrain-epochs 15 --quiet
```

CartPole solves quickly (recurrent_net typically reaches 500/500 within a few generations);
the VoxelForage runs above are smoke tests — for a real run, raise `--generations` and use
`suite` with multiple seeds (see `benchmarks/README.md`).

## Formatting

```bash
make format        # clang-format (C++) + ruff format & check --fix (Python), repo-wide
```

To format only a subset (e.g. while a wider refactor is pending), scope ruff directly:

```bash
conda run -n neat3p ruff format benchmarks src/neat3p/gym_envs
conda run -n neat3p ruff check --fix benchmarks src/neat3p/gym_envs
```

---

## Quickstart

### Library

```python
import neat3p

config = neat3p.GenomeConfig()
config.compatibility_weight_coefficient = 1.5
```

### Benchmark suite

```bash
# train one (task × model) combination
python -m benchmarks train --task voxel_forage --model recurrent_net --seed 42

# compare models on a task and build a report
python -m benchmarks suite --task voxel_forage --models recurrent_net feature_attention \
    --runs 2 --generations 50 --format both --output suite_report

# watch / play
python -m benchmarks replay benchmarks/output/recurrent_net_scent_seed42.pkl
python -m benchmarks play   --task voxel_forage
```

The reusable gym is importable on its own (importing it registers the Gymnasium envs):

```python
import gymnasium as gym
import neat3p.gym_envs          # registers VoxelForage-v0, -NoScent-v0, -Shaped-v0, …

env = gym.make("VoxelForage-Shaped-v0")
```

See **[`benchmarks/README.md`](benchmarks/README.md)** for the full CLI reference (all
subcommands, flags, the task × model matrix, evaluation strategies, and the output layout).

## Building with Docker

```bash
docker build --rm -t neat3p-builder .
docker run --rm -it -v "$(pwd)/dist":/project/dist neat3p-builder
```
