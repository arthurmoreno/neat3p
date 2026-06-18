# neat3p Benchmarks

Structured comparison of NEAT network architectures on canonical RL tasks.

## Task × model matrix

| Task | recurrent\_net | feature\_attention | hyper\_neat | adaptive\_hyperneat |
|---|:---:|:---:|:---:|:---:|
| cartpole | ✓ | ✓ | — | ✓ |
| lunarlander | — | ✓ | — | — |
| voxel\_forage | ✓ | ✓ | ✓ | ✓ |

**VoxelForage** (`VoxelForage-v0`) is the central neat3p gate: a 3D voxel foraging task
(`obs: (C,D,H,W) egocentric patch + 2 scalar attributes → Discrete(7) actions`) that mirrors
the game's observation/action contract. All four nets are scored on the same fixed-shape
problem so results are directly comparable.

## Commands

All commands go through the unified CLI:

```bash
python -m neat3p.benchmarks <subcommand> [options]
```

### train — one (task × model) run

```bash
# VoxelForage plain NEAT, 120 generations, scent variant
python -m neat3p.benchmarks train \
    --task voxel_forage --model recurrent_net \
    --seed 42 --generations 120 --episodes-per-genome 5

# Feature-attention with pretrain, no-scent (harder)
python -m neat3p.benchmarks train \
    --task voxel_forage --model feature_attention \
    --variant noscent --pretrain-episodes 300 --pretrain-epochs 120

# CartPole with Adaptive HyperNEAT, watch winner after training
python -m neat3p.benchmarks train \
    --task cartpole --model adaptive_hyperneat --render
```

### suite — compare all models on one task

```bash
# Short comparison (15 gens × 2 seeds), builds HTML report
python -m neat3p.benchmarks suite \
    --task voxel_forage --runs 2 --generations 15 \
    --output voxel_forage_report.html

# Only two models, scent variant, both HTML and Markdown
python -m neat3p.benchmarks suite \
    --task voxel_forage \
    --models recurrent_net feature_attention \
    --format both --output comparison

# Longer run, no-scent, custom seeds
python -m neat3p.benchmarks suite \
    --task voxel_forage --generations 60 --seeds 42,123,456 \
    --variant noscent
```

### replay — watch a saved champion

```bash
# List saved winners
python -m neat3p.benchmarks replay

# Watch a specific winner
python -m neat3p.benchmarks replay benchmarks/output/recurrent_net_scent_seed42.pkl
python -m neat3p.benchmarks replay benchmarks/output/feature_attention_scent_seed42.pkl --episodes 5
python -m neat3p.benchmarks replay benchmarks/output/hyper_neat_scent_seed42.pkl --no-render
```

### play — play VoxelForage yourself

```bash
python -m neat3p.benchmarks play --task voxel_forage
python -m neat3p.benchmarks play --task voxel_forage --no-scent
```

Controls: arrows/WASD to move, Q/E for up/down, Space to idle, R to reset, Esc to quit.

## Fixed-seed evaluation

During training, each generation uses **K shared seeded worlds** (the same K layouts for
every genome that generation, resampled the next generation). This gives fair intra-generation
ranking without letting any genome overfit a fixed layout across generations.

The final winner is evaluated on a **held-out seed stream** (`seed+1`) distinct from training,
measuring true generalisation.

Set `--eval-strategy fixed` for a stable fitness curve (same K worlds every generation,
cleaner slope but can overfit those K layouts), or `random` for the original unseeded behaviour.

## Pretraining (feature\_attention only)

The `feature_attention` model pre-trains a `SimpleEncoder + FeatureAttention` front-end as an
autoencoder on random-policy rollouts, then freezes it before NEAT evolution starts. This gives
the NEAT controller a compact, stable representation to work with.

Control it with:
- `--pretrain-episodes N` (default 250) — how many random-policy episodes to collect
- `--pretrain-epochs N` (default 100) — autoencoder training epochs

The final reconstruction loss is printed so you can judge convergence.

## Output layout

All winners and reports land in `benchmarks/output/` by default:

```
benchmarks/output/
  recurrent_net_scent_seed42.pkl        ← atomic write, self-contained
  feature_attention_scent_seed42.pkl
  hyper_neat_scent_seed42.pkl
  adaptive_hyperneat_noscent_seed42.pkl
  suite_report.html                     ← Plotly interactive report
  suite_report.md                       ← GitHub-readable + Mermaid chart
```

Filenames follow `<kind>_<scent|noscent>_seed<N>.pkl`. The `.pkl` is a self-contained
package (genome, config path, network kind, and frozen weights for feature_attention) that
`replay` can reconstruct without importing any benchmark script.

## How to add a task

1. Add a `TaskSpec` to `src/neat3p/benchmarks/tasks.py` with the env id, solve threshold,
   variants, and (for HyperNEAT-family) a `substrate` callable returning `(in_c, hid_c, out_c)`.
2. Create a NEAT config in `benchmarks/configs/<task_name>.cfg` (or `<task_name>_<model>.cfg`
   for per-model config overrides).
3. Add the task to `TASKS` in `tasks.py`.

## How to add a model

1. Subclass `ModelAdapter` in `src/neat3p/benchmarks/models.py`.
2. Implement `build(task, env_id, seed, device, verbose, **tunables) -> NetBuild` — the only
   model-specific logic.
3. Implement `rebuild(pkg, device) -> (net, style)` — reconstruction from a saved `.pkl`.
4. Declare `tunables: dict[str, type]` for model-specific CLI/suite knobs.
5. Instantiate and add to `MODELS`.
