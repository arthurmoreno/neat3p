#!/usr/bin/env python3
"""
CartPole-v1 benchmark with NEATNetWithFeatureAttention — two explicit phases.

Phase 1 — Pre-train the encoder + attention head (NEAT frozen):
  obs(4) → encoder(4→feature_dim) → attn → decoder(feature_dim→4) → obs

Phase 2 — Freeze front-end, evolve NEAT:
  obs(4) → [frozen encoder] → [frozen attn] → RecurrentNet (evolves) → action

Research note: without pre-training, even 4D→4D random projection + random attention
  gate prevents NEAT convergence (observed: no solution by generation 14+).

Usage:
    python benchmarks/cartpole_feature_attention.py
    python benchmarks/cartpole_feature_attention.py --no-pretrain   # random frozen baseline
    python benchmarks/cartpole_feature_attention.py --pretrain-epochs 50
    python benchmarks/cartpole_feature_attention.py --no-render
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.nn.composite import NEATNetWithFeatureAttention
from neat3p.nn.modules.attention import FeatureAttention
from neat3p.nn.modules.encoders import SimpleEncoder

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/cartpole.cfg"))

BENCHMARK_NAME = "cartpole_feature_attention"
SOLVE_THRESHOLD = 475.0
STATE_DIM = 4


def _pretrain_frontend(
    state_dim: int,
    feature_dim: int,
    device: str,
    pretrain_episodes: int,
    pretrain_epochs: int,
    pretrain_batch: int,
    pretrain_lr: float,
    seed: int,
    verbose: bool,
) -> tuple[SimpleEncoder, FeatureAttention]:
    import gymnasium as gym

    encoder = SimpleEncoder(state_dim, feature_dim, device=device)
    attn = FeatureAttention(input_dim=feature_dim, device=device)

    if verbose:
        print(f"  Collecting {pretrain_episodes} random-policy episodes from CartPole-v1...")

    env = gym.make("CartPole-v1")
    obs_buffer = []
    rng = np.random.default_rng(seed)
    for _ in range(pretrain_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        terminated = truncated = False
        while not (terminated or truncated):
            obs_buffer.append(obs.copy())
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
    env.close()

    obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32).to(device)
    if verbose:
        print(f"  Collected {len(obs_tensor)} observations.")

    decoder = nn.Linear(feature_dim, state_dim).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(attn.parameters()) + list(decoder.parameters()),
        lr=pretrain_lr,
    )
    dataset = torch.utils.data.TensorDataset(obs_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=pretrain_batch, shuffle=True)

    if verbose:
        print(
            f"  Training autoencoder {pretrain_epochs} epochs "
            f"({state_dim}→{feature_dim}→{state_dim})..."
        )
    encoder.train()
    attn.train()
    decoder.train()
    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(decoder(attn(encoder(batch))), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  epoch {epoch + 1:>4}/{pretrain_epochs}  loss={epoch_loss / len(loader):.5f}")

    encoder.eval()
    attn.eval()
    return encoder, attn


def run_benchmark(
    seed: int,
    generations: int = 300,
    episodes_per_genome: int = 10,
    feature_dim: int = 4,
    pretrain: bool = True,
    pretrain_episodes: int = 200,
    pretrain_epochs: int = 100,
    pretrain_batch: int = 256,
    pretrain_lr: float = 1e-3,
    eval_episodes: int = 100,
    verbose: bool = True,
) -> dict:
    """Run one CartPole-FeatureAttention trial. Returns a stats dict for the benchmark suite."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)

    if pretrain:
        if verbose:
            print("  Phase 1 — Pre-training encoder + attention...")
        encoder, attn = _pretrain_frontend(
            STATE_DIM, feature_dim, device,
            pretrain_episodes, pretrain_epochs, pretrain_batch, pretrain_lr,
            seed, verbose,
        )
        if verbose:
            print("  Pre-training complete.\n")
    else:
        encoder = SimpleEncoder(STATE_DIM, feature_dim, device=device).eval()
        attn = FeatureAttention(input_dim=feature_dim, device=device).eval()
        if verbose:
            print("  Skipping pre-training — random frozen front-end.\n")

    result = run_neat_gym(
        env_id="CartPole-v1",
        config_path=_CFG,
        max_generations=generations,
        episodes_per_genome=episodes_per_genome,
        seed=seed,
        net_class=NEATNetWithFeatureAttention,
        net_kwargs={"encoder": encoder, "attn": attn},
        verbose=verbose,
    )

    rewards = result.evaluate_rewards(n_episodes=eval_episodes, seed=seed + 1)
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)

    return {
        "benchmark_name": BENCHMARK_NAME,
        "seed": seed,
        "pretrain": pretrain,
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


def main():
    import argparse
    import gymnasium as gym

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--episodes-per-genome", type=int, default=10)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--pretrain-episodes", type=int, default=200)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--pretrain-batch", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Training on CartPole-v1 with NEATNetWithFeatureAttention")
    print(f"  seed={args.seed}  max_generations={args.generations}  feature_dim={args.feature_dim}")
    print(f"  pipeline: obs({STATE_DIM}) → SimpleEncoder({args.feature_dim}) → FeatureAttention → RecurrentNet → action")
    print()

    if not args.no_pretrain:
        print("  Phase 1 — Pre-training encoder + attention...")
        encoder, attn = _pretrain_frontend(
            STATE_DIM, args.feature_dim, device,
            args.pretrain_episodes, args.pretrain_epochs, args.pretrain_batch, args.pretrain_lr,
            args.seed, verbose=True,
        )
        print("  Pre-training complete.\n")
    else:
        encoder = SimpleEncoder(STATE_DIM, args.feature_dim, device=device).eval()
        attn = FeatureAttention(input_dim=args.feature_dim, device=device).eval()
        print("  Skipping pre-training — random frozen front-end.\n")

    print("=" * 60)
    print("Phase 2 — NEAT evolution with frozen front-end")

    result = run_neat_gym(
        env_id="CartPole-v1",
        config_path=_CFG,
        max_generations=args.generations,
        episodes_per_genome=args.episodes_per_genome,
        seed=args.seed,
        net_class=NEATNetWithFeatureAttention,
        net_kwargs={"encoder": encoder, "attn": attn},
        verbose=True,
    )

    rewards = result.evaluate_rewards(n_episodes=100, seed=args.seed + 1)
    gen_stats = result.generation_stats
    solve_gen = next((s["generation"] for s in gen_stats if s["best"] >= SOLVE_THRESHOLD), None)

    print(f"\nWinner fitness       : {result.winner.fitness:.1f}")
    print(f"Mean reward (100 ep) : {float(__import__('numpy').mean(rewards)):.1f} ± {float(__import__('numpy').std(rewards)):.1f}")
    print(f"Solve generation     : {solve_gen}")
    print(f"Wall time            : {result.wall_time_seconds:.1f}s")
    print(f"Winner nodes/conns   : {result.winner_nodes} / {result.winner_connections}")

    if not args.no_render:
        print(f"\nRendering {args.episodes} episode(s)...")
        env = gym.make("CartPole-v1", render_mode="human")
        net = NEATNetWithFeatureAttention(
            result.winner, result.state_dim, result.action_dim, result.config,
            encoder=encoder, attn=attn,
        )
        for ep in range(args.episodes):
            obs, _ = env.reset()
            net.reset(batch_size=1)
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                out = net(obs_t)
                action = int(out.argmax(dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
            print(f"  Episode {ep + 1}: reward={total:.0f}")
        env.close()


if __name__ == "__main__":
    main()
