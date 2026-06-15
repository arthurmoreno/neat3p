#!/usr/bin/env python3
"""
VoxelForage gate — NEATNetWithFeatureAttention (encoder + attention front-end).

Pipeline: obs(83) -> SimpleEncoder(83->12) -> FeatureAttention -> RecurrentNet(evolves) -> action

The encoder + attention are pre-trained as an autoencoder over random-policy observations,
then frozen and shared across every genome (so controller heritability isn't broken by
per-child random projections). ``--no-pretrain`` gives the random-frozen baseline.

Usage:
    python benchmarks/voxel_forage_feature_attention.py
    python benchmarks/voxel_forage_feature_attention.py --no-scent
    python benchmarks/voxel_forage_feature_attention.py --render
    python benchmarks/voxel_forage_feature_attention.py --no-pretrain
"""

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import neat3p.benchmarks.envs  # noqa: F401  — registers VoxelForage-v0
from neat3p.benchmarks.artifacts import save_winner
from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.nn.composite import NEATNetWithFeatureAttention
from neat3p.nn.modules.attention import FeatureAttention
from neat3p.nn.modules.encoders import SimpleEncoder

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/voxel_forage_feature_attention.cfg"))
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

BENCHMARK_NAME = "voxel_forage_feature_attention"
ENV_ID = "VoxelForage-v0"
SOLVE_THRESHOLD = 110.0
STATE_DIM = 83
FEATURE_DIM = 12  # must match num_inputs in voxel_forage_feature_attention.cfg


def _pretrain_frontend(env_id, device, pretrain_episodes, pretrain_epochs, pretrain_batch, pretrain_lr, seed, verbose):
    encoder = SimpleEncoder(STATE_DIM, FEATURE_DIM, device=device)
    attn = FeatureAttention(input_dim=FEATURE_DIM, device=device)

    env = gym.make(env_id)
    obs_buffer = []
    rng = np.random.default_rng(seed)
    for _ in range(pretrain_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        terminated = truncated = False
        while not (terminated or truncated):
            obs_buffer.append(obs.copy())
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
    env.close()

    obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32).to(device)
    decoder = nn.Linear(FEATURE_DIM, STATE_DIM).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(attn.parameters()) + list(decoder.parameters()), lr=pretrain_lr
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(obs_tensor), batch_size=pretrain_batch, shuffle=True
    )
    if verbose:
        print(f"  Pre-training autoencoder ({STATE_DIM}->{FEATURE_DIM}->{STATE_DIM}) on {len(obs_tensor)} obs...")
    encoder.train(); attn.train(); decoder.train()
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
    encoder.eval(); attn.eval()
    return encoder, attn


def _build_frontend(env_id, pretrain, seed, verbose, pretrain_episodes, pretrain_epochs, pretrain_batch, pretrain_lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrain:
        return _pretrain_frontend(
            env_id, device, pretrain_episodes, pretrain_epochs, pretrain_batch, pretrain_lr, seed, verbose
        )
    if verbose:
        print("  Random frozen front-end (no pre-training).")
    return (
        SimpleEncoder(STATE_DIM, FEATURE_DIM, device=device).eval(),
        FeatureAttention(input_dim=FEATURE_DIM, device=device).eval(),
    )


def run_benchmark(
    seed: int,
    generations: int = 60,
    episodes_per_genome: int = 2,
    eval_episodes: int = 20,
    verbose: bool = True,
    env_id: str = ENV_ID,
    save_dir: str = None,
    pretrain: bool = True,
    pretrain_episodes: int = 40,
    pretrain_epochs: int = 40,
    pretrain_batch: int = 256,
    pretrain_lr: float = 1e-3,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    encoder, attn = _build_frontend(
        env_id, pretrain, seed, verbose, pretrain_episodes, pretrain_epochs, pretrain_batch, pretrain_lr
    )

    result = run_neat_gym(
        env_id=env_id,
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

    winner_path = save_winner(
        save_dir or _OUTPUT_DIR, "feature_attention", result, env_id, seed, _CFG,
        feature_dim=FEATURE_DIM, encoder=encoder, attn=attn,
    )
    if verbose:
        print(f"  saved winner -> {winner_path}")

    return {
        "benchmark_name": BENCHMARK_NAME,
        "env_id": env_id,
        "winner_path": winner_path,
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


def _watch(winner, config, encoder, attn, episodes, env_id):
    env = gym.make(env_id, render_mode="human")
    net = NEATNetWithFeatureAttention(winner, STATE_DIM, env.action_space.n, config, encoder=encoder, attn=attn)
    for ep in range(episodes):
        obs, _ = env.reset()
        net.reset(batch_size=1)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = int(net(obs_t).argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        print(f"  Episode {ep + 1}: collected={total:.0f} steps={info['steps']}")
    env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--episodes-per-genome", type=int, default=2)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--no-scent", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env_id = "VoxelForage-NoScent-v0" if args.no_scent else ENV_ID
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Training {env_id} with NEATNetWithFeatureAttention (seed={args.seed}, gens={args.generations})...")

    encoder, attn = _build_frontend(env_id, not args.no_pretrain, args.seed, True, 40, 40, 256, 1e-3)
    result = run_neat_gym(
        env_id=env_id, config_path=_CFG, max_generations=args.generations,
        episodes_per_genome=args.episodes_per_genome, seed=args.seed,
        net_class=NEATNetWithFeatureAttention, net_kwargs={"encoder": encoder, "attn": attn}, verbose=True,
    )
    rewards = result.evaluate_rewards(n_episodes=20, seed=args.seed + 1)
    print(f"\nWinner fitness (food): {result.winner.fitness:.1f} / 120")
    print(f"Mean collected (20 ep): {float(np.mean(rewards)):.1f} ± {float(np.std(rewards)):.1f}")
    print(f"Wall time            : {result.wall_time_seconds:.1f}s")
    print(f"Winner nodes/conns   : {result.winner_nodes} / {result.winner_connections}")

    if args.render:
        print(f"\nRendering {args.episodes} episode(s)...")
        _watch(result.winner, result.config, encoder, attn, args.episodes, env_id)


if __name__ == "__main__":
    main()
