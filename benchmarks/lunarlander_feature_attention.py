#!/usr/bin/env python3
"""
LunarLander-v3 benchmark with NEATNetWithFeatureAttention — two explicit phases.

Phase 1 — Train the encoder + attention head (NEAT frozen):
  Collect random-policy observations from LunarLander.
  Train SimpleEncoder + FeatureAttention via reconstruction loss (autoencoder):
    obs(8) → encoder(8→4) → attn(4→4) → decoder(4→8) → obs_reconstructed
  The front-end learns a compact, meaningful representation of the 8D state
  (position, velocity, angle, leg contacts) before NEAT touches it.

Phase 2 — Freeze the front-end, evolve NEAT:
  Load the pre-trained encoder + attention weights into NEATNetWithFeatureAttention.
  Freeze them (requires_grad=False). NEAT evolves the RecurrentNet controller
  on top of the fixed, pre-trained representation.
  Pipeline: obs(8) → [frozen encoder] → [frozen attn] → RecurrentNet (evolves) → action

Usage:
    python benchmarks/lunarlander_feature_attention.py
    python benchmarks/lunarlander_feature_attention.py --no-render
    python benchmarks/lunarlander_feature_attention.py --pretrain-epochs 50 --generations 500
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neat3p.benchmarks.runners.gym_eval import run_neat_gym
from neat3p.nn.composite import NEATNetWithFeatureAttention
from neat3p.nn.modules.attention import FeatureAttention
from neat3p.nn.modules.encoders import SimpleEncoder

_CFG = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/lunarlander.cfg"))

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--feature-dim", type=int, default=4, help="Encoder output dim. LunarLander obs is 8D; 4D keeps half the info."
)
parser.add_argument(
    "--pretrain-episodes", type=int, default=200, help="Episodes of random rollouts to collect for pre-training."
)
parser.add_argument("--pretrain-epochs", type=int, default=100, help="Autoencoder training epochs.")
parser.add_argument("--pretrain-batch", type=int, default=256)
parser.add_argument("--pretrain-lr", type=float, default=1e-3)
parser.add_argument("--generations", type=int, default=500)
parser.add_argument("--episodes-per-genome", type=int, default=5)
parser.add_argument("--episodes", type=int, default=3, help="Episodes to render after training.")
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE_DIM = 8  # LunarLander-v3 observation space
ACTION_DIM = 4  # LunarLander-v3 discrete actions

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Phase 1 — Collect observations + train encoder + attention
# ---------------------------------------------------------------------------

print("=" * 60)
print("Phase 1 — Pre-training encoder + attention (NEAT frozen)")
print(f"  Collecting {args.pretrain_episodes} random-policy episodes...")

env = gym.make("LunarLander-v3")
obs_buffer = []
for _ in range(args.pretrain_episodes):
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        obs_buffer.append(obs.copy())
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
env.close()

obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32).to(DEVICE)
print(f"  Collected {len(obs_tensor)} observations from {STATE_DIM}D state space.")

# Autoencoder: encoder → attn → decoder
encoder = SimpleEncoder(STATE_DIM, args.feature_dim, device=DEVICE)
attn = FeatureAttention(input_dim=args.feature_dim, device=DEVICE)
decoder = nn.Linear(args.feature_dim, STATE_DIM).to(DEVICE)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(attn.parameters()) + list(decoder.parameters()),
    lr=args.pretrain_lr,
)

dataset = torch.utils.data.TensorDataset(obs_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.pretrain_batch, shuffle=True)

print(
    f"  Training autoencoder for {args.pretrain_epochs} epochs "
    f"(obs → encoder({STATE_DIM}→{args.feature_dim}) → attn → decoder → obs)..."
)

encoder.train()
attn.train()
decoder.train()
for epoch in range(args.pretrain_epochs):
    epoch_loss = 0.0
    for (batch,) in loader:
        optimizer.zero_grad()
        features = encoder(batch)
        gated = attn(features)
        reconstructed = decoder(gated)
        loss = F.mse_loss(reconstructed, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  epoch {epoch + 1:>4}/{args.pretrain_epochs}  loss={epoch_loss / len(loader):.5f}")

encoder.eval()
attn.eval()

encoder_state = {k: v.cpu() for k, v in encoder.state_dict().items()}
attn_state = {k: v.cpu() for k, v in attn.state_dict().items()}
print("  Pre-training complete. Weights captured.\n")

# ---------------------------------------------------------------------------
# Phase 2 — Freeze front-end, evolve NEAT
# ---------------------------------------------------------------------------

print("=" * 60)
print("Phase 2 — NEAT evolution with frozen pre-trained front-end")
print(
    f"  pipeline: obs(8) → [frozen encoder({STATE_DIM}→{args.feature_dim})] "
    f"→ [frozen attn] → RecurrentNet (evolves) → action"
)
print(f"  max_generations={args.generations}  episodes_per_genome={args.episodes_per_genome}\n")

result = run_neat_gym(
    env_id="LunarLander-v3",
    config_path=_CFG,
    max_generations=args.generations,
    episodes_per_genome=args.episodes_per_genome,
    seed=args.seed,
    net_class=NEATNetWithFeatureAttention,
    net_kwargs={
        "feature_dim": args.feature_dim,
        "encoder_state_dict": encoder_state,
        "attn_state_dict": attn_state,
        "freeze_encoder": True,
    },
)

mean = result.evaluate(n_episodes=100, seed=args.seed + 1)
print(f"\nWinner fitness     : {result.winner.fitness:.1f}")
print(f"Mean reward (100ep): {mean:.1f}")

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

if not args.no_render:
    print(f"\nRendering {args.episodes} episode(s)...")
    env = gym.make("LunarLander-v3", render_mode="human")
    net = NEATNetWithFeatureAttention(
        result.winner,
        STATE_DIM,
        ACTION_DIM,
        result.config,
        feature_dim=args.feature_dim,
        encoder_state_dict=encoder_state,
        attn_state_dict=attn_state,
        freeze_encoder=True,
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
        print(f"  Episode {ep + 1}: reward={total:.1f}")
    env.close()
