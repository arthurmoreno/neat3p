"""
Model registry — one ModelAdapter per NEAT network architecture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

_CONFIGS = os.path.join(os.path.dirname(__file__), "configs")


def _config(name: str) -> str:
    return os.path.join(_CONFIGS, name)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class NetBuild:
    net_class: Any
    net_kwargs: dict
    save_extras: dict


class ModelAdapter:
    kind: str = ""
    tunables: dict[str, type] = {}

    def accepts(self, kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items() if k in self.tunables}

    def config_path(self, task) -> str:
        specific = _config(f"{task.name}_{self.kind}.cfg")
        if os.path.isfile(specific):
            return specific
        return _config(f"{task.name}.cfg")

    def build(self, task, env_id: str, seed: int, device: str, verbose: bool, **tunables) -> NetBuild:
        raise NotImplementedError

    def rebuild(self, pkg: dict, device: str):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RecurrentNet
# ---------------------------------------------------------------------------


class _RecurrentNetAdapter(ModelAdapter):
    kind = "recurrent_net"
    tunables = {}

    def build(self, task, env_id, seed, device, verbose, **tunables) -> NetBuild:
        from neat3p.nn.composite import NEATRecurrentNet

        return NetBuild(NEATRecurrentNet, {}, {})

    def rebuild(self, pkg, device):
        import neat3p
        from neat3p.nn.composite import NEATRecurrentNet

        config = neat3p.Config(
            neat3p.DefaultGenome, neat3p.DefaultReproduction,
            neat3p.DefaultSpeciesSet, neat3p.DefaultStagnation, pkg["config_path"],
        )
        return NEATRecurrentNet(
            pkg["winner_genome"], pkg["state_dim"], pkg["action_dim"], config, device_alias=device
        ), "module"


# ---------------------------------------------------------------------------
# FeatureAttention
# ---------------------------------------------------------------------------


def _pretrain_frontend(state_dim, feature_dim, env_id, device, seed, verbose,
                       pretrain_episodes, pretrain_epochs, pretrain_batch, pretrain_lr):
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from tqdm.auto import tqdm as _tqdm

    from neat3p.nn.modules.attention import FeatureAttention
    from neat3p.nn.modules.encoders import SimpleEncoder

    encoder = SimpleEncoder(state_dim, feature_dim, device=device)
    attn = FeatureAttention(input_dim=feature_dim, device=device)

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
    decoder = nn.Linear(feature_dim, state_dim).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(attn.parameters()) + list(decoder.parameters()),
        lr=pretrain_lr,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(obs_tensor), batch_size=pretrain_batch, shuffle=True,
    )
    encoder.train(); attn.train(); decoder.train()
    ebar = _tqdm(range(pretrain_epochs), desc=f"pretrain {state_dim}->{feature_dim}", position=1, leave=False)
    epoch_loss = 0.0
    for _epoch in ebar:
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(decoder(attn(encoder(batch))), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        ebar.set_postfix(loss=f"{epoch_loss / len(loader):.5f}", refresh=False)
    ebar.close()
    final_loss = epoch_loss / len(loader)
    encoder.eval(); attn.eval()
    _tqdm.write(
        f"  [pretrain] encoder {state_dim}->{feature_dim}: {len(obs_tensor)} obs, "
        f"{pretrain_epochs} epochs, final recon loss={final_loss:.5f}"
    )
    return encoder, attn


class _FeatureAttentionAdapter(ModelAdapter):
    kind = "feature_attention"
    tunables = {
        "pretrain": bool,
        "pretrain_episodes": int,
        "pretrain_epochs": int,
        "pretrain_batch": int,
        "pretrain_lr": float,
        "feature_dim": int,
    }

    def build(self, task, env_id, seed, device, verbose, **tunables) -> NetBuild:
        import gymnasium as gym
        from neat3p.nn.composite import NEATNetWithFeatureAttention
        from neat3p.nn.modules.attention import FeatureAttention
        from neat3p.nn.modules.encoders import SimpleEncoder

        pretrain = tunables.get("pretrain", True)
        pretrain_episodes = tunables.get("pretrain_episodes", 250)
        pretrain_epochs = tunables.get("pretrain_epochs", 100)
        pretrain_batch = tunables.get("pretrain_batch", 256)
        pretrain_lr = tunables.get("pretrain_lr", 1e-3)

        env = gym.make(env_id)
        state_dim = env.observation_space.shape[0]
        env.close()
        feature_dim = tunables.get("feature_dim", max(4, state_dim // 7))

        if pretrain:
            encoder, attn = _pretrain_frontend(
                state_dim, feature_dim, env_id, device, seed, verbose,
                pretrain_episodes, pretrain_epochs, pretrain_batch, pretrain_lr,
            )
        else:
            if verbose:
                print("  Random frozen front-end (no pre-training).")
            encoder = SimpleEncoder(state_dim, feature_dim, device=device).eval()
            attn = FeatureAttention(input_dim=feature_dim, device=device).eval()

        return NetBuild(
            net_class=NEATNetWithFeatureAttention,
            net_kwargs={"encoder": encoder, "attn": attn},
            save_extras={"feature_dim": feature_dim, "encoder": encoder, "attn": attn},
        )

    def rebuild(self, pkg, device):
        import neat3p
        from neat3p.nn.composite import NEATNetWithFeatureAttention
        from neat3p.nn.modules.attention import FeatureAttention
        from neat3p.nn.modules.encoders import SimpleEncoder

        fd, sd, ad = pkg["feature_dim"], pkg["state_dim"], pkg["action_dim"]
        enc = SimpleEncoder(sd, fd, device=device)
        enc.load_state_dict(pkg["encoder_state_dict"]); enc.eval()
        attn = FeatureAttention(input_dim=fd, device=device)
        attn.load_state_dict(pkg["attn_state_dict"]); attn.eval()
        config = neat3p.Config(
            neat3p.DefaultGenome, neat3p.DefaultReproduction,
            neat3p.DefaultSpeciesSet, neat3p.DefaultStagnation, pkg["config_path"],
        )
        return NEATNetWithFeatureAttention(
            pkg["winner_genome"], sd, ad, config, device_alias=device, encoder=enc, attn=attn
        ), "module"


# ---------------------------------------------------------------------------
# HyperNEAT
# ---------------------------------------------------------------------------


def _make_hyperneat_wrapper(in_c, hid_c, out_c, device):
    from neat3p.nn.composite import HyperNEATNet

    class _W:
        def __init__(self, net): self._net = net

        @classmethod
        def create(cls, genome, config, batch_size=1, use_current_activs=True, _dev=device):
            return cls(HyperNEATNet.create(genome, config, in_c, hid_c, out_c,
                                           batch_size=batch_size, device=_dev))

        def activate(self, inputs): return self._net.activate(inputs)
        def reset(self, batch_size=None): self._net.reset(batch_size)

    return _W


class _HyperNEATAdapter(ModelAdapter):
    kind = "hyper_neat"
    tunables = {}

    def build(self, task, env_id, seed, device, verbose, **tunables) -> NetBuild:
        if task.substrate is None:
            raise ValueError(f"Task '{task.name}' has no substrate; hyper_neat requires one.")
        in_c, hid_c, out_c = task.substrate()
        return NetBuild(_make_hyperneat_wrapper(in_c, hid_c, out_c, device), {}, {})

    def rebuild(self, pkg, device):
        import neat3p
        from neat3p.benchmarks.substrates import voxel_forage_substrate
        from neat3p.nn.composite import HyperNEATNet

        in_c, hid_c, out_c = voxel_forage_substrate()
        config = neat3p.Config(
            neat3p.DefaultGenome, neat3p.DefaultReproduction,
            neat3p.DefaultSpeciesSet, neat3p.DefaultStagnation, pkg["config_path"],
        )
        return HyperNEATNet.create(pkg["winner_genome"], config, in_c, hid_c, out_c, device=device), "recurrent"


# ---------------------------------------------------------------------------
# Adaptive HyperNEAT
# ---------------------------------------------------------------------------


def _make_adaptive_wrapper(in_c, hid_c, out_c, device):
    from neat3p.nn.composite import AdaptiveNet

    class _W:
        def __init__(self, net): self._net = net

        @classmethod
        def create(cls, genome, config, batch_size=1, use_current_activs=True, _dev=device):
            return cls(AdaptiveNet.create(genome, config,
                                          input_coords=in_c, hidden_coords=hid_c, output_coords=out_c,
                                          batch_size=batch_size, device=_dev))

        def activate(self, inputs): return self._net.activate(inputs)
        def reset(self, batch_size=None): self._net.reset()

    return _W


class _AdaptiveHyperNEATAdapter(ModelAdapter):
    kind = "adaptive_hyperneat"
    tunables = {}

    def build(self, task, env_id, seed, device, verbose, **tunables) -> NetBuild:
        if task.substrate is None:
            raise ValueError(f"Task '{task.name}' has no substrate; adaptive_hyperneat requires one.")
        in_c, hid_c, out_c = task.substrate()
        return NetBuild(_make_adaptive_wrapper(in_c, hid_c, out_c, device), {}, {})

    def rebuild(self, pkg, device):
        import neat3p
        from neat3p.benchmarks.substrates import voxel_forage_substrate
        from neat3p.nn.composite import AdaptiveNet

        in_c, hid_c, out_c = voxel_forage_substrate()
        config = neat3p.Config(
            neat3p.DefaultGenome, neat3p.DefaultReproduction,
            neat3p.DefaultSpeciesSet, neat3p.DefaultStagnation, pkg["config_path"],
        )
        return AdaptiveNet.create(pkg["winner_genome"], config,
                                   input_coords=in_c, hidden_coords=hid_c, output_coords=out_c,
                                   device=device), "recurrent"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RECURRENT_NET = _RecurrentNetAdapter()
FEATURE_ATTENTION = _FeatureAttentionAdapter()
HYPER_NEAT = _HyperNEATAdapter()
ADAPTIVE_HYPERNEAT = _AdaptiveHyperNEATAdapter()

MODELS: dict[str, ModelAdapter] = {
    m.kind: m for m in (RECURRENT_NET, FEATURE_ATTENTION, HYPER_NEAT, ADAPTIVE_HYPERNEAT)
}
