"""
Save / load trained benchmark winners so a network from any run can be replayed later.

A saved package is self-contained: the winner genome, the NEAT config path, the network
"kind", and (for feature-attention) the frozen encoder/attention weights. ``load_winner``
reconstructs the exact net purely from library components — no benchmark scripts needed —
and ``select_action`` / ``reset_net`` drive it for a rollout regardless of net style.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any

import neat3p

if TYPE_CHECKING:
    from benchmarks.runners.gym_eval import GymEvalResult

VALID_KINDS = ("recurrent_net", "feature_attention", "hyper_neat", "adaptive_hyperneat")


def _env_tag(env_id: str) -> str:
    return "noscent" if "NoScent" in env_id else "scent"


def save_winner(
    save_dir: str,
    kind: str,
    result: GymEvalResult,
    env_id: str,
    seed: int,
    config_path: str,
    feature_dim: int | None = None,
    encoder: Any = None,
    attn: Any = None,
    generation: int | None = None,
) -> str:
    """Pickle a self-contained winner package and return its path."""
    assert kind in VALID_KINDS, f"unknown kind {kind}"
    os.makedirs(save_dir, exist_ok=True)
    pkg = {
        "kind": kind,
        "env_id": env_id,
        "config_path": os.path.abspath(config_path),
        "state_dim": int(result.state_dim),
        "action_dim": int(result.action_dim),
        "winner_genome": result.winner,
        "fitness": float(result.winner.fitness),
        "seed": int(seed),
        "generation": generation,
    }
    if kind == "feature_attention":
        pkg["feature_dim"] = int(feature_dim)
        pkg["encoder_state_dict"] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
        pkg["attn_state_dict"] = {k: v.detach().cpu() for k, v in attn.state_dict().items()}

    path = os.path.join(save_dir, f"{kind}_{_env_tag(env_id)}_seed{seed}.pkl")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(pkg, f)
    os.replace(tmp, path)
    return path


def _build_config(config_path: str) -> neat3p.Config:
    return neat3p.Config(
        neat3p.DefaultGenome,
        neat3p.DefaultReproduction,
        neat3p.DefaultSpeciesSet,
        neat3p.DefaultStagnation,
        config_path,
    )


def load_winner(path: str, device: str = "cuda:0") -> tuple[dict, Any, str]:
    """Return (package_dict, net, style). style is 'module' or 'recurrent'."""
    with open(path, "rb") as f:
        pkg = pickle.load(f)

    kind = pkg["kind"]
    assert kind in VALID_KINDS, f"unknown kind {kind!r}"

    config = _build_config(pkg["config_path"])
    sd, ad = pkg["state_dim"], pkg["action_dim"]

    if kind == "recurrent_net":
        from neat3p.nn.composite import NEATRecurrentNet

        net = NEATRecurrentNet(pkg["winner_genome"], sd, ad, config, device_alias=device)
        return pkg, net, "module"

    if kind == "feature_attention":
        from neat3p.nn.composite import NEATNetWithFeatureAttention
        from neat3p.nn.modules.attention import FeatureAttention
        from neat3p.nn.modules.encoders import SimpleEncoder

        fd = pkg["feature_dim"]
        enc = SimpleEncoder(sd, fd, device=device)
        enc.load_state_dict(pkg["encoder_state_dict"])
        enc.eval()
        attn = FeatureAttention(input_dim=fd, device=device)
        attn.load_state_dict(pkg["attn_state_dict"])
        attn.eval()
        net = NEATNetWithFeatureAttention(
            pkg["winner_genome"], sd, ad, config, device_alias=device, encoder=enc, attn=attn
        )
        return pkg, net, "module"

    if kind == "hyper_neat":
        from neat3p.gym_envs.substrates import voxel_forage_substrate
        from neat3p.nn.composite import HyperNEATNet

        in_c, hid_c, out_c = voxel_forage_substrate()
        net = HyperNEATNet.create(pkg["winner_genome"], config, in_c, hid_c, out_c, device=device)
        return pkg, net, "recurrent"

    if kind == "adaptive_hyperneat":
        from neat3p.gym_envs.substrates import voxel_forage_substrate
        from neat3p.nn.composite import AdaptiveNet

        in_c, hid_c, out_c = voxel_forage_substrate()
        net = AdaptiveNet.create(
            pkg["winner_genome"],
            config,
            input_coords=in_c,
            hidden_coords=hid_c,
            output_coords=out_c,
            device=device,
        )
        return pkg, net, "recurrent"

    raise ValueError(f"Unhandled kind {kind!r}")


def reset_net(net: Any, style: str) -> None:
    if style == "module":
        net.reset(batch_size=1)
    else:
        net.reset()


def select_action(net: Any, obs: Any, style: str) -> int:
    import torch

    if style == "module":
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(net(obs_t).argmax(dim=1).item())
    out = net.activate([obs.tolist()])
    return int(out[0].argmax().item())


def list_winners(output_dir: str) -> list[str]:
    if not os.path.isdir(output_dir):
        return []
    return sorted(os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".pkl"))
