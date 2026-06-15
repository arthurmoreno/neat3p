"""
NEATNetWithFeatureAttention — encoder + feature-attention gate + NEAT controller.

Three-stage inference pipeline:
    raw perception → SimpleEncoder → FeatureAttention gate → NEAT RecurrentNet → action
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim

import neat3p.genome
from neat3p.nn.modules.attention import FeatureAttention
from neat3p.nn.modules.encoders import SimpleEncoder
from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

logger = logging.getLogger(__name__)


class NEATNetWithFeatureAttention(nn.Module):
    """
    Three-stage inference pipeline:
      raw perception → SimpleEncoder → FeatureAttention gate → NEAT RecurrentNet → action

    The encoder and attention are initialised from a genome (SimpleEncoderGenome /
    FeatureAttentionGenome) stored in the BrainGenome.  The NEAT controller is the only
    component that evolves across generations; the encoder/attention should be frozen
    (or trained offline — see familyB-plan.md) so that controller heritability is not
    broken by per-child random projections.
    """

    def __init__(
        self,
        genome: neat3p.genome.DefaultGenome,
        state_dim: int,
        action_dim: int,
        config,
        device_alias: str = "cuda",
        max_experience_size: int = 100,
        feature_dim=None,
        encoder: nn.Module = None,
        attn: nn.Module = None,
        encoder_state_dict=None,
        attn_state_dict=None,
        freeze_encoder: bool = False,
    ):
        """
        feature_dim controls the encoder bottleneck:
          - None (default): max(1, state_dim // 10)  — the lifesim ratio for large perception vectors
          - int: use that value directly (e.g. feature_dim=4 for CartPole, no compression)
          - callable: called as feature_dim(state_dim) — e.g. lambda s: max(2, s // 2)

        encoder / attn: pre-built, shared front-end modules.  Pass these when running NEAT
          evolution so every genome in every generation sees the same fixed projection.
          The modules are moved to device and frozen automatically; feature_dim is inferred
          from the encoder's output layer.  encoder_state_dict / freeze_encoder are ignored
          when a pre-built encoder is supplied.

        encoder_state_dict / attn_state_dict: load pre-trained weights into a freshly
          constructed encoder / FeatureAttention (ignored when encoder/attn are supplied).

        freeze_encoder: freeze a freshly constructed encoder + attention in-place.
        """
        super(NEATNetWithFeatureAttention, self).__init__()
        self.device = torch.device(device_alias)
        self.genome = genome
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_rate = 0.125

        if encoder is not None:
            # Shared front-end path: move to device, freeze, skip compilation.
            self.encoder = encoder.to(self.device).eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            # Infer feature_dim from the encoder's final linear layer.
            self.feature_dim = next(
                reversed([m for m in self.encoder.modules() if isinstance(m, nn.Linear)])
            ).out_features
            self.compiled_encoder = self.encoder
        else:
            if feature_dim is None:
                self.feature_dim = max(1, state_dim // 10)
            elif callable(feature_dim):
                self.feature_dim = feature_dim(state_dim)
            else:
                self.feature_dim = int(feature_dim)
            self.encoder = SimpleEncoder(self.state_dim, self.feature_dim, device=str(self.device))
            if encoder_state_dict is not None:
                self.encoder.load_state_dict(encoder_state_dict)
            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
            self._compile_encoder()

        if attn is not None:
            self.feature_attn = attn.to(self.device).eval()
            for p in self.feature_attn.parameters():
                p.requires_grad_(False)
            self.compiled_feature_attn = self.feature_attn
        else:
            self.feature_attn = FeatureAttention(input_dim=self.feature_dim, device=str(self.device))
            if attn_state_dict is not None:
                self.feature_attn.load_state_dict(attn_state_dict)
            if freeze_encoder:
                for p in self.feature_attn.parameters():
                    p.requires_grad_(False)
            self._compile_feature_attn()

        self.net = RecurrentNet.create(genome, config)

        owned_params = (
            ([] if encoder is not None else list(self.compiled_encoder.parameters()))
            + ([] if attn is not None else list(self.compiled_feature_attn.parameters()))
        )
        if owned_params:
            self.optimizer = optim.Adam(owned_params, lr=1e-3)

        self.states = torch.zeros((1, self.state_dim), device=self.device)
        self.actions = torch.zeros((1, 1), device=self.device)
        self.rewards = torch.zeros((1, 1), device=self.device)
        self.max_experience_size = max_experience_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        with torch.no_grad():
            features = self.compiled_encoder(state)
            filtered_features = self.compiled_feature_attn(features)
            output = self.net.activate(filtered_features)
        return output

    def reset(self, batch_size: int = 1) -> None:
        """Reset the inner RecurrentNet hidden state between episodes."""
        self.net.reset(batch_size=batch_size)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        if torch.rand(1, device=self.device) < self.explore_rate:
            return torch.randint(self.action_dim, size=(batch_size, 1), device=self.device)
        return self.forward(state).argmax(dim=1, keepdim=True)

    def _compile_encoder(self):
        if not hasattr(self, "compiled_encoder"):
            try:
                self.encoder = self.encoder.cuda()
                self.compiled_encoder = torch.jit.script(self.encoder)
            except Exception as e:
                logger.warning("Could not TorchScript encoder: %s", e)
                self.compiled_encoder = self.encoder

    def _compile_feature_attn(self):
        if not hasattr(self, "compiled_feature_attn"):
            try:
                self.feature_attn = self.feature_attn.cuda()
                self.compiled_feature_attn = torch.jit.script(self.feature_attn)
            except Exception as e:
                logger.warning("Could not TorchScript feature_attn: %s", e)
                self.compiled_feature_attn = self.feature_attn
