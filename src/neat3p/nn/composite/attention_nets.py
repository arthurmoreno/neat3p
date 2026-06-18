"""
Attention-augmented NEAT architectures.

NEATNetWithAttention     — RecurrentNet + SimpleAttention pooled to an action head
NEATTemporalAttentionNet — processes a fixed-length history of observations through attention
"""

import torch
import torch.nn as nn
from torch import Tensor

import neat3p.genome
from neat3p.nn.modules.attention import SimpleAttention
from neat3p.nn.phenotypes.recurrent_net import RecurrentNet


class NEATNetWithAttention(nn.Module):
    """RecurrentNet controller with a SimpleAttention pooling head."""

    def __init__(
        self,
        genome: neat3p.genome.DefaultGenome,
        state_dim: int,
        action_dim: int,
        config,
        device_alias: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device_alias)
        self.genome = genome
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = RecurrentNet.create(genome, config)
        self.attn = SimpleAttention(state_dim, self.device)
        self.fc = nn.Linear(state_dim, action_dim).to(self.device)

    def forward(self, state: Tensor) -> Tensor:
        state = state.to(self.device)
        net_out = self.net.activate(state)
        attn_out = self.attn(net_out)
        context = attn_out.mean(dim=1)
        return self.fc(context)


class NEATTemporalAttentionNet(nn.Module):
    """Processes a fixed-length history of observations through NEAT + temporal attention."""

    def __init__(
        self,
        genome: neat3p.genome.DefaultGenome,
        state_dim: int,
        action_dim: int,
        config,
        history_length: int = 10,
        device_alias: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device_alias)
        self.history_length = history_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.genome = genome
        self.net = RecurrentNet.create(genome, config).to(self.device)
        self.attn = SimpleAttention(input_dim=state_dim, device=self.device)
        self.fc = nn.Linear(state_dim, action_dim).to(self.device)

    def forward(self, state_sequence: Tensor) -> Tensor:
        """
        state_sequence: (B, L, state_dim) — last L observations per creature.
        returns: (B, action_dim)
        """
        state_sequence = state_sequence.to(self.device)
        batch_size, seq_len, _ = state_sequence.shape
        flat_states = state_sequence.reshape(batch_size * seq_len, self.state_dim)
        flat_features = self.net.activate(flat_states)
        features = flat_features.reshape(batch_size, seq_len, self.state_dim)
        attn_out = self.attn(features)
        context = attn_out.mean(dim=1)
        return self.fc(context)
