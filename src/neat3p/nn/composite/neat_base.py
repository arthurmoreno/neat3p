"""
NEATNetBase — minimal NEAT phenotype wrapper with a QNet-compatible interface.
"""

import torch
import torch.nn as nn
from torch import Tensor

import neat3p.genome
from neat3p.nn.phenotypes.recurrent_net import RecurrentNet


class NEATNetBase(nn.Module):
    """
    Minimal NEAT genome/network wrapper with a QNet-compatible interface.

      forward(state)   → activations
      get_action(state) → epsilon-greedy action index
    """

    def __init__(
        self,
        genome: neat3p.genome.DefaultGenome,
        state_dim: int,
        action_dim: int,
        config,
        device_alias: str = "cuda",
    ):
        super().__init__()
        self.genome = genome
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device_alias)

        self.net = RecurrentNet.create(genome, config)
        self.explore_rate = 0.125

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / (self.state_std + 1e-6)

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg

    def forward(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            return self.net.activate(state)

    def get_action(self, state: Tensor) -> Tensor:
        batch_size = state.shape[0]
        if torch.rand(1) < self.explore_rate:
            return torch.randint(self.action_dim, size=(batch_size, 1), device=self.device)
        return self.forward(state).argmax(dim=1, keepdim=True)

    def clone_net(self):
        import copy

        cloned_genome = copy.deepcopy(self.genome)
        return NEATNetBase(cloned_genome, self.state_dim, self.action_dim, self.config, str(self.device))
