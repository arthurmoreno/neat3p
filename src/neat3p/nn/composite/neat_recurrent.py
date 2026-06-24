"""
NEATRecurrentNet — a NEAT RecurrentNet controller behind the game brain contract.
"""

import torch
import torch.nn as nn

import neat3p.genome
from neat3p.nn.phenotypes.recurrent_net import RecurrentNet


class NEATRecurrentNet(nn.Module):
    """
    The simplest game-contract brain net: a NEAT-evolved ``RecurrentNet`` phenotype run
    directly on the raw observation, with no encoder / attention front-end.

    It deliberately shares the contract of ``NEATNetWithFeatureAttention`` so the two are
    interchangeable as Beast brains (the front-end is the only difference):

        __init__(genome, state_dim, action_dim, config, device_alias=...)
        forward(state)      -> (batch, action_dim)
        reset(batch_size=1) -> clears the recurrent hidden state between episodes
        get_action(state)   -> epsilon-greedy action index

    This is the class the game would plug in for a "plain NEAT" brain, and the one the
    CartPole recurrent-net benchmark exercises — so the benchmark tests what ships.
    """

    def __init__(
        self,
        genome: neat3p.genome.DefaultGenome,
        state_dim: int,
        action_dim: int,
        config,
        device_alias: str = "cuda",
        use_current_activs: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device_alias)
        self.genome = genome
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_rate = 0.125
        self.net = RecurrentNet.create(
            genome, config, use_current_activs=use_current_activs, device=str(self.device)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        with torch.no_grad():
            return self.net.activate(state)

    def reset(self, batch_size: int = 1) -> None:
        """Reset the inner RecurrentNet hidden state between episodes."""
        self.net.reset(batch_size=batch_size)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        if torch.rand(1, device=self.device) < self.explore_rate:
            return torch.randint(self.action_dim, size=(batch_size, 1), device=self.device)
        return self.forward(state).argmax(dim=1, keepdim=True)
