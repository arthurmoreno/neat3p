"""
NEAT networks that combine a PyTorch front-end with a NEAT controller.

NEATNetBase                 — minimal NEAT phenotype wrapper (QNet-compatible interface)
NEATNetWithFeatureAttention — encoder + feature-attention gate + NEAT controller
NEATNetWithAttention        — RecurrentNet + SimpleAttention pooled to action head
NEATTemporalAttentionNet    — processes a history of observations through attention
AdaptiveNet                 — CPPN-generated recurrent net with online Hebbian weight updates
AdaptiveLinearNet           — CPPN-generated linear net with online Hebbian weight updates
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import neat3p.genome
from neat3p.nn.modules.activations import identity_activation, tanh_activation
from neat3p.nn.modules.attention import FeatureAttention, SimpleAttention
from neat3p.nn.modules.encoders import SimpleEncoder
from neat3p.nn.phenotypes.cppn import clamp_weights_, create_cppn, get_coord_inputs
from neat3p.nn.phenotypes.recurrent_net import RecurrentNet

logger = logging.getLogger(__name__)


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
        encoder_state_dict=None,
        attn_state_dict=None,
        freeze_encoder: bool = False,
    ):
        """
        feature_dim controls the encoder bottleneck:
          - None (default): max(1, state_dim // 10)  — the lifesim ratio for large perception vectors
          - int: use that value directly (e.g. feature_dim=4 for CartPole, no compression)
          - callable: called as feature_dim(state_dim) — e.g. lambda s: max(2, s // 2)

        encoder_state_dict / attn_state_dict: load pre-trained weights into the encoder /
          FeatureAttention before the NEAT controller is built. Pass state dicts from a
          pre-training phase so the front-end starts from a meaningful representation.

        freeze_encoder: if True, sets requires_grad=False on all encoder + attention params
          so Adam does not update them during joint training.
        """
        super(NEATNetWithFeatureAttention, self).__init__()
        self.device = torch.device(device_alias)
        self.genome = genome
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_rate = 0.125

        if feature_dim is None:
            self.feature_dim = max(1, state_dim // 10)
        elif callable(feature_dim):
            self.feature_dim = feature_dim(state_dim)
        else:
            self.feature_dim = int(feature_dim)

        self.encoder = SimpleEncoder(self.state_dim, self.feature_dim, device=str(self.device))
        self.feature_attn = FeatureAttention(input_dim=self.feature_dim, device=str(self.device))

        if encoder_state_dict is not None:
            self.encoder.load_state_dict(encoder_state_dict)
        if attn_state_dict is not None:
            self.feature_attn.load_state_dict(attn_state_dict)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.feature_attn.parameters():
                p.requires_grad_(False)

        self.net = RecurrentNet.create(genome, config)
        self._compile_encoder()
        self._compile_feature_attn()

        self.optimizer = optim.Adam(
            list(self.compiled_encoder.parameters()) + list(self.compiled_feature_attn.parameters()), lr=1e-3
        )

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


# ---------------------------------------------------------------------------
# Attention-augmented architectures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CPPN-based adaptive architectures
# ---------------------------------------------------------------------------


def _get_coords(dim: int, y_value: float = 0.0) -> list:
    if dim == 1:
        return [[0.0, y_value]]
    return [[-1 + 2 * i / (dim - 1), y_value] for i in range(dim)]


class AdaptiveNet:
    """
    CPPN-generated recurrent network with online Hebbian weight updates.

    Weights are initialised from a CPPN genome and updated each step via a
    learned delta_w CPPN node — a form of meta-learning in weight space.
    """

    def __init__(
        self,
        w_ih_node,
        b_h_node,
        w_hh_node,
        b_o_node,
        w_ho_node,
        delta_w_node,
        input_coords,
        hidden_coords,
        output_coords,
        weight_threshold=0.2,
        activation=tanh_activation,
        batch_size=1,
        device="cuda:0",
    ):
        self.w_ih_node = w_ih_node
        self.b_h_node = b_h_node
        self.w_hh_node = w_hh_node
        self.b_o_node = b_o_node
        self.w_ho_node = w_ho_node
        self.delta_w_node = delta_w_node
        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(input_coords, dtype=torch.float32, device=device)
        self.n_hidden = len(hidden_coords)
        self.hidden_coords = torch.tensor(hidden_coords, dtype=torch.float32, device=device)
        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(output_coords, dtype=torch.float32, device=device)
        self.weight_threshold = weight_threshold
        self.activation = activation
        self.batch_size = batch_size
        self.device = device
        self.reset()

    def _get_init_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)
        n_in, n_out = len(in_coords), len(out_coords)
        zeros = torch.zeros((n_out, n_in), dtype=torch.float32, device=self.device)
        weights = w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in, pre=zeros, post=zeros, w=zeros)
        clamp_weights_(weights, self.weight_threshold)
        return weights

    def reset(self):
        with torch.no_grad():
            self.input_to_hidden = self._get_init_weights(self.input_coords, self.hidden_coords, self.w_ih_node)
            bias_coords = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
            self.bias_hidden = (
                self._get_init_weights(bias_coords, self.hidden_coords, self.b_h_node)
                .unsqueeze(0)
                .expand(self.batch_size, self.n_hidden, 1)
            )
            self.hidden_to_hidden = (
                self._get_init_weights(self.hidden_coords, self.hidden_coords, self.w_hh_node)
                .unsqueeze(0)
                .expand(self.batch_size, self.n_hidden, self.n_hidden)
            )
            self.bias_output = self._get_init_weights(bias_coords, self.output_coords, self.b_o_node)
            self.hidden_to_output = self._get_init_weights(self.hidden_coords, self.output_coords, self.w_ho_node)
            self.hidden = torch.zeros((self.batch_size, self.n_hidden, 1), dtype=torch.float32)
            self.batched_hidden_coords = get_coord_inputs(
                self.hidden_coords, self.hidden_coords, batch_size=self.batch_size
            )

    def activate(self, inputs):
        """inputs: (batch_size, n_inputs) → (batch_size, n_outputs)"""
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(2)
            self.hidden = self.activation(
                self.input_to_hidden.matmul(inputs) + self.hidden_to_hidden.matmul(self.hidden) + self.bias_hidden
            )
            outputs = self.activation(self.hidden_to_output.matmul(self.hidden) + self.bias_output)
            hidden_outputs = self.hidden.expand(self.batch_size, self.n_hidden, self.n_hidden)
            hidden_inputs = hidden_outputs.transpose(1, 2)
            (x_out, y_out), (x_in, y_in) = self.batched_hidden_coords
            self.hidden_to_hidden += self.delta_w_node(
                x_out=x_out,
                y_out=y_out,
                x_in=x_in,
                y_in=y_in,
                pre=hidden_inputs,
                post=hidden_outputs,
                w=self.hidden_to_hidden,
            )
        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        hidden_coords,
        output_coords,
        weight_threshold=0.2,
        activation=tanh_activation,
        batch_size=1,
        device="cuda:0",
    ):
        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
            ["w_ih", "b_h", "w_hh", "b_o", "w_ho", "delta_w"],
        )
        return AdaptiveNet(
            *nodes,
            input_coords,
            hidden_coords,
            output_coords,
            weight_threshold=weight_threshold,
            activation=activation,
            batch_size=batch_size,
            device=device,
        )


class AdaptiveLinearNet:
    """
    CPPN-generated linear network with online Hebbian weight updates.

    Simpler than AdaptiveNet: no hidden layer, direct input→output with
    per-step weight updates from a CPPN delta_w node.
    """

    def __init__(
        self,
        delta_w_node,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):
        self.delta_w_node = delta_w_node
        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(input_coords, dtype=torch.float32, device=device)
        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(output_coords, dtype=torch.float32, device=device)
        self.weight_threshold = weight_threshold
        self.weight_max = weight_max
        self.activation = activation
        self.cppn_activation = cppn_activation
        self.batch_size = batch_size
        self.device = device
        self.reset()

    def _get_init_weights(self, in_coords, out_coords):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)
        n_in, n_out = len(in_coords), len(out_coords)
        zeros = torch.zeros((n_out, n_in), dtype=torch.float32, device=self.device)
        weights = self.cppn_activation(
            self.delta_w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in, pre=zeros, post=zeros, w=zeros)
        )
        clamp_weights_(weights, self.weight_threshold, self.weight_max)
        return weights

    def reset(self):
        with torch.no_grad():
            self.input_to_output = (
                self._get_init_weights(self.input_coords, self.output_coords)
                .unsqueeze(0)
                .expand(self.batch_size, self.n_outputs, self.n_inputs)
            )
            self.w_expressed = self.input_to_output != 0
            self.batched_coords = get_coord_inputs(self.input_coords, self.output_coords, batch_size=self.batch_size)

    def activate(self, inputs):
        """inputs: (batch_size, n_inputs) → (batch_size, n_outputs)"""
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(2)
            outputs = self.activation(self.input_to_output.matmul(inputs))
            input_activs = inputs.transpose(1, 2).expand(self.batch_size, self.n_outputs, self.n_inputs)
            output_activs = outputs.expand(self.batch_size, self.n_outputs, self.n_inputs)
            (x_out, y_out), (x_in, y_in) = self.batched_coords
            delta_w = self.cppn_activation(
                self.delta_w_node(
                    x_out=x_out,
                    y_out=y_out,
                    x_in=x_in,
                    y_in=y_in,
                    pre=input_activs,
                    post=output_activs,
                    w=self.input_to_output,
                )
            )
            self.input_to_output[self.w_expressed] += delta_w[self.w_expressed]
            clamp_weights_(self.input_to_output, weight_threshold=0.0, weight_max=self.weight_max)
        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        state_dim,
        action_dim,
        weight_threshold=0.2,
        weight_max=3.0,
        output_activation=None,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):
        input_coords = _get_coords(state_dim, y_value=0.5)
        output_coords = _get_coords(action_dim, y_value=-0.5)
        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
            ["delta_w"],
            output_activation=output_activation,
        )
        return AdaptiveLinearNet(
            nodes[0],
            input_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            cppn_activation=cppn_activation,
            batch_size=batch_size,
            device=device,
        )
