"""
CPPN-based adaptive architectures with online Hebbian weight updates.

AdaptiveNet       — CPPN-generated recurrent net; per-step delta_w updates on hidden→hidden
AdaptiveLinearNet — CPPN-generated linear net; per-step delta_w updates on input→output

Weights are initialised from a CPPN genome evaluated over substrate coordinates, then
updated each step via a learned ``delta_w`` CPPN node — a form of meta-learning in
weight space (Adaptive HyperNEAT).
"""

import torch

from neat3p.nn.modules.activations import identity_activation, tanh_activation
from neat3p.nn.phenotypes.cppn import clamp_weights_, create_cppn, get_coord_inputs


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
            self.hidden = torch.zeros((self.batch_size, self.n_hidden, 1), dtype=torch.float32, device=self.device)
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
