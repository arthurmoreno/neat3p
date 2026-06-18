"""
Plain HyperNEAT phenotypes — CPPN-encoded networks with FIXED weights.

HyperNEATNet       — CPPN → fixed input→hidden→output feed-forward substrate net
HyperNEATLinearNet — CPPN → fixed input→output substrate net (no hidden layer)

This is the static counterpart to the ``adaptive`` module: where AdaptiveNet /
AdaptiveLinearNet keep updating their weights each step via a learned ``delta_w``
CPPN node (Adaptive HyperNEAT), the nets here query the CPPN **once** to build a
weight matrix over the substrate geometry and then hold it fixed for the whole
episode. The CPPN is the canonical HyperNEAT mapping:

    (x_in, y_in, x_out, y_out) → connection weight

so the genome only needs 4 CPPN inputs (the substrate coordinates) and one weight
output per connection layer — no ``pre`` / ``post`` / ``w`` synaptic-activity inputs.

Substrate geometry is supplied as coordinate lists ``[[x, y], ...]`` for the input,
hidden and output layers. ``make_grid_coords`` builds an evenly spaced 1-D row.
"""

import torch

from neat3p.nn.modules.activations import tanh_activation
from neat3p.nn.phenotypes.cppn import clamp_weights_, create_cppn, get_coord_inputs


def make_grid_coords(dim: int, y_value: float = 0.0) -> list:
    """Evenly spaced coordinates on the line y = ``y_value``, x ∈ [-1, 1]."""
    if dim == 1:
        return [[0.0, y_value]]
    return [[-1 + 2 * i / (dim - 1), y_value] for i in range(dim)]


class HyperNEATNet:
    """
    Plain HyperNEAT: a CPPN paints a **fixed** input→hidden→output feed-forward net
    over the substrate geometry. Weights are queried once at construction and never
    change during the episode (no plasticity).

    CPPN outputs: ``w_ih`` (input→hidden), ``b_h`` (hidden bias),
    ``w_ho`` (hidden→output), ``b_o`` (output bias).
    """

    def __init__(
        self,
        w_ih_node,
        b_h_node,
        w_ho_node,
        b_o_node,
        input_coords,
        hidden_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        batch_size=1,
        device="cuda:0",
    ):
        self.w_ih_node = w_ih_node
        self.b_h_node = b_h_node
        self.w_ho_node = w_ho_node
        self.b_o_node = b_o_node
        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(input_coords, dtype=torch.float32, device=device)
        self.n_hidden = len(hidden_coords)
        self.hidden_coords = torch.tensor(hidden_coords, dtype=torch.float32, device=device)
        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(output_coords, dtype=torch.float32, device=device)
        self.weight_threshold = weight_threshold
        self.weight_max = weight_max
        self.activation = activation
        self.batch_size = batch_size
        self.device = device
        self.reset()

    def _get_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)
        weights = w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in)
        clamp_weights_(weights, self.weight_threshold, self.weight_max)
        return weights

    def reset(self, batch_size=None):
        """Build the fixed substrate weights from the CPPN.

        Weights are batch-independent (shared across the batch via broadcasting), so the
        optional ``batch_size`` argument exists only for interface parity with the
        recurrent-style nets and is otherwise ignored.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        with torch.no_grad():
            bias_coords = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
            self.input_to_hidden = self._get_weights(self.input_coords, self.hidden_coords, self.w_ih_node)
            self.bias_hidden = self._get_weights(bias_coords, self.hidden_coords, self.b_h_node)
            self.hidden_to_output = self._get_weights(self.hidden_coords, self.output_coords, self.w_ho_node)
            self.bias_output = self._get_weights(bias_coords, self.output_coords, self.b_o_node)

    def activate(self, inputs):
        """inputs: (batch_size, n_inputs) → (batch_size, n_outputs)"""
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(2)
            hidden = self.activation(self.input_to_hidden.matmul(inputs) + self.bias_hidden)
            outputs = self.activation(self.hidden_to_output.matmul(hidden) + self.bias_output)
        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        hidden_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        batch_size=1,
        device="cuda:0",
    ):
        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out"],
            ["w_ih", "b_h", "w_ho", "b_o"],
        )
        return HyperNEATNet(
            *nodes,
            input_coords,
            hidden_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            batch_size=batch_size,
            device=device,
        )


class HyperNEATLinearNet:
    """
    Plain HyperNEAT with no hidden layer: a CPPN paints a **fixed** input→output
    weight matrix over the substrate geometry. The minimal HyperNEAT phenotype.

    CPPN outputs: ``w`` (input→output), ``b_o`` (output bias).
    """

    def __init__(
        self,
        w_node,
        b_o_node,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        batch_size=1,
        device="cuda:0",
    ):
        self.w_node = w_node
        self.b_o_node = b_o_node
        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(input_coords, dtype=torch.float32, device=device)
        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(output_coords, dtype=torch.float32, device=device)
        self.weight_threshold = weight_threshold
        self.weight_max = weight_max
        self.activation = activation
        self.batch_size = batch_size
        self.device = device
        self.reset()

    def _get_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)
        weights = w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in)
        clamp_weights_(weights, self.weight_threshold, self.weight_max)
        return weights

    def reset(self, batch_size=None):
        """Build the fixed input→output weights from the CPPN (batch-independent)."""
        if batch_size is not None:
            self.batch_size = batch_size
        with torch.no_grad():
            bias_coords = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
            self.input_to_output = self._get_weights(self.input_coords, self.output_coords, self.w_node)
            self.bias_output = self._get_weights(bias_coords, self.output_coords, self.b_o_node)

    def activate(self, inputs):
        """inputs: (batch_size, n_inputs) → (batch_size, n_outputs)"""
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(2)
            outputs = self.activation(self.input_to_output.matmul(inputs) + self.bias_output)
        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        state_dim,
        action_dim,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        batch_size=1,
        device="cuda:0",
    ):
        input_coords = make_grid_coords(state_dim, y_value=0.5)
        output_coords = make_grid_coords(action_dim, y_value=-0.5)
        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out"],
            ["w", "b_o"],
        )
        return HyperNEATLinearNet(
            *nodes,
            input_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            batch_size=batch_size,
            device=device,
        )
