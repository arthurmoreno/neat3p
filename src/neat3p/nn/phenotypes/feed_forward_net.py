import torch
import torch.nn as nn

from neat3p.graphs import feed_forward_layers


class TorchFeedForwardNetwork(nn.Module):
    """
    A PyTorch nn.Module version of a NEAT feed-forward network.

    Node evaluation tuple: (node, act_func, agg_func, bias, response, links)
    """

    def __init__(self, input_nodes, output_nodes, node_evals):
        super(TorchFeedForwardNetwork, self).__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.node_evals = node_evals

        all_node_ids = list(input_nodes) + list(output_nodes) + [ne[0] for ne in node_evals]
        self.max_index = max(all_node_ids) if all_node_ids else 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs[0]
        if inputs.size(0) != len(self.input_nodes):
            raise RuntimeError("Expected {} inputs, got {}".format(len(self.input_nodes), inputs.size(0)))

        device = inputs.device
        dtype = inputs.dtype

        node_values = torch.zeros(self.max_index + 1, device=device, dtype=dtype)

        for idx, node_id in enumerate(self.input_nodes):
            node_values[node_id] = inputs[idx]

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            link_values = []
            for i, weight in links:
                link_values.append(node_values[i] * weight)

            if link_values:
                s = agg_func(link_values)
            else:
                s = torch.tensor(0.0, device=device, dtype=dtype)

            node_values[node] = act_func(bias + response * s)

        outputs = torch.stack([node_values[i] for i in self.output_nodes])
        return outputs

    @staticmethod
    def create(genome, config):
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                links = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        links.append((inode, cg.weight))
                ng = genome.nodes[node]
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, links))

        return TorchFeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)
