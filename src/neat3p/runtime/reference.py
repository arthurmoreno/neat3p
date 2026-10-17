"""Readable numpy evaluator of a recurrent NEAT phenotype.

Same math and operation order as ``RecurrentNet.activate``; it is the spec as code and the middle link of
the torch <-> numpy <-> native parity chain. Not a runtime path.
"""

from __future__ import annotations

import numpy as np

from neat3p.runtime.layout import RecurrentLayout


def sigmoid5(z: np.ndarray) -> np.ndarray:
    """``sigmoid(5 * z)``, the activation every node of the live brain uses."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-5.0 * z))


class ReferenceRecurrentNet:
    def __init__(self, layout: RecurrentLayout, batch_size: int = 1):
        self.layout = layout
        self.dense = layout.to_dense()
        self.reset(batch_size)

    def reset(self, batch_size: int = 1) -> None:
        lo = self.layout
        self.activs = np.zeros((batch_size, lo.n_hidden)) if lo.n_hidden > 0 else None
        self.outputs = np.zeros((batch_size, lo.n_outputs))

    def activate(self, inputs) -> np.ndarray:
        """inputs: (batch, n_inputs) -> (batch, n_outputs) float64."""
        lo, d = self.layout, self.dense
        x = np.asarray(inputs, dtype=np.float64)
        activs_for_output = self.activs
        with np.errstate(invalid="ignore"):
            if lo.n_hidden > 0:
                for _ in range(lo.n_internal_steps):
                    self.activs = sigmoid5(
                        lo.hidden_responses
                        * (
                            x @ d["input_to_hidden"].T
                            + self.activs @ d["hidden_to_hidden"].T
                            + self.outputs @ d["output_to_hidden"].T
                        )
                        + lo.hidden_biases
                    )
                if lo.use_current_activs:
                    activs_for_output = self.activs
            output_inputs = x @ d["input_to_output"].T + self.outputs @ d["output_to_output"].T
            if lo.n_hidden > 0:
                output_inputs = output_inputs + activs_for_output @ d["hidden_to_output"].T
            self.outputs = sigmoid5(lo.output_responses * output_inputs + lo.output_biases)
        return self.outputs
