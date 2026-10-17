"""Native (Eigen) evaluator of a recurrent NEAT phenotype."""

from __future__ import annotations

import numpy as np

from neat3p.runtime.layout import RecurrentLayout

_DTYPES = ("float64", "float32")


class CompiledRecurrentNet:
    """A layout compiled into the C++ kernel; holds the recurrent state between ``activate`` calls.

    Not thread-safe per instance; separate instances are independent. ``activate`` releases the GIL.
    """

    def __init__(self, layout: RecurrentLayout, dtype: str = "float64"):
        from neat3p import _neat3p

        if dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {dtype!r}; expected one of {_DTYPES}")
        kernel_cls = _neat3p.RecurrentKernelF64 if dtype == "float64" else _neat3p.RecurrentKernelF32

        dense = layout.to_dense()
        empty = np.zeros((layout.n_hidden, 0))

        def block(name: str) -> np.ndarray:
            return np.ascontiguousarray(dense[name]) if name in dense else np.zeros(layout.block_shape(name))

        self._kernel = kernel_cls(
            block("input_to_hidden") if layout.n_hidden else empty.reshape(0, layout.n_inputs),
            block("hidden_to_hidden"),
            block("output_to_hidden"),
            block("input_to_output"),
            block("hidden_to_output"),
            block("output_to_output"),
            layout.hidden_biases,
            layout.hidden_responses,
            layout.output_biases,
            layout.output_responses,
            layout.use_current_activs,
            layout.n_internal_steps,
        )
        self.dtype = dtype
        self.layout = layout

    @property
    def n_inputs(self) -> int:
        return self._kernel.n_inputs

    @property
    def n_hidden(self) -> int:
        return self._kernel.n_hidden

    @property
    def n_outputs(self) -> int:
        return self._kernel.n_outputs

    @property
    def hidden_state(self) -> np.ndarray:
        return self._kernel.hidden_state

    @property
    def output_state(self) -> np.ndarray:
        return self._kernel.output_state

    def reset(self) -> None:
        self._kernel.reset()

    def activate(self, x) -> np.ndarray:
        """x: (n_inputs,) or (1, n_inputs), float32 or float64 -> (n_outputs,) float64."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise ValueError(f"expected input of shape (n_inputs,) or (1, n_inputs), got {arr.shape}")
        return self._kernel.activate(np.ascontiguousarray(arr))


def compile_recurrent(layout: RecurrentLayout, dtype: str = "float64") -> CompiledRecurrentNet:
    return CompiledRecurrentNet(layout, dtype=dtype)
