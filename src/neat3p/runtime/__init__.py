"""Torch-free runtime for NEAT phenotypes (importing this package never imports torch)."""

from neat3p.runtime.compiled import CompiledRecurrentNet, compile_recurrent
from neat3p.runtime.layout import BLOCK_NAMES, CooBlock, RecurrentLayout, build_recurrent_layout
from neat3p.runtime.native import NativeRecurrentNet
from neat3p.runtime.reference import ReferenceRecurrentNet

__all__ = [
    "BLOCK_NAMES",
    "CompiledRecurrentNet",
    "CooBlock",
    "NativeRecurrentNet",
    "RecurrentLayout",
    "ReferenceRecurrentNet",
    "build_recurrent_layout",
    "compile_recurrent",
]
