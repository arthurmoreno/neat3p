"""Composite architectures: encoder + attention + NEAT controller.

Each architecture lives in its own module:
  neat_base.py         — NEATNetBase
  neat_recurrent.py    — NEATRecurrentNet (plain RecurrentNet controller, game contract)
  feature_attention.py — NEATNetWithFeatureAttention
  attention_nets.py    — NEATNetWithAttention, NEATTemporalAttentionNet
  hyper_neat.py        — HyperNEATNet, HyperNEATLinearNet (plain HyperNEAT, fixed weights)
  adaptive.py          — AdaptiveNet, AdaptiveLinearNet (Adaptive HyperNEAT, Hebbian plasticity)
"""

from .adaptive import AdaptiveLinearNet, AdaptiveNet
from .attention_nets import NEATNetWithAttention, NEATTemporalAttentionNet
from .feature_attention import NEATNetWithFeatureAttention
from .hyper_neat import HyperNEATLinearNet, HyperNEATNet, make_grid_coords
from .neat_base import NEATNetBase
from .neat_recurrent import NEATRecurrentNet

__all__ = [
    "NEATNetWithFeatureAttention",
    "NEATNetBase",
    "NEATRecurrentNet",
    "NEATNetWithAttention",
    "NEATTemporalAttentionNet",
    "HyperNEATNet",
    "HyperNEATLinearNet",
    "make_grid_coords",
    "AdaptiveNet",
    "AdaptiveLinearNet",
]
