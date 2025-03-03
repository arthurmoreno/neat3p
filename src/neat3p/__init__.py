"""A NEAT (NeuroEvolution of Augmenting Topologies) implementation"""
from ._neat3p import GenomeConfig, DefaultNodeGene

from .utils import this_is_neat

# import .nn as nn
# import .ctrnn as ctrnn
# import .iznn as iznn
# import .distributed as distributed

from .config import Config
from .population import Population, CompleteExtinctionException
from .genome import DefaultGenome
from .reproduction import DefaultReproduction
from .stagnation import DefaultStagnation
from .reporting import StdOutReporter
from .species import DefaultSpeciesSet
from .statistics import StatisticsReporter
from .parallel import ParallelEvaluator
from .distributed import DistributedEvaluator, host_is_local
from .threaded import ThreadedEvaluator
from .checkpoint import Checkpointer

__all__ = ["GenomeConfig", "DefaultNodeGene"]