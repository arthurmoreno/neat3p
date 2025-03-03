"""A NEAT (NeuroEvolution of Augmenting Topologies) implementation"""

from ._neat3p import DefaultNodeGene, GenomeConfig
from .checkpoint import Checkpointer

# import .nn as nn
# import .ctrnn as ctrnn
# import .iznn as iznn
# import .distributed as distributed
from .config import Config
from .distributed import DistributedEvaluator, host_is_local
from .genome import DefaultGenome
from .parallel import ParallelEvaluator
from .population import CompleteExtinctionException, Population
from .reporting import StdOutReporter
from .reproduction import DefaultReproduction
from .species import DefaultSpeciesSet
from .stagnation import DefaultStagnation
from .statistics import StatisticsReporter
from .threaded import ThreadedEvaluator
from .utils import this_is_neat

__all__ = ["GenomeConfig", "DefaultNodeGene"]
