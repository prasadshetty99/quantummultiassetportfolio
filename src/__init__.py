"""
Quantum Multi-Asset Portfolio Optimization Framework

A hybrid quantum-classical optimization framework for multi-asset portfolio construction
that leverages quantum algorithms (QAOA, VQE, HHL) and classical optimization techniques
(MAD/CVaR, Linear Programming) to solve complex portfolio allocation problems.
"""

__version__ = "0.1.0"
__author__ = "Prasad Shetty"

from . import core
from . import quantum
from . import classical
from . import data
from . import utils

__all__ = ["core", "quantum", "classical", "data", "utils"]
