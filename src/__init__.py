"""Quantum Multi-Asset Portfolio Optimization Framework"""

__version__ = "0.1.0"
__author__ = "Prasad Shetty"
__email__ = "prasadshetty99@gmail.com"

from src.data_pipeline import DataPipeline
from src.portfolio import Portfolio
from src.quantum_optimizer import QuantumPortfolioOptimizer
from src.classical_optimizer import ClassicalOptimizer

__all__ = [
    "DataPipeline",
    "Portfolio",
    "QuantumPortfolioOptimizer",
    "ClassicalOptimizer",
]
