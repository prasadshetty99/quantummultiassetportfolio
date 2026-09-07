"""
QUBO (Quadratic Unconstrained Binary Optimization) Formulation.

Convert portfolio optimization to QUBO for quantum annealing.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from ..core.portfolio import Portfolio
from ..core.objective import ObjectiveFunction, OptimizationObjective


class QUBOFormulation:
    """
    Convert portfolio optimization to QUBO form for quantum annealing.
    
    QUBO: minimize x^T Q x where x ∈ {0,1}^n
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize QUBO formulation.
        
        Args:
            portfolio: Portfolio object
        """
        self.portfolio = portfolio
        self.Q_matrix = None
        self.offset = 0.0
    
    def formulate(
        self,
        objective: ObjectiveFunction,
        penalty_weight: float = 1000.0
    ) -> Tuple[np.ndarray, float]:
        """
        Formulate portfolio optimization as QUBO.
        
        Args:
            objective: Optimization objective
            penalty_weight: Weight for constraint penalties
            
        Returns:
            Tuple of (Q_matrix, offset)
        """
        n_assets = self.portfolio.n_assets
        Q = np.zeros((n_assets, n_assets))
        
        # Objective terms: negative returns (to maximize)
        for i in range(n_assets):
            Q[i, i] -= self.portfolio.expected_returns[i]
        
        # Risk penalty: lambda * covariance
        Q += objective.risk_aversion * self.portfolio.covariance
        
        # Budget constraint penalty: if sum(x_i) != 1
        # Add penalty term: penalty_weight * (sum(x_i) - 1)^2
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    Q[i, i] += penalty_weight
                else:
                    Q[i, j] += 2 * penalty_weight
        
        # Offset from constraint
        offset = penalty_weight
        
        self.Q_matrix = Q
        self.offset = offset
        
        return Q, offset
    
    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO to Ising formulation.
        
        QUBO: x^T Q x  ->  Ising: -0.25 * s^T J s - 0.5 * h^T s
        where s_i = 2*x_i - 1 (maps {0,1} to {-1,+1})
        
        Returns:
            Tuple of (J_matrix, h_vector, offset)
        """
        if self.Q_matrix is None:
            raise ValueError("Must call formulate() first")
        
        n = self.Q_matrix.shape[0]
        Q = self.Q_matrix
        
        # Convert to Ising
        J = -Q / 4.0  # Coupling terms
        h = -np.diag(Q) / 2.0  # Local fields
        
        # Adjust offset
        offset = self.offset + np.sum(np.diag(Q)) / 4.0 + np.sum(np.triu(Q, k=1)) / 2.0
        
        return J, h, offset
    
    def get_summary(self) -> Dict:
        """Get QUBO formulation summary."""
        if self.Q_matrix is None:
            return {'status': 'Not formulated'}
        
        return {
            'num_variables': self.Q_matrix.shape[0],
            'Q_norm': float(np.linalg.norm(self.Q_matrix)),
            'Q_condition': float(np.linalg.cond(self.Q_matrix)),
            'offset': self.offset,
        }
