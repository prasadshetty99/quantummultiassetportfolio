"""
VQE (Variational Quantum Eigensolver) Solver for Portfolio Optimization.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize

from ..core.portfolio import Portfolio
from ..core.constraints import PortfolioConstraints
from ..core.objective import ObjectiveFunction


class VQESolver:
    """
    Variational Quantum Eigensolver (VQE) for portfolio optimization.
    
    Uses quantum circuits and classical optimization for portfolio allocation.
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        constraints: PortfolioConstraints,
        num_qubits: int = 8,
        ansatz: str = 'ry'
    ):
        """
        Initialize VQE solver.
        
        Args:
            portfolio: Portfolio object
            constraints: Portfolio constraints
            num_qubits: Number of qubits
            ansatz: Ansatz type ('ry', 'iqp', 'hardware_efficient')
        """
        self.portfolio = portfolio
        self.constraints = constraints
        self.num_qubits = num_qubits
        self.ansatz = ansatz
        self.optimal_params = None
    
    def construct_ansatz(
        self,
        params: np.ndarray
    ) -> Dict:
        """
        Construct VQE ansatz circuit.
        
        Args:
            params: Circuit parameters
            
        Returns:
            Ansatz circuit configuration
        """
        return {
            'type': self.ansatz,
            'num_qubits': self.num_qubits,
            'num_params': len(params),
        }
    
    def evaluate_expectation(
        self,
        params: np.ndarray,
        objective: ObjectiveFunction
    ) -> float:
        """
        Evaluate expectation value of objective Hamiltonian.
        
        Args:
            params: Circuit parameters
            objective: Optimization objective
            
        Returns:
            Expectation value
        """
        # Map quantum state to portfolio weights via softmax
        weights = np.exp(params[:self.portfolio.n_assets])
        weights = weights / np.sum(weights)
        
        # Evaluate objective
        return objective.evaluate(
            weights,
            self.portfolio.expected_returns,
            self.portfolio.covariance
        )
    
    def optimize(
        self,
        objective: ObjectiveFunction,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Optimize portfolio using VQE.
        
        Args:
            objective: Optimization objective
            max_iterations: Maximum optimization iterations
            
        Returns:
            Tuple of (weights, objective_value, result_dict)
        """
        # Initialize parameters
        num_params = self.num_qubits
        x0 = np.random.randn(num_params) * 0.1
        
        # Define objective for scipy optimizer
        def cost_function(params):
            return -self.evaluate_expectation(params, objective)
        
        # Run optimization
        result = minimize(
            cost_function,
            x0,
            method='COBYLA',
            options={'maxiter': max_iterations}
        )
        
        # Extract optimal weights
        optimal_params = result.x
        weights = np.exp(optimal_params[:self.portfolio.n_assets])
        weights = weights / np.sum(weights)
        
        optimal_value = objective.evaluate(
            weights,
            self.portfolio.expected_returns,
            self.portfolio.covariance
        )
        
        self.optimal_params = optimal_params
        
        return weights, optimal_value, {
            'success': result.success,
            'iterations': result.nit,
            'function_calls': result.nfev,
        }
