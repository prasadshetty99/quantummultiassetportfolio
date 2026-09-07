"""
Linear Programming Solver for Portfolio Optimization.

Implements linear programming approaches to portfolio optimization.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict, Tuple

from ..core.portfolio import Portfolio
from ..core.constraints import PortfolioConstraints
from ..core.objective import ObjectiveFunction, OptimizationObjective


class LinearProgrammingSolver:
    """
    Linear programming solver for portfolio optimization.
    
    Handles linear objectives and constraints.
    """
    
    def __init__(self, portfolio: Portfolio, constraints: PortfolioConstraints):
        """
        Initialize LP solver.
        
        Args:
            portfolio: Portfolio object
            constraints: Portfolio constraints
        """
        self.portfolio = portfolio
        self.constraints = constraints
    
    def optimize_returns(
        self,
        min_volatility: float = 0.0,
        max_volatility: float = 1.0
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Maximize portfolio returns subject to volatility bounds.
        
        Args:
            min_volatility: Minimum portfolio volatility
            max_volatility: Maximum portfolio volatility
            
        Returns:
            Tuple of (weights, return, result_dict)
        """
        n_assets = self.portfolio.n_assets
        
        # Minimize negative returns (maximize returns)
        c = -self.portfolio.expected_returns
        
        # Equality constraint: sum(w) = 1
        A_eq = np.ones((1, n_assets))
        b_eq = np.array([1.0])
        
        # Bounds: 0 <= w_i <= 1
        bounds = self.portfolio.get_asset_bounds()
        
        # Solve
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            weights = result.x
            port_return = np.dot(weights, self.portfolio.expected_returns)
            return weights, port_return, {'status': 'success'}
        else:
            return None, None, {'status': 'failed'}
    
    def optimize_sharpe_ratio(
        self,
        risk_free_rate: float = 0.02
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Maximize Sharpe ratio (linear approximation).
        
        Args:
            risk_free_rate: Risk-free rate
            
        Returns:
            Tuple of (weights, sharpe_ratio, result_dict)
        """
        # Objective: maximize (return - rf) / volatility
        # Approximate as: maximize (return - rf)
        n_assets = self.portfolio.n_assets
        
        c = -(self.portfolio.expected_returns - risk_free_rate)
        
        A_eq = np.ones((1, n_assets))
        b_eq = np.array([1.0])
        
        bounds = self.portfolio.get_asset_bounds()
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            weights = result.x
            port_return = np.dot(weights, self.portfolio.expected_returns)
            port_volatility = np.sqrt(weights @ self.portfolio.covariance @ weights)
            sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
            return weights, sharpe, {'status': 'success'}
        else:
            return None, None, {'status': 'failed'}
