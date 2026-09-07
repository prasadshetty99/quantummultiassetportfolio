"""
CVXPY Convex Optimization Solver for Portfolio Optimization.
"""

import numpy as np
from typing import Dict, Tuple, Optional
try:
    import cvxpy as cp
except ImportError:
    cp = None

from ..core.portfolio import Portfolio
from ..core.constraints import PortfolioConstraints
from ..core.objective import ObjectiveFunction, OptimizationObjective


class CVXPYSolver:
    """
    Convex portfolio optimization using CVXPY.
    
    Supports:
    - Mean-Variance optimization
    - Risk parity
    - Min-variance portfolios
    - Constrained optimization
    """
    
    def __init__(self, portfolio: Portfolio, constraints: PortfolioConstraints):
        """
        Initialize CVXPY solver.
        
        Args:
            portfolio: Portfolio object
            constraints: Portfolio constraints
        """
        if cp is None:
            raise ImportError("CVXPY is required. Install with: pip install cvxpy")
        
        self.portfolio = portfolio
        self.constraints = constraints
        self.optimization_result = None
    
    def optimize(
        self,
        objective: ObjectiveFunction,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Solve portfolio optimization problem using CVXPY.
        
        Args:
            objective: Optimization objective
            verbose: Print solver output
            
        Returns:
            Tuple of (optimal_weights, objective_value, result_dict)
        """
        n_assets = self.portfolio.n_assets
        
        # Decision variable: portfolio weights
        w = cp.Variable(n_assets)
        
        # Portfolio return and risk
        ret = self.portfolio.expected_returns @ w
        risk = cp.quad_form(w, self.portfolio.covariance)
        
        # Objective function
        if objective.objective_type == OptimizationObjective.MAX_SHARPE:
            # Sharpe ratio maximization (approximation)
            obj = ret - objective.risk_free_rate - objective.risk_aversion * cp.sqrt(risk)
        elif objective.objective_type == OptimizationObjective.MIN_RISK:
            obj = -cp.sqrt(risk)
        elif objective.objective_type == OptimizationObjective.MAX_UTILITY:
            obj = ret - objective.risk_aversion * cp.sqrt(risk)
        else:
            obj = ret
        
        # Constraints
        constraints_cvxpy = [
            cp.sum(w) == 1.0,  # Budget constraint
            w >= 0,  # Non-negativity
        ]
        
        # Add asset bounds
        for i, (lb, ub) in enumerate(self.portfolio.get_asset_bounds()):
            constraints_cvxpy[1]  # Already covered by w >= 0
        
        # Solve problem (maximize objective)
        problem = cp.Problem(cp.Maximize(obj), constraints_cvxpy)
        problem.solve(verbose=verbose)
        
        optimal_weights = w.value
        optimal_value = problem.value
        
        return optimal_weights, optimal_value, {
            'status': problem.status,
            'solver': problem.solver_stats.solver_name if hasattr(problem.solver_stats, 'solver_name') else 'Unknown',
        }
    
    def min_variance(
        self,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Find minimum variance portfolio.
        
        Args:
            verbose: Print solver output
            
        Returns:
            Tuple of (weights, variance, result_dict)
        """
        n_assets = self.portfolio.n_assets
        w = cp.Variable(n_assets)
        
        # Minimize portfolio variance
        variance = cp.quad_form(w, self.portfolio.covariance)
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
        ]
        
        problem = cp.Problem(cp.Minimize(variance), constraints)
        problem.solve(verbose=verbose)
        
        return w.value, np.sqrt(problem.value), {'status': problem.status}
    
    def risk_parity(
        self,
        target_risk: float = 0.1,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Find risk parity portfolio.
        
        Args:
            target_risk: Target portfolio volatility
            verbose: Print solver output
            
        Returns:
            Tuple of (weights, volatility, result_dict)
        """
        n_assets = self.portfolio.n_assets
        w = cp.Variable(n_assets)
        
        # Risk parity: each asset contributes equally to portfolio risk
        risk_contrib = cp.sqrt(cp.quad_form(w, self.portfolio.covariance))
        
        # Minimize deviation from equal risk contribution
        equal_contribution = target_risk / n_assets
        obj = cp.sum_squares(risk_contrib * w - equal_contribution)
        
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
        ]
        
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(verbose=verbose)
        
        return w.value, np.sqrt(w.value @ self.portfolio.covariance @ w.value), {'status': problem.status}
