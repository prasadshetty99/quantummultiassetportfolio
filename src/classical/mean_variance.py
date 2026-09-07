"""
Mean-Variance Optimizer - Classical Markowitz Portfolio Optimization.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List
from ..core.portfolio import Portfolio
from ..core.constraints import PortfolioConstraints
from ..core.objective import ObjectiveFunction


class MeanVarianceOptimizer:
    """
    Classical Mean-Variance Portfolio Optimizer.
    
    Implements Markowitz portfolio optimization using scipy.optimize.
    """
    
    def __init__(self, portfolio: Portfolio, constraints: PortfolioConstraints):
        """
        Initialize optimizer.
        
        Args:
            portfolio: Portfolio object
            constraints: Portfolio constraints
        """
        self.portfolio = portfolio
        self.constraints = constraints
        self.optimization_result = None
    
    def optimize(
        self,
        objective: ObjectiveFunction,
        method: str = "SLSQP",
        max_iterations: int = 1000
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Optimize portfolio weights.
        
        Args:
            objective: Optimization objective function
            method: Optimization method (SLSQP, L-BFGS-B)
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (optimal_weights, objective_value, result_dict)
        """
        # Initial weights (equal-weight)
        x0 = self.portfolio.weights.copy()
        
        # Bounds for weights
        bounds = self.portfolio.get_asset_bounds()
        
        # Budget constraint (weights sum to 1)
        constraints_list = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        }]
        
        # Objective function to minimize (negative for maximization)
        def objective_func(w):
            return -objective.evaluate(
                w,
                self.portfolio.expected_returns,
                self.portfolio.covariance
            )
        
        # Run optimization
        result = minimize(
            objective_func,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': max_iterations}
        )
        
        self.optimization_result = result
        optimal_weights = result.x
        optimal_value = -result.fun  # Negate back
        
        return optimal_weights, optimal_value, {
            'success': result.success,
            'message': result.message,
            'iterations': result.nit,
            'function_calls': result.nfev,
        }
    
    def efficient_frontier(
        self,
        num_portfolios: int = 50,
        objective: Optional[ObjectiveFunction] = None
    ) -> Dict:
        """
        Generate efficient frontier.
        
        Args:
            num_portfolios: Number of portfolios on frontier
            objective: Optimization objective
            
        Returns:
            Dictionary with frontier data
        """
        target_returns = np.linspace(
            self.portfolio.expected_returns.min(),
            self.portfolio.expected_returns.max(),
            num_portfolios
        )
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []
        
        for target_return in target_returns:
            # Add return constraint
            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.portfolio.expected_returns) - target_return}
            ]
            
            # Minimize volatility
            def min_volatility(w):
                return np.sqrt(w @ self.portfolio.covariance @ w)
            
            result = minimize(
                min_volatility,
                self.portfolio.weights,
                method='SLSQP',
                bounds=self.portfolio.get_asset_bounds(),
                constraints=constraints_list
            )
            
            if result.success:
                weights = result.x
                frontier_weights.append(weights)
                frontier_returns.append(np.dot(weights, self.portfolio.expected_returns))
                frontier_volatilities.append(np.sqrt(weights @ self.portfolio.covariance @ weights))
        
        return {
            'returns': np.array(frontier_returns),
            'volatilities': np.array(frontier_volatilities),
            'weights': frontier_weights,
        }
    
    def get_summary(self) -> Dict:
        """Get optimization summary."""
        if self.optimization_result is None:
            return {'status': 'Not optimized yet'}
        
        weights = self.optimization_result.x
        return {
            'optimal_weights': weights.tolist(),
            'portfolio_return': float(np.dot(weights, self.portfolio.expected_returns)),
            'portfolio_volatility': float(np.sqrt(weights @ self.portfolio.covariance @ weights)),
            'optimization_success': bool(self.optimization_result.success),
            'iterations': self.optimization_result.nit,
        }
