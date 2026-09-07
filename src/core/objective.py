"""
Objective Function - Define portfolio optimization objectives.
"""

import numpy as np
from typing import Dict, Callable
from enum import Enum


class RiskMetric(Enum):
    """Risk metric types."""
    VOLATILITY = "volatility"
    VAR = "var"
    CVAR = "cvar"
    DOWNSIDE_DEV = "downside_dev"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MAX_RETURN = "max_return"
    MIN_RISK = "min_risk"
    MAX_SHARPE = "max_sharpe"
    MIN_CVAR = "min_cvar"
    MIN_MAD = "min_mad"  # Mean Absolute Deviation
    MAX_UTILITY = "max_utility"  # Risk-adjusted return


class ObjectiveFunction:
    """
    Portfolio optimization objective function.
    
    Supports multiple objective types:
    - Mean-Variance (Markowitz)
    - Mean-CVaR (Conditional Value at Risk)
    - Mean-MAD (Mean Absolute Deviation)
    - Utility maximization
    - Custom objectives
    """
    
    def __init__(
        self,
        objective_type: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        risk_aversion: float = 1.0,
        risk_metric: RiskMetric = RiskMetric.VOLATILITY
    ):
        """
        Initialize objective function.
        
        Args:
            objective_type: Type of optimization objective
            risk_aversion: Risk aversion coefficient (lambda)
            risk_metric: Risk metric to use
        """
        self.objective_type = objective_type
        self.risk_aversion = risk_aversion
        self.risk_metric = risk_metric
        self.risk_free_rate = 0.02
        self.custom_objective: Callable = None
    
    def set_risk_free_rate(self, rate: float) -> None:
        """Set risk-free rate for Sharpe ratio calculation."""
        self.risk_free_rate = rate
    
    def evaluate(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraint_penalty: float = 0.0
    ) -> float:
        """
        Evaluate objective function for given weights.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns of assets
            covariance: Covariance matrix
            constraint_penalty: Penalty for constraint violations
            
        Returns:
            Objective value (to be maximized or minimized)
        """
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        variance = weights @ covariance @ weights
        volatility = np.sqrt(variance)
        
        if self.objective_type == OptimizationObjective.MAX_RETURN:
            return float(portfolio_return - constraint_penalty)
        
        elif self.objective_type == OptimizationObjective.MIN_RISK:
            return float(-volatility - constraint_penalty)
        
        elif self.objective_type == OptimizationObjective.MAX_SHARPE:
            excess_return = portfolio_return - self.risk_free_rate
            if volatility == 0:
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = excess_return / volatility
            return float(sharpe_ratio - constraint_penalty)
        
        elif self.objective_type == OptimizationObjective.MAX_UTILITY:
            # Utility = Return - (risk_aversion * Risk)
            utility = portfolio_return - self.risk_aversion * volatility
            return float(utility - constraint_penalty)
        
        elif self.objective_type == OptimizationObjective.MIN_CVAR:
            # Use volatility as proxy for CVaR in optimization
            return float(-portfolio_return + self.risk_aversion * volatility - constraint_penalty)
        
        elif self.objective_type == OptimizationObjective.MIN_MAD:
            # Mean Absolute Deviation (approximated)
            mad = np.sqrt(2/np.pi) * volatility  # Approximation
            return float(-portfolio_return + self.risk_aversion * mad - constraint_penalty)
        
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")
    
    def gradient(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Calculate gradient of objective function.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance: Covariance matrix
            
        Returns:
            Gradient vector
        """
        portfolio_return = np.dot(weights, expected_returns)
        variance = weights @ covariance @ weights
        volatility = np.sqrt(variance)
        
        if self.objective_type == OptimizationObjective.MAX_RETURN:
            return expected_returns
        
        elif self.objective_type == OptimizationObjective.MIN_RISK:
            if volatility == 0:
                return np.zeros_like(weights)
            return -(covariance @ weights) / volatility
        
        elif self.objective_type == OptimizationObjective.MAX_SHARPE:
            excess_return = portfolio_return - self.risk_free_rate
            grad_return = expected_returns
            grad_vol = (covariance @ weights) / volatility if volatility > 0 else np.zeros_like(weights)
            
            if volatility == 0:
                return grad_return
            
            sharpe = excess_return / volatility
            grad_sharpe = (grad_return * volatility - excess_return * grad_vol) / (volatility ** 2)
            return grad_sharpe
        
        elif self.objective_type == OptimizationObjective.MAX_UTILITY:
            grad_vol = (covariance @ weights) / volatility if volatility > 0 else np.zeros_like(weights)
            return expected_returns - self.risk_aversion * grad_vol
        
        else:
            # Default gradient for other objectives
            return expected_returns - self.risk_aversion * (covariance @ weights)
    
    def hessian(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Hessian (second derivative matrix).
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance: Covariance matrix
            
        Returns:
            Hessian matrix
        """
        if self.objective_type in [OptimizationObjective.MIN_RISK, OptimizationObjective.MIN_CVAR]:
            # For variance-based objectives, Hessian is approximately 2*Covariance
            return 2.0 * covariance
        
        elif self.objective_type == OptimizationObjective.MAX_UTILITY:
            return -self.risk_aversion * covariance
        
        else:
            # Default Hessian
            return covariance
    
    def get_config(self) -> Dict:
        """Get objective function configuration."""
        return {
            "objective_type": self.objective_type.value,
            "risk_aversion": self.risk_aversion,
            "risk_metric": self.risk_metric.value,
            "risk_free_rate": self.risk_free_rate,
        }
