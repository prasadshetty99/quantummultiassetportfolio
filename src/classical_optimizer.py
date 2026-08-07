"""Classical optimization methods for portfolio optimization."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from loguru import logger

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("CVXPY not available. Install with: pip install cvxpy")


class ClassicalOptimizer:
    """Classical portfolio optimization methods."""

    def __init__(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize Classical Optimizer.

        Args:
            returns: Mean returns array
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.returns = np.array(returns)
        self.cov_matrix = np.array(cov_matrix)
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns)
        logger.info(f"ClassicalOptimizer initialized with {self.n_assets} assets")

    def markowitz_optimization(
        self,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        min_allocation: float = 0.0,
        max_allocation: float = 1.0,
    ) -> np.ndarray:
        """
        Markowitz Mean-Variance Portfolio Optimization.

        Args:
            target_return: Target portfolio return (optional)
            max_volatility: Maximum portfolio volatility (optional)
            min_allocation: Minimum allocation per asset
            max_allocation: Maximum allocation per asset

        Returns:
            Optimized portfolio weights
        """
        logger.info("Starting Markowitz optimization")

        if not CVXPY_AVAILABLE:
            return self._markowitz_scipy(min_allocation, max_allocation)

        # CVXPY formulation
        weights = cp.Variable(self.n_assets)

        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, self.cov_matrix)

        constraints = [
            cp.sum(weights) == 1,
            weights >= min_allocation,
            weights <= max_allocation,
        ]

        if target_return is not None:
            constraints.append(weights @ self.returns >= target_return)

        if max_volatility is not None:
            constraints.append(cp.sqrt(portfolio_variance) <= max_volatility)

        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve()

        if problem.status == cp.OPTIMAL:
            logger.info(f"Markowitz optimization successful. Min variance: {problem.value:.6f}")
            return np.array(weights.value)
        else:
            logger.warning(f"Optimization status: {problem.status}")
            return np.ones(self.n_assets) / self.n_assets

    def max_sharpe_ratio(
        self,
        min_allocation: float = 0.0,
        max_allocation: float = 1.0,
    ) -> np.ndarray:
        """
        Maximize Sharpe Ratio portfolio.

        Args:
            min_allocation: Minimum allocation per asset
            max_allocation: Maximum allocation per asset

        Returns:
            Optimized portfolio weights
        """
        logger.info("Starting Max Sharpe Ratio optimization")

        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.returns)
            portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe  # Negative because we minimize

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((min_allocation, max_allocation) for _ in range(self.n_assets))
        initial_guess = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            negative_sharpe,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        logger.info(f"Max Sharpe Ratio: {-result.fun:.4f}")
        return result.x

    def minimum_variance(
        self,
        min_allocation: float = 0.0,
        max_allocation: float = 1.0,
    ) -> np.ndarray:
        """
        Minimum Variance Portfolio.

        Args:
            min_allocation: Minimum allocation per asset
            max_allocation: Maximum allocation per asset

        Returns:
            Optimized portfolio weights
        """
        logger.info("Starting Minimum Variance optimization")

        def portfolio_variance(weights):
            return np.dot(weights, np.dot(self.cov_matrix, weights))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((min_allocation, max_allocation) for _ in range(self.n_assets))
        initial_guess = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            portfolio_variance,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        logger.info(f"Minimum Variance: {result.fun:.6f}")
        return result.x

    def cvar_optimization(
        self,
        confidence_level: float = 0.95,
        target_return: Optional[float] = None,
        min_allocation: float = 0.0,
        max_allocation: float = 1.0,
    ) -> np.ndarray:
        """
        CVaR (Conditional Value-at-Risk) Optimization.

        Args:
            confidence_level: Confidence level for CVaR calculation
            target_return: Target portfolio return
            min_allocation: Minimum allocation per asset
            max_allocation: Maximum allocation per asset

        Returns:
            Optimized portfolio weights
        """
        logger.info(f"Starting CVaR optimization (confidence={confidence_level})")
        logger.warning("CVaR optimization requires scenario data")

        # Placeholder: use minimum variance as fallback
        return self.minimum_variance(min_allocation, max_allocation)

    def efficient_frontier(
        self,
        n_points: int = 100,
        min_allocation: float = 0.0,
        max_allocation: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate efficient frontier.

        Args:
            n_points: Number of points on frontier
            min_allocation: Minimum allocation per asset
            max_allocation: Maximum allocation per asset

        Returns:
            Tuple of (volatilities, returns)
        """
        logger.info(f"Generating efficient frontier with {n_points} points")

        min_ret = np.min(self.returns)
        max_ret = np.max(self.returns)
        target_returns = np.linspace(min_ret, max_ret, n_points)

        volatilities = []
        returns_list = []

        for target_return in target_returns:
            weights = self.markowitz_optimization(
                target_return=target_return,
                min_allocation=min_allocation,
                max_allocation=max_allocation,
            )
            portfolio_return = np.dot(weights, self.returns)
            portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            volatilities.append(portfolio_std)
            returns_list.append(portfolio_return)

        return np.array(volatilities), np.array(returns_list)

    def _markowitz_scipy(
        self,
        min_allocation: float,
        max_allocation: float,
    ) -> np.ndarray:
        """
        Markowitz optimization using scipy (fallback).

        Args:
            min_allocation: Minimum allocation per asset
            max_allocation: Maximum allocation per asset

        Returns:
            Optimized portfolio weights
        """
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(self.cov_matrix, weights))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((min_allocation, max_allocation) for _ in range(self.n_assets))
        initial_guess = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            portfolio_variance,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_guess
