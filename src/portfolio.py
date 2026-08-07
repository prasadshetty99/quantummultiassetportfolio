"""Portfolio management and analysis."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class Portfolio:
    """Portfolio manager for tracking and analyzing portfolios."""

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None,
        initial_capital: float = 100000.0,
    ):
        """
        Initialize Portfolio.

        Args:
            assets: List of asset symbols
            weights: Portfolio weights
            initial_capital: Initial investment amount
        """
        self.assets = assets or []
        self.weights = weights or np.ones(len(self.assets)) / len(self.assets)
        self.initial_capital = initial_capital
        self.returns = None
        self.cov_matrix = None
        logger.info(f"Portfolio initialized with {len(self.assets)} assets")

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set portfolio weights.

        Args:
            weights: Portfolio weights (must sum to 1)
        """
        if not np.isclose(np.sum(weights), 1.0):
            logger.warning(f"Weights don't sum to 1: {np.sum(weights):.6f}. Normalizing...")
            weights = weights / np.sum(weights)
        self.weights = weights
        logger.info(f"Portfolio weights updated")

    def get_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation.

        Returns:
            Dictionary of asset: allocation
        """
        return {asset: float(weight) for asset, weight in zip(self.assets, self.weights)}

    def calculate_metrics(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
    ) -> Dict:
        """
        Calculate portfolio metrics.

        Args:
            returns: Historical returns DataFrame
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary of metrics
        """
        self.returns = returns
        self.cov_matrix = cov_matrix

        weights = self.weights
        expected_return = np.dot(weights, returns.mean()) * 252
        variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        volatility = np.sqrt(variance)
        sharpe_ratio = (expected_return - risk_free_rate) / volatility

        metrics = {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "var_95": np.percentile(returns @ weights, 5),
            "var_99": np.percentile(returns @ weights, 1),
        }

        logger.info(f"Portfolio metrics calculated")
        return metrics

    def summary(self) -> str:
        """
        Get portfolio summary.

        Returns:
            Formatted summary string
        """
        allocation = self.get_allocation()
        summary_lines = ["\n=== Portfolio Summary ==="]
        summary_lines.append(f"Number of Assets: {len(self.assets)}")
        summary_lines.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        summary_lines.append("\nAllocation:")

        for asset, weight in allocation.items():
            summary_lines.append(f"  {asset}: {weight*100:.2f}%")

        return "\n".join(summary_lines)

    def rebalance(
        self,
        new_weights: np.ndarray,
        transaction_cost: float = 0.001,
    ) -> Dict:
        """
        Rebalance portfolio.

        Args:
            new_weights: New portfolio weights
            transaction_cost: Transaction cost ratio

        Returns:
            Dictionary with rebalancing details
        """
        weight_changes = np.abs(new_weights - self.weights)
        transaction_cost_amount = self.initial_capital * np.sum(weight_changes) * transaction_cost

        logger.info(f"Portfolio rebalanced. Transaction cost: ${transaction_cost_amount:,.2f}")

        self.set_weights(new_weights)

        return {
            "old_weights": self.weights,
            "new_weights": new_weights,
            "transaction_cost": transaction_cost_amount,
        }
