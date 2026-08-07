"""Risk analysis and metrics calculation."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class RiskAnalyzer:
    """Analyzes portfolio risk metrics."""

    def __init__(self, returns: pd.DataFrame, confidence_level: float = 0.95):
        """
        Initialize RiskAnalyzer.

        Args:
            returns: Historical returns DataFrame
            confidence_level: Confidence level for VaR/CVaR
        """
        self.returns = returns
        self.confidence_level = confidence_level
        logger.info(f"RiskAnalyzer initialized with {len(returns)} observations")

    def calculate_var(
        self,
        weights: np.ndarray,
        method: str = "historical",
        horizon: int = 1,
    ) -> float:
        """
        Calculate Value-at-Risk (VaR).

        Args:
            weights: Portfolio weights
            method: "historical" or "parametric"
            horizon: Time horizon in days

        Returns:
            VaR value
        """
        alpha = 1 - self.confidence_level
        portfolio_returns = self.returns @ weights

        if method == "historical":
            var = np.percentile(portfolio_returns, alpha * 100)
        elif method == "parametric":
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            var = mean - std * 1.645  # 95% confidence
        else:
            raise ValueError(f"Unknown method: {method}")

        var *= np.sqrt(horizon)
        logger.info(f"VaR ({self.confidence_level*100:.0f}%): {var:.4f}")
        return var

    def calculate_cvar(
        self,
        weights: np.ndarray,
        method: str = "historical",
    ) -> float:
        """
        Calculate Conditional Value-at-Risk (CVaR).

        Args:
            weights: Portfolio weights
            method: "historical" or "parametric"

        Returns:
            CVaR value
        """
        alpha = 1 - self.confidence_level
        portfolio_returns = self.returns @ weights

        if method == "historical":
            var = np.percentile(portfolio_returns, alpha * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
        elif method == "parametric":
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            var = mean - std * 1.645
            cvar = mean - std * (np.exp(-0.5 * 1.645**2) / alpha) / np.sqrt(2 * np.pi)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"CVaR ({self.confidence_level*100:.0f}%): {cvar:.4f}")
        return cvar

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Portfolio returns series

        Returns:
            Maximum drawdown ratio
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        logger.info(f"Maximum Drawdown: {max_drawdown:.4f}")
        return max_drawdown

    def calculate_sharpe_ratio(
        self,
        weights: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            weights: Portfolio weights
            risk_free_rate: Risk-free rate

        Returns:
            Sharpe ratio
        """
        portfolio_returns = self.returns @ weights
        excess_return = portfolio_returns.mean() - risk_free_rate / 252
        volatility = portfolio_returns.std()
        sharpe = excess_return / volatility * np.sqrt(252)

        logger.info(f"Sharpe Ratio: {sharpe:.4f}")
        return sharpe

    def calculate_sortino_ratio(
        self,
        weights: np.ndarray,
        risk_free_rate: float = 0.02,
        target_return: float = 0.0,
    ) -> float:
        """
        Calculate Sortino ratio.

        Args:
            weights: Portfolio weights
            risk_free_rate: Risk-free rate
            target_return: Target return

        Returns:
            Sortino ratio
        """
        portfolio_returns = self.returns @ weights
        excess_return = portfolio_returns.mean() - risk_free_rate / 252

        downside_returns = portfolio_returns[portfolio_returns < target_return]
        downside_deviation = downside_returns.std()

        sortino = excess_return / downside_deviation * np.sqrt(252)

        logger.info(f"Sortino Ratio: {sortino:.4f}")
        return sortino

    def calculate_information_ratio(
        self,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        Calculate Information ratio.

        Args:
            weights: Portfolio weights
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio
        """
        portfolio_returns = self.returns @ weights
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std()
        information_ratio = active_returns.mean() / tracking_error * np.sqrt(252)

        logger.info(f"Information Ratio: {information_ratio:.4f}")
        return information_ratio

    def get_risk_decomposition(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Get marginal and component contribution to risk.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary with risk decomposition
        """
        cov_matrix = self.returns.cov() * 252
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)

        marginal_contrib = cov_matrix @ weights / portfolio_std
        component_contrib = weights * marginal_contrib

        return {
            "marginal_contribution": marginal_contrib,
            "component_contribution": component_contrib,
            "total_risk": portfolio_std,
        }
