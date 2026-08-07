"""Backtesting engine for portfolio strategies."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class Backtester:
    """Backtest portfolio strategies on historical data."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize Backtester.

        Args:
            initial_capital: Initial investment amount
            transaction_cost: Transaction cost ratio
            slippage: Slippage ratio
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.portfolio_values = []
        self.returns_series = []
        logger.info("Backtester initialized")

    def backtest(
        self,
        price_data: pd.DataFrame,
        weights_over_time: List[np.ndarray],
        rebalance_dates: List[pd.Timestamp],
    ) -> Dict:
        """
        Run backtest of portfolio strategy.

        Args:
            price_data: Historical price data
            weights_over_time: List of portfolio weights over time
            rebalance_dates: Dates for rebalancing

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {price_data.index[0]} to {price_data.index[-1]}")

        returns = price_data.pct_change().dropna()
        portfolio_value = self.initial_capital
        portfolio_values = [portfolio_value]
        returns_list = [0.0]

        current_weights_idx = 0
        current_weights = weights_over_time[0]

        for date, daily_returns in returns.iterrows():
            # Check for rebalancing
            if current_weights_idx < len(rebalance_dates) - 1:
                if date >= rebalance_dates[current_weights_idx + 1]:
                    # Rebalance
                    new_weights = weights_over_time[current_weights_idx + 1]
                    transaction_cost_amount = portfolio_value * np.sum(
                        np.abs(new_weights - current_weights)
                    ) * self.transaction_cost
                    portfolio_value -= transaction_cost_amount
                    current_weights = new_weights
                    current_weights_idx += 1
                    logger.info(f"Rebalanced at {date}. Cost: ${transaction_cost_amount:,.2f}")

            # Calculate daily return
            daily_return = np.dot(current_weights, daily_returns)
            portfolio_value *= 1 + daily_return

            portfolio_values.append(portfolio_value)
            returns_list.append(daily_return)

        self.portfolio_values = portfolio_values
        self.returns_series = returns_list

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_values, returns_list)
        logger.info(f"Backtest completed. Final value: ${portfolio_value:,.2f}")

        return metrics

    def _calculate_metrics(self, portfolio_values: List, returns: List) -> Dict:
        """
        Calculate backtest performance metrics.

        Args:
            portfolio_values: List of portfolio values over time
            returns: List of daily returns

        Returns:
            Dictionary with metrics
        """
        returns_array = np.array(returns[1:])  # Skip first zero return
        portfolio_array = np.array(portfolio_values)

        total_return = (portfolio_array[-1] - self.initial_capital) / self.initial_capital
        annual_return = (portfolio_array[-1] / self.initial_capital) ** (252 / len(returns)) - 1
        volatility = np.std(returns_array) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_value": portfolio_array[-1],
            "total_trades": len(portfolio_array),
        }

        return metrics

    def get_performance_dataframe(self) -> pd.DataFrame:
        """
        Get performance as DataFrame.

        Returns:
            DataFrame with portfolio values and returns
        """
        return pd.DataFrame(
            {
                "portfolio_value": self.portfolio_values,
                "daily_return": self.returns_series,
            }
        )
