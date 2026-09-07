"""
Portfolio Analytics - Calculate performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict


class PortfolioAnalytics:
    """
    Portfolio performance analysis and metrics.
    """
    
    @staticmethod
    def calculate_returns(
        prices: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """
        Calculate portfolio returns over time.
        
        Args:
            prices: Historical prices
            weights: Portfolio weights
            
        Returns:
            Time series of portfolio returns
        """
        returns = prices.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        return portfolio_returns
    
    @staticmethod
    def calculate_cumulative_return(returns: pd.Series) -> float:
        """
        Calculate cumulative return.
        
        Args:
            returns: Series of returns
            
        Returns:
            Cumulative return
        """
        return float((1 + returns).prod() - 1)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        excess_return = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return float(excess_return / volatility) if volatility > 0 else 0.0
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio (downside risk only).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            target_return: Target return
            
        Returns:
            Sortino ratio
        """
        excess_return = returns.mean() * 252 - risk_free_rate
        downside = returns[returns < target_return].std() * np.sqrt(252)
        return float(excess_return / downside) if downside > 0 else 0.0
    
    @staticmethod
    def calculate_maximum_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    @staticmethod
    def get_summary(returns: pd.Series) -> Dict:
        """
        Get portfolio performance summary.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary of performance metrics
        """
        return {
            'annual_return': float(returns.mean() * 252),
            'annual_volatility': float(returns.std() * np.sqrt(252)),
            'sharpe_ratio': PortfolioAnalytics.calculate_sharpe_ratio(returns),
            'sortino_ratio': PortfolioAnalytics.calculate_sortino_ratio(returns),
            'max_drawdown': PortfolioAnalytics.calculate_maximum_drawdown(returns),
            'cumulative_return': PortfolioAnalytics.calculate_cumulative_return(returns),
        }
