"""
Portfolio Visualization Utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class PortfolioVisualizer:
    """
    Visualization tools for portfolio analysis.
    """
    
    @staticmethod
    def prepare_efficient_frontier_plot(frontier: Dict) -> Dict:
        """
        Prepare efficient frontier data for plotting.
        
        Args:
            frontier: Frontier data from optimizer
            
        Returns:
            Plotting data
        """
        return {
            'x': frontier.get('volatilities', []),
            'y': frontier.get('returns', []),
            'title': 'Efficient Frontier',
            'xlabel': 'Volatility',
            'ylabel': 'Expected Return',
        }
    
    @staticmethod
    def prepare_weight_distribution(weights: np.ndarray, asset_names: list) -> Dict:
        """
        Prepare weight distribution for plotting.
        
        Args:
            weights: Portfolio weights
            asset_names: Asset names
            
        Returns:
            Plotting data
        """
        return {
            'labels': asset_names,
            'values': weights,
            'title': 'Portfolio Weight Distribution',
        }
    
    @staticmethod
    def prepare_returns_chart(returns: pd.Series) -> Dict:
        """
        Prepare cumulative returns for plotting.
        
        Args:
            returns: Series of returns
            
        Returns:
            Plotting data
        """
        cumulative = (1 + returns).cumprod()
        return {
            'dates': returns.index,
            'values': cumulative.values,
            'title': 'Cumulative Portfolio Returns',
            'ylabel': 'Cumulative Value',
        }
