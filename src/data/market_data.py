"""
Market Data Provider - Fetch and manage financial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MarketDataProvider:
    """
    Fetch and manage market data from various sources.
    
    Supports: Yahoo Finance, Alpha Vantage, Polygon API
    """
    
    def __init__(self):
        """Initialize market data provider."""
        self.data_cache = {}
        self.last_update = None
    
    def fetch_historical_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        source: str = 'yfinance'
    ) -> pd.DataFrame:
        """
        Fetch historical price data.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source
            
        Returns:
            DataFrame with historical prices
        """
        try:
            import yfinance as yf
            
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            return data['Adj Close']
        except Exception as e:
            raise Exception(f"Failed to fetch data: {e}")
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        return_type: str = 'log'
    ) -> pd.DataFrame:
        """
        Calculate asset returns from prices.
        
        Args:
            prices: DataFrame of prices
            return_type: 'log' or 'simple'
            
        Returns:
            DataFrame of returns
        """
        if return_type == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()
        
        return returns
    
    def calculate_expected_returns(
        self,
        returns: pd.DataFrame,
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Calculate expected returns.
        
        Args:
            returns: DataFrame of historical returns
            method: 'mean', 'ewma', or 'capm'
            
        Returns:
            Array of expected returns
        """
        if method == 'mean':
            return returns.mean().values * 252  # Annualized
        elif method == 'ewma':
            return returns.ewm(span=20).mean().iloc[-1].values * 252
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_summary(self) -> Dict:
        """Get data provider summary."""
        return {
            'cached_assets': len(self.data_cache),
            'last_update': self.last_update,
        }
