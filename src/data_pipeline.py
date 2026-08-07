"""Data Pipeline for fetching and processing market data."""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


class DataPipeline:
    """Pipeline for fetching and processing financial market data."""

    def __init__(self, source: str = "yfinance", lookback_years: int = 5):
        """
        Initialize DataPipeline.

        Args:
            source: Data source ("yfinance", "alpha_vantage", "polygon")
            lookback_years: Years of historical data to fetch
        """
        self.source = source
        self.lookback_years = lookback_years
        self.assets = []
        self.data = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.cov_matrix = pd.DataFrame()
        logger.info(f"DataPipeline initialized with source: {source}")

    def fetch_data(
        self,
        assets: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch market data for given assets.

        Args:
            assets: List of asset symbols
            start_date: Start date (YYYY-MM-DD). If None, uses lookback_years
            end_date: End date (YYYY-MM-DD). If None, uses today
            interval: Data interval ("1d", "1wk", "1mo")

        Returns:
            Tuple of (returns DataFrame, covariance matrix)
        """
        self.assets = assets

        # Set dates
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (
                datetime.now() - timedelta(days=self.lookback_years * 365)
            ).strftime("%Y-%m-%d")

        logger.info(f"Fetching data for {len(assets)} assets from {start_date} to {end_date}")

        try:
            # Fetch data using yfinance
            data = yf.download(
                assets,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
            )

            # Handle single asset case
            if len(assets) == 1:
                data = data[["Adj Close"]]
                data.columns = assets
            else:
                data = data["Adj Close"]

            self.data = data.dropna()
            logger.info(f"Fetched {len(self.data)} data points")

            # Calculate returns
            self.returns = self.data.pct_change().dropna()
            self.cov_matrix = self.returns.cov() * 252  # Annualized

            logger.info(f"Calculated returns and covariance matrix")
            return self.returns, self.cov_matrix

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def get_price_data(self) -> pd.DataFrame:
        """Get raw price data."""
        return self.data

    def get_returns(self) -> pd.DataFrame:
        """Get returns data."""
        return self.returns

    def get_covariance_matrix(self) -> pd.DataFrame:
        """Get covariance matrix."""
        return self.cov_matrix

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix."""
        return self.returns.corr()

    def get_statistics(self) -> Dict:
        """Get statistical summary of assets."""
        stats = {
            "mean_return": self.returns.mean() * 252,
            "std_dev": self.returns.std() * np.sqrt(252),
            "sharpe_ratio": (self.returns.mean() / self.returns.std()) * np.sqrt(252),
            "skewness": self.returns.skew(),
            "kurtosis": self.returns.kurtosis(),
        }
        return stats
