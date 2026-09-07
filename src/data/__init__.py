"""Data ingestion and processing for portfolio optimization.

Supports:
- Live market data (Yahoo Finance, Alpha Vantage, Polygon)
- Historical data processing
- Risk metric calculation
- Covariance matrix estimation
"""

from .market_data import MarketDataProvider
from .risk_estimator import RiskEstimator

__all__ = ["MarketDataProvider", "RiskEstimator"]
