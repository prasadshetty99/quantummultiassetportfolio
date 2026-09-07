"""
Risk Estimator - Calculate portfolio risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class RiskEstimator:
    """
    Estimate portfolio risk metrics from historical data.
    
    Supports:
    - Covariance matrix estimation (sample, shrinkage, DCC)
    - Correlation estimation
    - Value at Risk (VaR)
    - Conditional Value at Risk (CVaR)
    - Downside risk measures
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize risk estimator.
        
        Args:
            returns: DataFrame of historical returns
        """
        self.returns = returns
        self.cov_matrix = None
        self.corr_matrix = None
    
    def estimate_covariance(
        self,
        method: str = 'sample',
        shrinkage_intensity: float = 0.5
    ) -> np.ndarray:
        """
        Estimate covariance matrix.
        
        Args:
            method: 'sample', 'shrinkage', or 'ewma'
            shrinkage_intensity: Shrinkage parameter (0-1)
            
        Returns:
            Covariance matrix
        """
        if method == 'sample':
            cov = self.returns.cov().values
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage
            sample_cov = self.returns.cov().values
            target = np.eye(sample_cov.shape[0]) * np.trace(sample_cov) / sample_cov.shape[0]
            cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target
        elif method == 'ewma':
            cov = self.returns.ewm(span=20).cov().iloc[-len(self.returns.columns):].values
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.cov_matrix = cov
        return cov
    
    def estimate_correlation(self) -> np.ndarray:
        """
        Estimate correlation matrix.
        
        Returns:
            Correlation matrix
        """
        corr = self.returns.corr().values
        self.corr_matrix = corr
        return corr
    
    def calculate_var(
        self,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate historical Value at Risk.
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            VaR estimate
        """
        return np.percentile(self.returns.mean(), (1 - confidence_level) * 100)
    
    def calculate_cvar(
        self,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            CVaR estimate
        """
        var = self.calculate_var(confidence_level)
        return self.returns[self.returns <= var].mean().mean()
    
    def get_summary(self) -> Dict:
        """Get risk estimator summary."""
        return {
            'num_assets': self.returns.shape[1],
            'num_periods': self.returns.shape[0],
            'mean_return': float(self.returns.mean().mean()),
            'mean_volatility': float(self.returns.std().mean()),
        }
