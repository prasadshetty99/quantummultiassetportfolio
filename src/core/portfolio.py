"""
Portfolio Class - Core data structure for portfolio representation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AssetClass:
    """Represents an asset class with its properties."""
    name: str
    expected_return: float
    volatility: float
    liquidity_score: float  # 0-1 scale
    transaction_cost_pct: float  # Percentage
    min_weight: float = 0.0
    max_weight: float = 1.0
    asset_ids: Optional[List[str]] = None


class Portfolio:
    """
    Core portfolio class managing asset allocation and risk metrics.
    
    Attributes:
        assets: Dict of asset classes
        weights: Current portfolio weights
        returns: Expected returns of assets
        covariance: Covariance matrix of asset returns
        constraints: Portfolio constraints
    """
    
    def __init__(self, asset_classes: Dict[str, AssetClass]):
        """
        Initialize portfolio with asset classes.
        
        Args:
            asset_classes: Dictionary mapping asset names to AssetClass objects
        """
        self.asset_classes = asset_classes
        self.asset_names = list(asset_classes.keys())
        self.n_assets = len(asset_classes)
        self.weights = np.ones(self.n_assets) / self.n_assets  # Equal weight initially
        
        # Initialize expected returns from asset classes
        self.expected_returns = np.array([
            asset_classes[name].expected_return 
            for name in self.asset_names
        ])
        
        # Placeholder for covariance matrix - will be set by data provider
        self.covariance = None
        self.correlation = None
        
    def set_covariance(self, cov_matrix: np.ndarray) -> None:
        """Set the covariance matrix for portfolio assets."""
        if cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"Covariance matrix shape {cov_matrix.shape} does not match number of assets {self.n_assets}")
        self.covariance = cov_matrix
        
        # Calculate correlation matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        self.correlation = cov_matrix / np.outer(std_devs, std_devs)
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set portfolio weights with validation."""
        if len(weights) != self.n_assets:
            raise ValueError(f"Weights length {len(weights)} does not match number of assets {self.n_assets}")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {np.sum(weights)}")
        if np.any(weights < 0):
            raise ValueError("Weights cannot be negative")
        
        self.weights = weights
    
    def expected_return(self) -> float:
        """Calculate expected portfolio return."""
        if self.weights is None:
            raise ValueError("Weights not set")
        return float(np.dot(self.weights, self.expected_returns))
    
    def portfolio_volatility(self) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        if self.covariance is None:
            raise ValueError("Covariance matrix not set")
        if self.weights is None:
            raise ValueError("Weights not set")
        
        variance = self.weights @ self.covariance @ self.weights
        return float(np.sqrt(variance))
    
    def calculate_var(self, confidence_level: float = 0.95, periods: int = 1) -> float:
        """
        Calculate Value at Risk (VaR) using historical volatility.
        
        Args:
            confidence_level: Confidence level (default 95%)
            periods: Number of periods forward
            
        Returns:
            VaR estimate
        """
        from scipy import stats
        
        volatility = self.portfolio_volatility()
        z_score = stats.norm.ppf(1 - confidence_level)
        var = self.expected_return() - z_score * volatility * np.sqrt(periods)
        return float(var)
    
    def calculate_cvar(self, confidence_level: float = 0.95, periods: int = 1, n_simulations: int = 10000) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) - Expected Shortfall.
        
        Args:
            confidence_level: Confidence level (default 95%)
            periods: Number of periods forward
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            CVaR estimate
        """
        # Generate returns using multivariate normal distribution
        returns = np.random.multivariate_normal(
            self.expected_returns,
            self.covariance,
            n_simulations
        )
        
        portfolio_returns = returns @ self.weights
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return float(cvar)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio."""
        excess_return = self.expected_return() - risk_free_rate
        volatility = self.portfolio_volatility()
        
        if volatility == 0:
            return 0.0
        
        return float(excess_return / volatility)
    
    def transaction_costs(self, new_weights: np.ndarray) -> float:
        """
        Calculate transaction costs when rebalancing to new weights.
        
        Args:
            new_weights: Target portfolio weights
            
        Returns:
            Total transaction cost in basis points
        """
        total_cost = 0.0
        for i, asset_name in enumerate(self.asset_names):
            weight_change = abs(new_weights[i] - self.weights[i])
            asset_cost_pct = self.asset_classes[asset_name].transaction_cost_pct
            total_cost += weight_change * asset_cost_pct
        
        return total_cost
    
    def get_asset_bounds(self) -> List[Tuple[float, float]]:
        """Get min/max weight bounds for each asset."""
        bounds = []
        for asset_name in self.asset_names:
            asset = self.asset_classes[asset_name]
            bounds.append((asset.min_weight, asset.max_weight))
        return bounds
    
    def get_summary(self) -> Dict:
        """Get portfolio summary statistics."""
        return {
            "n_assets": self.n_assets,
            "expected_return": self.expected_return(),
            "volatility": self.portfolio_volatility(),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "weights": self.weights.tolist(),
            "asset_names": self.asset_names,
        }
