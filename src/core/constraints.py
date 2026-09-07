"""
Portfolio Constraints - Define optimization constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class ConstraintSpec:
    """Specification for a single constraint."""
    name: str
    constraint_type: str  # 'equality', 'inequality', 'bound'
    assets: List[str]
    value: float
    operator: str = '<='  # '<=', '>=', '==', '<', '>'
    priority: int = 1  # Lower priority constraints may be relaxed
    penalty_weight: float = 1000.0


class PortfolioConstraints:
    """
    Portfolio constraint manager.
    
    Supports:
    - Budget constraint (weights sum to 1)
    - Sector exposure limits
    - Individual asset bounds
    - Diversification constraints
    - Risk limits
    - Concentration limits
    """
    
    def __init__(self):
        """Initialize constraint container."""
        self.constraints: Dict[str, ConstraintSpec] = {}
        self.asset_names: List[str] = []
        
        # Add default budget constraint
        self.add_budget_constraint()
    
    def add_budget_constraint(self) -> None:
        """Add default budget constraint (weights sum to 1)."""
        self.constraints["budget"] = ConstraintSpec(
            name="budget",
            constraint_type="equality",
            assets=[],  # Applies to all assets
            value=1.0,
            operator="==",
            priority=1,
            penalty_weight=10000.0
        )
    
    def add_sector_exposure_limit(
        self,
        sector_name: str,
        assets: List[str],
        min_exposure: float = 0.0,
        max_exposure: float = 1.0
    ) -> None:
        """
        Add sector exposure constraints.
        
        Args:
            sector_name: Name of sector
            assets: List of assets in sector
            min_exposure: Minimum sector weight
            max_exposure: Maximum sector weight
        """
        constraint_name_min = f"sector_{sector_name}_min"
        constraint_name_max = f"sector_{sector_name}_max"
        
        if min_exposure > 0:
            self.constraints[constraint_name_min] = ConstraintSpec(
                name=constraint_name_min,
                constraint_type="inequality",
                assets=assets,
                value=min_exposure,
                operator=">=",
                priority=2,
                penalty_weight=1000.0
            )
        
        if max_exposure < 1.0:
            self.constraints[constraint_name_max] = ConstraintSpec(
                name=constraint_name_max,
                constraint_type="inequality",
                assets=assets,
                value=max_exposure,
                operator="<=",
                priority=2,
                penalty_weight=1000.0
            )
    
    def add_diversification_constraint(
        self,
        min_positions: int,
        max_single_weight: float = 0.25
    ) -> None:
        """
        Add diversification constraints.
        
        Args:
            min_positions: Minimum number of non-zero positions
            max_single_weight: Maximum weight for single asset
        """
        self.constraints["diversification_max"] = ConstraintSpec(
            name="diversification_max",
            constraint_type="inequality",
            assets=[],  # Applies to all
            value=max_single_weight,
            operator="<=",
            priority=3,
            penalty_weight=500.0
        )
        
        self.constraints["diversification_min_positions"] = ConstraintSpec(
            name="diversification_min_positions",
            constraint_type="inequality",
            assets=[],  # Custom evaluation
            value=min_positions,
            operator=">=",
            priority=3,
            penalty_weight=500.0
        )
    
    def add_risk_limit(
        self,
        risk_metric: str,  # 'volatility', 'var', 'cvar'
        limit: float
    ) -> None:
        """
        Add risk constraints.
        
        Args:
            risk_metric: Type of risk metric
            limit: Risk limit value
        """
        self.constraints[f"risk_{risk_metric}"] = ConstraintSpec(
            name=f"risk_{risk_metric}",
            constraint_type="inequality",
            assets=[],
            value=limit,
            operator="<=",
            priority=1,
            penalty_weight=5000.0
        )
    
    def add_concentration_limit(
        self,
        assets: List[str],
        max_concentration: float
    ) -> None:
        """
        Add concentration constraints for asset groups.
        
        Args:
            assets: List of assets to constrain
            max_concentration: Maximum combined weight
        """
        self.constraints["concentration"] = ConstraintSpec(
            name="concentration",
            constraint_type="inequality",
            assets=assets,
            value=max_concentration,
            operator="<=",
            priority=2,
            penalty_weight=1000.0
        )
    
    def add_custom_constraint(
        self,
        name: str,
        constraint_func: Callable,
        penalty_weight: float = 1000.0
    ) -> None:
        """
        Add custom constraint with user-defined function.
        
        Args:
            name: Constraint name
            constraint_func: Function that evaluates constraint
            penalty_weight: Penalty weight for violation
        """
        # Store callable for later evaluation
        if not hasattr(self, '_custom_constraints'):
            self._custom_constraints = {}
        self._custom_constraints[name] = constraint_func
    
    def get_linear_constraints_matrix(
        self,
        asset_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate constraint matrices for linear constraints.
        
        Args:
            asset_names: List of asset names
            
        Returns:
            Tuple of (A_ub, b_ub, constraint_names)
        """
        self.asset_names = asset_names
        n_assets = len(asset_names)
        
        linear_constraints = []
        constraint_names = []
        
        for const_name, constraint in self.constraints.items():
            if constraint.name == "budget":
                # Budget constraint: sum(weights) == 1
                A = np.ones(n_assets)
                linear_constraints.append((A, constraint.value, constraint.operator, constraint.name))
                constraint_names.append(constraint.name)
                
            elif constraint.assets:
                # Sector or concentration constraints
                A = np.zeros(n_assets)
                for asset in constraint.assets:
                    if asset in asset_names:
                        idx = asset_names.index(asset)
                        A[idx] = 1.0
                
                linear_constraints.append((A, constraint.value, constraint.operator, constraint.name))
                constraint_names.append(constraint.name)
        
        return linear_constraints, constraint_names
    
    def check_constraints(self, weights: np.ndarray, asset_names: List[str]) -> Dict:
        """
        Check if solution satisfies all constraints.
        
        Args:
            weights: Portfolio weights
            asset_names: List of asset names
            
        Returns:
            Dictionary of constraint satisfaction status
        """
        status = {}
        
        # Check budget constraint
        budget_sum = np.sum(weights)
        status["budget"] = np.isclose(budget_sum, 1.0)
        
        # Check individual bounds
        for i, asset_name in enumerate(asset_names):
            status[f"bound_{asset_name}"] = 0 <= weights[i] <= 1.0
        
        # Check diversification
        non_zero_positions = np.sum(weights > 1e-6)
        status["diversification"] = non_zero_positions >= 2
        
        return status
    
    def get_summary(self) -> Dict:
        """Get constraint summary."""
        return {
            "n_constraints": len(self.constraints),
            "constraint_names": list(self.constraints.keys()),
            "constraints": {
                name: {
                    "type": constraint.constraint_type,
                    "value": constraint.value,
                    "operator": constraint.operator,
                }
                for name, constraint in self.constraints.items()
            }
        }
