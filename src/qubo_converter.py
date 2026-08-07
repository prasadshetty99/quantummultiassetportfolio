"""Convert optimization problems to QUBO (Quadratic Unconstrained Binary Optimization)."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger


class QUBOConverter:
    """Convert portfolio optimization to QUBO formulation."""

    def __init__(self, n_assets: int, n_bits_per_asset: int = 3):
        """
        Initialize QUBOConverter.

        Args:
            n_assets: Number of assets
            n_bits_per_asset: Number of bits to represent each asset weight
        """
        self.n_assets = n_assets
        self.n_bits_per_asset = n_bits_per_asset
        self.n_qubits = n_assets * n_bits_per_asset
        logger.info(
            f"QUBOConverter initialized: {n_assets} assets, "
            f"{n_bits_per_asset} bits/asset, {self.n_qubits} total qubits"
        )

    def portfolio_to_qubo(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        constraints: Dict = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert portfolio optimization to QUBO matrix.

        Args:
            returns: Mean returns vector
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter (lambda)
            constraints: Optimization constraints

        Returns:
            Tuple of (QUBO matrix, metadata)
        """
        logger.info("Converting portfolio optimization to QUBO")

        # Normalize inputs
        returns_norm = returns / np.max(np.abs(returns))
        cov_norm = cov_matrix / np.max(np.abs(cov_matrix))

        # Objective: maximize return - lambda * risk
        # Convert to binary representation
        Q = np.zeros((self.n_qubits, self.n_qubits))

        for i in range(self.n_assets):
            for j in range(self.n_bits_per_asset):
                qubit_idx = i * self.n_bits_per_asset + j
                # Return term (to be maximized, so negative for minimization)
                Q[qubit_idx, qubit_idx] -= returns_norm[i] * (2 ** (-j))

        # Risk term (covariance matrix)
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                for bi in range(self.n_bits_per_asset):
                    for bj in range(self.n_bits_per_asset):
                        qi = i * self.n_bits_per_asset + bi
                        qj = j * self.n_bits_per_asset + bj
                        Q[qi, qj] += risk_aversion * cov_norm[i, j] * (2 ** (-bi - bj))

        # Add constraint penalties if needed
        if constraints:
            Q = self._add_constraint_penalties(Q, constraints)

        metadata = {
            "n_qubits": self.n_qubits,
            "n_assets": self.n_assets,
            "n_bits_per_asset": self.n_bits_per_asset,
            "risk_aversion": risk_aversion,
            "normalized": True,
        }

        logger.info(f"QUBO matrix created: shape={Q.shape}")
        return Q, metadata

    def _add_constraint_penalties(
        self,
        Q: np.ndarray,
        constraints: Dict,
    ) -> np.ndarray:
        """
        Add penalty terms for constraints.

        Args:
            Q: QUBO matrix
            constraints: Dictionary of constraints

        Returns:
            Modified QUBO matrix
        """
        logger.info("Adding constraint penalties to QUBO")
        penalty_weight = 10.0  # Penalty scaling factor

        # Sum-to-one constraint
        if "sum_to_one" in constraints and constraints["sum_to_one"]:
            for i in range(self.n_qubits):
                for j in range(self.n_qubits):
                    Q[i, j] += penalty_weight * 2
                Q[i, i] -= penalty_weight * 2

        # Min/max allocation constraints
        if "min_allocation" in constraints:
            min_alloc = constraints["min_allocation"]
            for i in range(self.n_assets):
                # Penalty if allocation below minimum
                Q[i * self.n_bits_per_asset, i * self.n_bits_per_asset] -= (
                    penalty_weight * min_alloc
                )

        return Q

    def ising_to_qubo(self, H: np.ndarray, J: Dict) -> np.ndarray:
        """
        Convert Ising model to QUBO.

        Args:
            H: Single qubit terms (diagonal)
            J: Coupling terms (dictionary)

        Returns:
            QUBO matrix
        """
        logger.info("Converting Ising model to QUBO")
        Q = np.diag(H)

        for (i, j), coupling in J.items():
            Q[i, j] += coupling / 2
            Q[j, i] += coupling / 2

        return Q

    def qubo_to_ising(self, Q: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Convert QUBO to Ising model.

        Args:
            Q: QUBO matrix

        Returns:
            Tuple of (H diagonal terms, J coupling dictionary)
        """
        logger.info("Converting QUBO to Ising model")
        # QUBO uses binary variables {0, 1}
        # Ising uses spin variables {-1, +1}
        # Conversion: x_i = (s_i + 1) / 2

        H = np.diag(Q) + np.sum(Q) / 2
        J = {}

        for i in range(Q.shape[0]):
            for j in range(i + 1, Q.shape[1]):
                if Q[i, j] != 0:
                    J[(i, j)] = Q[i, j] / 2

        return H, J

    def validate_qubo(self, Q: np.ndarray) -> bool:
        """
        Validate QUBO matrix.

        Args:
            Q: QUBO matrix to validate

        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating QUBO matrix")

        # Check if square
        if Q.shape[0] != Q.shape[1]:
            logger.error("QUBO matrix must be square")
            return False

        # Check if symmetric
        if not np.allclose(Q, Q.T):
            logger.warning("QUBO matrix is not symmetric, symmetrizing...")

        # Check eigenvalues (optional - for conditioning)
        eigenvalues = np.linalg.eigvalsh(Q)
        condition_number = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
        logger.info(f"Condition number: {condition_number:.2f}")

        return True
