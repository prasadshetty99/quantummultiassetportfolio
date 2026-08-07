"""Tests for classical optimization methods."""

import unittest

import numpy as np
import pandas as pd

from src.classical_optimizer import ClassicalOptimizer


class TestClassicalOptimizer(unittest.TestCase):
    """Test ClassicalOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.returns = np.array([0.08, 0.10, 0.06, 0.09])
        self.cov_matrix = np.array(
            [
                [0.04, 0.01, 0.02, 0.01],
                [0.01, 0.05, 0.01, 0.02],
                [0.02, 0.01, 0.03, 0.01],
                [0.01, 0.02, 0.01, 0.04],
            ]
        )
        self.optimizer = ClassicalOptimizer(self.returns, self.cov_matrix)

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.n_assets, 4)
        self.assertEqual(self.optimizer.risk_free_rate, 0.02)

    def test_minimum_variance(self):
        """Test minimum variance optimization."""
        weights = self.optimizer.minimum_variance()
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))

    def test_max_sharpe_ratio(self):
        """Test max Sharpe ratio optimization."""
        weights = self.optimizer.max_sharpe_ratio()
        self.assertAlmostEqual(np.sum(weights), 1.0)
        self.assertTrue(np.all(weights >= 0))

    def test_weight_constraints(self):
        """Test weight constraints."""
        min_alloc = 0.1
        max_alloc = 0.5
        weights = self.optimizer.minimum_variance(
            min_allocation=min_alloc,
            max_allocation=max_alloc,
        )
        self.assertTrue(np.all(weights >= min_alloc))
        self.assertTrue(np.all(weights <= max_alloc))


if __name__ == "__main__":
    unittest.main()
