"""Tests for portfolio management."""

import unittest

import numpy as np
import pandas as pd

from src.portfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    """Test Portfolio class."""

    def setUp(self):
        """Set up test fixtures."""
        self.assets = ["SPY", "AGG", "GLD", "VNQ"]
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.portfolio = Portfolio(
            assets=self.assets,
            weights=self.weights,
            initial_capital=100000,
        )

    def test_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(len(self.portfolio.assets), 4)
        self.assertAlmostEqual(np.sum(self.portfolio.weights), 1.0)

    def test_get_allocation(self):
        """Test allocation retrieval."""
        allocation = self.portfolio.get_allocation()
        self.assertEqual(len(allocation), 4)
        self.assertAlmostEqual(allocation["SPY"], 0.4)

    def test_set_weights(self):
        """Test weight setting."""
        new_weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.portfolio.set_weights(new_weights)
        self.assertAlmostEqual(np.sum(self.portfolio.weights), 1.0)
        np.testing.assert_array_almost_equal(self.portfolio.weights, new_weights)

    def test_weight_normalization(self):
        """Test weight normalization."""
        unnormalized_weights = np.array([1, 2, 3, 4])  # Sums to 10
        self.portfolio.set_weights(unnormalized_weights)
        self.assertAlmostEqual(np.sum(self.portfolio.weights), 1.0)


if __name__ == "__main__":
    unittest.main()
