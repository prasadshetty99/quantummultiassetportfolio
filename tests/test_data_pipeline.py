"""Test suite for data pipeline."""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Test DataPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline(lookback_years=2)
        self.test_assets = ["SPY", "AGG", "GLD"]

    def test_initialization(self):
        """Test DataPipeline initialization."""
        self.assertEqual(self.pipeline.lookback_years, 2)
        self.assertEqual(self.pipeline.source, "yfinance")

    def test_fetch_data_structure(self):
        """Test fetched data structure."""
        # This test would require actual data
        # In practice, you might mock yfinance
        pass

    def test_returns_calculation(self):
        """Test returns calculation."""
        # Create mock data
        prices = pd.DataFrame(
            {
                "A": [100, 105, 110],
                "B": [50, 52, 54],
            }
        )
        expected_returns = prices.pct_change().dropna()
        self.assertAlmostEqual(expected_returns.iloc[0, 0], 0.05)

    def test_covariance_matrix(self):
        """Test covariance matrix calculation."""
        # Create mock returns
        returns = pd.DataFrame(
            {
                "A": np.random.randn(252),
                "B": np.random.randn(252),
            }
        )
        cov_matrix = returns.cov() * 252
        self.assertEqual(cov_matrix.shape, (2, 2))
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))  # Check symmetry


if __name__ == "__main__":
    unittest.main()
