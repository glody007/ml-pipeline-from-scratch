"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureEngineer


@pytest.fixture
def sample_dataframe():
    """Create a sample test DataFrame."""
    return pd.DataFrame({
        "customer_id": ["CUST_001", "CUST_002", "CUST_003"],
        "age": [25, 35, 45],
        "income": [50000.0, 75000.0, 100000.0],
        "gender": ["M", "F", "Other"],
        "region": ["North", "South", "East"],
        "tenure_months": [12, 24, 0],  # 0 to test division by zero handling
        "monthly_charges": [50.0, 75.0, 100.0],
        "total_charges": [600.0, 1800.0, 100.0],
        "churn": [0, 1, 0],
    })


@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance."""
    return FeatureEngineer("configs/features.yaml")


class TestFeatureEngineering:
    """Test suite for feature engineering."""

    def test_derived_feature_created(self, feature_engineer, sample_dataframe):
        """Derived feature charges_per_month should be created."""
        feature_engineer.fit(sample_dataframe)
        result = feature_engineer.transform(sample_dataframe)

        assert "charges_per_month" in result.columns

    def test_binned_feature_created(self, feature_engineer, sample_dataframe):
        """Binned feature age_group should be created."""
        feature_engineer.fit(sample_dataframe)
        result = feature_engineer.transform(sample_dataframe)

        assert "age_group" in result.columns
        # Check that values are from expected labels
        expected_labels = {"young", "adult", "middle_aged", "senior", "elderly"}
        actual_values = set(result["age_group"].dropna().unique())
        assert actual_values.issubset(expected_labels)

    def test_handles_missing_values(self, feature_engineer, sample_dataframe):
        """Transform should work with NaN values."""
        df = sample_dataframe.copy()
        df.loc[0, "income"] = np.nan
        df.loc[1, "total_charges"] = np.nan

        feature_engineer.fit(df)
        result = feature_engineer.transform(df)

        # Should not raise an error
        assert result is not None
        assert len(result) == len(df)

    def test_binary_feature_created(self, feature_engineer, sample_dataframe):
        """Binary feature is_high_value should be created with 0/1 values."""
        feature_engineer.fit(sample_dataframe)
        result = feature_engineer.transform(sample_dataframe)

        assert "is_high_value" in result.columns
        assert set(result["is_high_value"].unique()).issubset({0, 1})

    def test_transformer_is_fitted(self, feature_engineer, sample_dataframe):
        """After fit, learned_params_ should be populated."""
        feature_engineer.fit(sample_dataframe)

        assert hasattr(feature_engineer, "learned_params_")
        assert len(feature_engineer.learned_params_) > 0

    def test_division_by_zero_handled(self, feature_engineer, sample_dataframe):
        """tenure_months of 0 should not cause division error."""
        df = sample_dataframe.copy()
        df.loc[2, "tenure_months"] = 0  # This is already in the fixture

        feature_engineer.fit(df)
        result = feature_engineer.transform(df)

        # charges_per_month should be computed without error
        assert "charges_per_month" in result.columns
        # The value should be total_charges / (0 + 1) = total_charges
        assert result.loc[2, "charges_per_month"] == df.loc[2, "total_charges"]

    def test_tenure_group_created(self, feature_engineer, sample_dataframe):
        """Binned feature tenure_group should be created."""
        feature_engineer.fit(sample_dataframe)
        result = feature_engineer.transform(sample_dataframe)

        assert "tenure_group" in result.columns
        expected_labels = {"new", "established", "loyal", "veteran"}
        actual_values = set(result["tenure_group"].dropna().unique())
        assert actual_values.issubset(expected_labels)

    def test_income_to_charges_ratio_created(self, feature_engineer, sample_dataframe):
        """Derived feature income_to_charges_ratio should be created."""
        feature_engineer.fit(sample_dataframe)
        result = feature_engineer.transform(sample_dataframe)

        assert "income_to_charges_ratio" in result.columns
        # Check that values are positive
        assert all(result["income_to_charges_ratio"].dropna() >= 0)
