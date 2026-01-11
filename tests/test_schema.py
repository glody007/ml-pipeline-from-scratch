"""Unit tests for schema validation."""

import pandas as pd
import pytest

from src.schema import SchemaValidator, ValidationResult


@pytest.fixture
def valid_dataframe():
    """Create a valid test DataFrame."""
    return pd.DataFrame({
        "customer_id": ["CUST_001", "CUST_002", "CUST_003"],
        "age": [25, 35, 45],
        "income": [50000.0, 75000.0, 100000.0],
        "gender": ["M", "F", "Other"],
        "region": ["North", "South", "East"],
        "tenure_months": [12, 24, 36],
        "monthly_charges": [50.0, 75.0, 100.0],
        "total_charges": [600.0, 1800.0, 3600.0],
        "churn": [0, 1, 0],
    })


@pytest.fixture
def validator():
    """Create a SchemaValidator instance."""
    return SchemaValidator("configs/schema.yaml")


class TestSchemaValidation:
    """Test suite for schema validation."""

    def test_valid_data_passes(self, validator, valid_dataframe):
        """Valid data should pass validation."""
        # Add more rows to meet min_rows requirement
        df = pd.concat([valid_dataframe] * 40, ignore_index=True)
        df["customer_id"] = [f"CUST_{i:04d}" for i in range(len(df))]

        result = validator.validate(df)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_column_fails(self, validator, valid_dataframe):
        """Missing required column should fail validation."""
        df = valid_dataframe.drop(columns=["age"])

        result = validator.validate(df)

        assert result.is_valid is False
        assert any("Missing required column: age" in e for e in result.errors)

    def test_invalid_type_fails(self, validator):
        """String in numeric column should fail validation."""
        df = pd.DataFrame({
            "customer_id": ["CUST_001"],
            "age": ["not_a_number"],  # Invalid type
            "income": [50000.0],
            "gender": ["M"],
            "region": ["North"],
            "tenure_months": [12],
            "monthly_charges": [50.0],
            "total_charges": [600.0],
            "churn": [0],
        })

        result = validator.validate(df)

        assert result.is_valid is False
        assert any("numeric" in e.lower() for e in result.errors)

    def test_out_of_range_fails(self, validator, valid_dataframe):
        """Age value out of range should fail validation."""
        df = valid_dataframe.copy()
        df.loc[0, "age"] = 150  # Out of range (max is 120)

        result = validator.validate(df)

        assert result.is_valid is False
        assert any("above maximum" in e for e in result.errors)

    def test_invalid_category_fails(self, validator, valid_dataframe):
        """Invalid category value should fail validation."""
        df = valid_dataframe.copy()
        df.loc[0, "gender"] = "X"  # Invalid category

        result = validator.validate(df)

        assert result.is_valid is False
        assert any("invalid values" in e.lower() for e in result.errors)

    def test_nullable_column_allows_nulls(self, validator, valid_dataframe):
        """Nullable column should allow null values."""
        df = pd.concat([valid_dataframe] * 40, ignore_index=True)
        df["customer_id"] = [f"CUST_{i:04d}" for i in range(len(df))]
        df.loc[0, "income"] = None  # income is nullable

        result = validator.validate(df)

        # Should not have error for nullable income column
        assert not any("income" in e and "null" in e.lower() for e in result.errors)

    def test_non_nullable_fails_with_nulls(self, validator, valid_dataframe):
        """Non-nullable column should fail with null values."""
        df = valid_dataframe.copy()
        df.loc[0, "age"] = None  # age is not nullable

        result = validator.validate(df)

        assert result.is_valid is False
        assert any("age" in e and "null" in e.lower() for e in result.errors)

    def test_duplicate_unique_column_fails(self, validator, valid_dataframe):
        """Duplicate values in unique column should fail validation."""
        df = pd.concat([valid_dataframe] * 40, ignore_index=True)
        # Keep some duplicate customer_ids
        df.loc[0, "customer_id"] = "DUPLICATE"
        df.loc[1, "customer_id"] = "DUPLICATE"

        result = validator.validate(df)

        assert result.is_valid is False
        assert any("duplicate" in e.lower() for e in result.errors)
