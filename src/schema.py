"""Data schema validation module."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class SchemaValidator:
    """Validates DataFrame against a YAML schema definition."""

    def __init__(self, schema_path: str = "configs/schema.yaml"):
        """Load schema from YAML file."""
        with open(schema_path, "r") as f:
            config = yaml.safe_load(f)
        self.schema = config["schema"]
        self.columns = self.schema["columns"]
        self.target = self.schema.get("target")
        self.min_rows = self.schema.get("min_rows", 0)
        self.max_missing_pct = self.schema.get("max_missing_pct", 100)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a DataFrame against the schema."""
        errors = []
        warnings = []
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": {},
        }

        # Check required columns
        for col_name in self.columns:
            if col_name not in df.columns:
                errors.append(f"Missing required column: {col_name}")

        # Check data types and constraints for existing columns
        for col_name, col_schema in self.columns.items():
            if col_name not in df.columns:
                continue

            col_data = df[col_name]
            col_type = col_schema.get("type")
            nullable = col_schema.get("nullable", True)

            # Track missing values
            missing_count = col_data.isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            stats["missing_values"][col_name] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2),
            }

            # Check nullable constraint
            if not nullable and missing_count > 0:
                errors.append(
                    f"Column '{col_name}' has {missing_count} null values but is not nullable"
                )

            # Check missing percentage threshold
            if missing_pct > self.max_missing_pct:
                warnings.append(
                    f"Column '{col_name}' has {missing_pct:.1f}% missing values "
                    f"(threshold: {self.max_missing_pct}%)"
                )

            # Check data types
            if col_type in ("integer", "float"):
                non_null_data = col_data.dropna()
                if len(non_null_data) > 0:
                    if not pd.api.types.is_numeric_dtype(non_null_data):
                        errors.append(
                            f"Column '{col_name}' should be numeric but contains non-numeric values"
                        )
                    else:
                        # Check value ranges
                        min_val = col_schema.get("min")
                        max_val = col_schema.get("max")
                        if min_val is not None and non_null_data.min() < min_val:
                            errors.append(
                                f"Column '{col_name}' has values below minimum {min_val}"
                            )
                        if max_val is not None and non_null_data.max() > max_val:
                            errors.append(
                                f"Column '{col_name}' has values above maximum {max_val}"
                            )

            # Check categorical values
            if col_type == "category":
                allowed_values = col_schema.get("allowed_values", [])
                if allowed_values:
                    non_null_data = col_data.dropna()
                    invalid_values = set(non_null_data.unique()) - set(allowed_values)
                    if invalid_values:
                        errors.append(
                            f"Column '{col_name}' has invalid values: {invalid_values}. "
                            f"Allowed: {allowed_values}"
                        )

            # Check allowed values for integer type (like churn)
            if col_type == "integer":
                allowed_values = col_schema.get("allowed_values")
                if allowed_values is not None:
                    non_null_data = col_data.dropna()
                    invalid_values = set(non_null_data.unique()) - set(allowed_values)
                    if invalid_values:
                        errors.append(
                            f"Column '{col_name}' has invalid values: {invalid_values}. "
                            f"Allowed: {allowed_values}"
                        )

        # Check minimum row count
        if len(df) < self.min_rows:
            errors.append(
                f"DataFrame has {len(df)} rows, minimum required is {self.min_rows}"
            )

        # Check unique constraints
        for col_name, col_schema in self.columns.items():
            if col_name not in df.columns:
                continue
            if col_schema.get("unique", False):
                duplicates = df[col_name].duplicated().sum()
                if duplicates > 0:
                    errors.append(
                        f"Column '{col_name}' has {duplicates} duplicate values but must be unique"
                    )

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )


def validate_data(
    data_path: str = "data/raw/customers.csv",
    schema_path: str = "configs/schema.yaml",
    output_path: str = "data/validation_report.json",
) -> ValidationResult:
    """Main entry point for data validation.

    Args:
        data_path: Path to the CSV data file
        schema_path: Path to the schema YAML file
        output_path: Path to save the validation report

    Returns:
        ValidationResult with validation status and details
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    logger.info(f"Validating against schema: {schema_path}")
    validator = SchemaValidator(schema_path)
    result = validator.validate(df)

    # Log results
    if result.is_valid:
        logger.info("Validation PASSED")
    else:
        logger.error("Validation FAILED")
        for error in result.errors:
            logger.error(f"  - {error}")

    for warning in result.warnings:
        logger.warning(f"  - {warning}")

    # Save report
    report = {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "stats": result.stats,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report saved to {output_path}")

    return result


if __name__ == "__main__":
    result = validate_data()
    exit(0 if result.is_valid else 1)
