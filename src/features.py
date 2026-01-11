"""Feature engineering module."""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer."""

    def __init__(self, config_path: str = "configs/features.yaml"):
        """Load feature configuration from YAML."""
        self.config_path = config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config["features"]
        self.numeric_features = self.config.get("numeric", [])
        self.categorical_features = self.config.get("categorical", [])
        self.derived_features = self.config.get("derived", {})
        self.binary_features = self.config.get("binary", {})
        self.binned_features = self.config.get("binned", {})
        self.target = self.config.get("target")
        self.learned_params_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Learn parameters from training data."""
        # Learn median values for numeric columns
        for col in self.numeric_features:
            if col in X.columns:
                self.learned_params_[f"{col}_median"] = X[col].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations."""
        df = X.copy()

        # Create derived features
        for feature_name, feature_def in self.derived_features.items():
            formula = feature_def["formula"]
            try:
                # Parse simple formulas like "total_charges / (tenure_months + 1)"
                if "/" in formula:
                    parts = formula.split("/")
                    numerator = parts[0].strip()
                    denominator = parts[1].strip()

                    # Handle denominator with addition (e.g., "(tenure_months + 1)")
                    if "+" in denominator:
                        denom_parts = denominator.replace("(", "").replace(")", "").split("+")
                        denom_col = denom_parts[0].strip()
                        denom_add = float(denom_parts[1].strip())
                        df[feature_name] = df[numerator] / (df[denom_col] + denom_add)
                    else:
                        df[feature_name] = df[numerator] / df[denominator]
            except Exception as e:
                logger.warning(f"Could not create derived feature {feature_name}: {e}")

        # Create binary features
        for feature_name, feature_def in self.binary_features.items():
            condition = feature_def["condition"]
            try:
                # Parse simple conditions like "monthly_charges > 70"
                if ">" in condition:
                    parts = condition.split(">")
                    col = parts[0].strip()
                    threshold = float(parts[1].strip())
                    df[feature_name] = (df[col] > threshold).astype(int)
                elif "<" in condition:
                    parts = condition.split("<")
                    col = parts[0].strip()
                    threshold = float(parts[1].strip())
                    df[feature_name] = (df[col] < threshold).astype(int)
            except Exception as e:
                logger.warning(f"Could not create binary feature {feature_name}: {e}")

        # Create binned features
        for feature_name, feature_def in self.binned_features.items():
            source = feature_def["source"]
            bins = feature_def["bins"]
            labels = feature_def["labels"]
            try:
                df[feature_name] = pd.cut(
                    df[source],
                    bins=bins,
                    labels=labels,
                    include_lowest=True,
                )
            except Exception as e:
                logger.warning(f"Could not create binned feature {feature_name}: {e}")

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        return df


def build_preprocessing_pipeline(config_path: str = "configs/features.yaml") -> ColumnTransformer:
    """Build sklearn preprocessing pipeline.

    Returns:
        ColumnTransformer with numeric and categorical transformers
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    features_config = config["features"]

    numeric_features = features_config.get("numeric", [])
    categorical_features = features_config.get("categorical", [])

    # Add derived and binary features to numeric (they produce numeric values)
    derived_features = features_config.get("derived", {})
    binary_features = features_config.get("binary", {})
    numeric_features = numeric_features + list(derived_features.keys()) + list(binary_features.keys())

    # Add binned features to categorical (they produce string labels)
    binned_features = features_config.get("binned", {})
    categorical_features = categorical_features + list(binned_features.keys())

    # Numeric transformer: impute missing then scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical transformer: impute missing then one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    return preprocessor


def run_feature_engineering(
    input_path: str = "data/raw/customers.csv",
    output_path: str = "data/processed/customers_featured.csv",
    config_path: str = "configs/features.yaml",
    transformer_path: str = "models/feature_engineer.pkl",
) -> pd.DataFrame:
    """Main entry point for feature engineering.

    Args:
        input_path: Path to raw data CSV
        output_path: Path to save transformed data
        config_path: Path to features YAML config
        transformer_path: Path to save fitted transformer

    Returns:
        Transformed DataFrame
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    # Apply feature engineering
    logger.info("Applying feature transformations")
    feature_engineer = FeatureEngineer(config_path)
    feature_engineer.fit(df)
    df_transformed = feature_engineer.transform(df)

    # Save transformed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_csv(output_path, index=False)
    logger.info(f"Saved transformed data to {output_path}")
    logger.info(f"Output shape: {df_transformed.shape}")

    # Save transformer
    Path(transformer_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_engineer, transformer_path)
    logger.info(f"Saved transformer to {transformer_path}")

    return df_transformed


if __name__ == "__main__":
    run_feature_engineering()
