"""Model training module with MLflow tracking."""

import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import FeatureEngineer, build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_class(class_path: str) -> type:
    """Dynamically import a model class from a string path.

    Args:
        class_path: Full path like 'sklearn.ensemble.RandomForestClassifier'

    Returns:
        The model class
    """
    parts = class_path.rsplit(".", 1)
    module_path = parts[0]
    class_name = parts[1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def train_model(
    data_path: str = "data/processed/customers_featured.csv",
    feature_config_path: str = "configs/features.yaml",
    model_config_path: str = "configs/model.yaml",
    output_model_path: str = "models/model.pkl",
    metrics_path: str = "models/metrics.yaml",
) -> Dict[str, Any]:
    """Train and evaluate models, tracking with MLflow.

    Args:
        data_path: Path to featured data CSV
        feature_config_path: Path to features config
        model_config_path: Path to model config
        output_model_path: Path to save best model
        metrics_path: Path to save metrics YAML

    Returns:
        Dictionary with best_model name, metrics, and model_path
    """
    # Load configs
    with open(feature_config_path, "r") as f:
        feature_config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    target = feature_config["features"]["target"]
    experiment_name = model_config["experiment"]["name"]
    test_size = model_config.get("test_size", 0.2)
    random_state = model_config.get("random_state", 42)
    cv_config = model_config.get("cross_validation", {})

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Prepare features and target
    # Drop customer_id if present (it's an identifier, not a feature)
    drop_cols = [target]
    if "customer_id" in df.columns:
        drop_cols.append("customer_id")
    X = df.drop(columns=drop_cols)
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Setup MLflow
    mlflow.set_experiment(experiment_name)

    # Track best model
    best_model_name = None
    best_f1 = 0.0
    best_pipeline = None
    best_metrics = {}

    # Train each model
    for model_name, model_def in model_config["models"].items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model: {model_name}")

        with mlflow.start_run(run_name=model_name):
            # Build model
            model_class = load_model_class(model_def["class"])
            model_params = model_def.get("params", {})
            model = model_class(**model_params)

            # Build full pipeline
            feature_engineer = FeatureEngineer(feature_config_path)
            preprocessor = build_preprocessing_pipeline(feature_config_path)

            pipeline = Pipeline(
                steps=[
                    ("feature_engineer", feature_engineer),
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            # Cross validation
            n_splits = cv_config.get("n_splits", 5)
            shuffle = cv_config.get("shuffle", True)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
            logger.info(f"CV F1 scores: {cv_scores}")
            logger.info(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Log CV metrics
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())

            # Fit on full training data
            pipeline.fit(X_train, y_train)

            # Evaluate on test data
            y_pred = pipeline.predict(X_test)
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_proba)

            # Log metrics and params
            mlflow.log_params(model_params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"  {metric_name}: {metric_value:.4f}")

            # Track best model
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model_name = model_name
                best_pipeline = pipeline
                best_metrics = metrics

    # Save best model
    logger.info(f"\n{'='*50}")
    logger.info(f"Best model: {best_model_name} (F1: {best_f1:.4f})")

    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_model_path)
    logger.info(f"Saved best model to {output_model_path}")

    # Save metrics (convert numpy types to Python types for YAML compatibility)
    metrics_output = {
        "model_name": best_model_name,
        "metrics": {k: float(v) for k, v in best_metrics.items()},
        "timestamp": datetime.now().isoformat(),
    }
    with open(metrics_path, "w") as f:
        yaml.dump(metrics_output, f)
    logger.info(f"Saved metrics to {metrics_path}")

    return {
        "best_model": best_model_name,
        "metrics": best_metrics,
        "model_path": output_model_path,
    }


if __name__ == "__main__":
    result = train_model()
    print(f"Training complete. Best model: {result['best_model']}")
