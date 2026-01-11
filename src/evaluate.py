"""Model evaluation and deployment gate module."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    threshold_checks: Dict[str, bool] = field(default_factory=dict)
    inference_tests: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ModelEvaluator:
    """Evaluates model against deployment criteria."""

    def __init__(
        self,
        deployment_config_path: str = "configs/deployment.yaml",
        model_path: str = "models/model.pkl",
        metrics_path: str = "models/metrics.yaml",
        previous_metrics_path: str = None,
    ):
        """Initialize evaluator with config and model paths."""
        with open(deployment_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.thresholds = self.config.get("thresholds", {}).get("minimum", {})
        self.regression_tolerance = self.config.get("regression_tolerance", 0.05)
        self.inference_tests = self.config.get("inference_tests", [])

        self.model = joblib.load(model_path)
        self.model_path = model_path

        with open(metrics_path, "r") as f:
            metrics_data = yaml.safe_load(f)
        self.metrics = metrics_data.get("metrics", {})

        self.previous_metrics = None
        if previous_metrics_path and Path(previous_metrics_path).exists():
            with open(previous_metrics_path, "r") as f:
                prev_data = yaml.safe_load(f)
            self.previous_metrics = prev_data.get("metrics", {})

    def _check_thresholds(self) -> Dict[str, bool]:
        """Check if metrics meet minimum thresholds."""
        results = {}
        for metric_name, min_value in self.thresholds.items():
            actual_value = self.metrics.get(metric_name, 0)
            passed = actual_value >= min_value
            results[metric_name] = passed
            status = "PASS" if passed else "FAIL"
            logger.info(
                f"  {metric_name}: {actual_value:.4f} >= {min_value} ... {status}"
            )
        return results

    def _check_regression(self) -> Dict[str, bool]:
        """Check if metrics haven't regressed from previous version."""
        if self.previous_metrics is None:
            logger.info("  No previous metrics to compare against")
            return {}

        results = {}
        for metric_name, prev_value in self.previous_metrics.items():
            curr_value = self.metrics.get(metric_name, 0)
            max_drop = prev_value * self.regression_tolerance
            passed = (prev_value - curr_value) <= max_drop
            results[f"{metric_name}_regression"] = passed
            status = "PASS" if passed else "FAIL"
            logger.info(
                f"  {metric_name}: {prev_value:.4f} -> {curr_value:.4f} "
                f"(max drop: {max_drop:.4f}) ... {status}"
            )
        return results

    def _run_inference_tests(self) -> Dict[str, bool]:
        """Run inference tests on the model."""
        results = {}

        # Create sample input data
        sample_data = pd.DataFrame(
            {
                "customer_id": ["TEST_001"],
                "age": [35],
                "income": [50000.0],
                "gender": ["M"],
                "region": ["North"],
                "tenure_months": [24],
                "monthly_charges": [65.0],
                "total_charges": [1500.0],
            }
        )

        for test in self.inference_tests:
            test_name = test["name"]
            test_type = test.get("type", "")

            try:
                if test_type == "prediction_test" or test_name == "sample_prediction":
                    # Test that model can make a prediction
                    prediction = self.model.predict(sample_data)
                    passed = len(prediction) == 1 and prediction[0] in [0, 1]
                    results[test_name] = passed
                    status = "PASS" if passed else "FAIL"
                    logger.info(f"  {test_name}: prediction={prediction[0]} ... {status}")

                elif test_type == "latency_test" or test_name == "latency_check":
                    # Test prediction latency
                    max_latency_ms = test.get("max_latency_ms", 100)
                    n_predictions = 100

                    # Warm up
                    self.model.predict(sample_data)

                    # Time predictions
                    start = time.time()
                    for _ in range(n_predictions):
                        self.model.predict(sample_data)
                    elapsed_ms = (time.time() - start) * 1000

                    avg_latency_ms = elapsed_ms / n_predictions
                    passed = elapsed_ms < max_latency_ms
                    results[test_name] = passed
                    status = "PASS" if passed else "FAIL"
                    logger.info(
                        f"  {test_name}: {n_predictions} predictions in {elapsed_ms:.1f}ms "
                        f"(avg: {avg_latency_ms:.2f}ms, max: {max_latency_ms}ms) ... {status}"
                    )

                elif test_type == "determinism_test" or test_name == "determinism_check":
                    # Test that predictions are deterministic
                    pred1 = self.model.predict(sample_data)
                    pred2 = self.model.predict(sample_data)
                    passed = np.array_equal(pred1, pred2)
                    results[test_name] = passed
                    status = "PASS" if passed else "FAIL"
                    logger.info(f"  {test_name}: pred1={pred1[0]}, pred2={pred2[0]} ... {status}")

            except Exception as e:
                results[test_name] = False
                logger.error(f"  {test_name}: ERROR - {e}")

        return results

    def evaluate(self) -> EvaluationResult:
        """Run all evaluation checks.

        Returns:
            EvaluationResult with pass/fail status and details
        """
        errors = []

        # Run threshold checks
        logger.info("\nThreshold Checks:")
        threshold_results = self._check_thresholds()

        # Run regression checks
        logger.info("\nRegression Checks:")
        regression_results = self._check_regression()
        threshold_results.update(regression_results)

        # Run inference tests
        logger.info("\nInference Tests:")
        inference_results = self._run_inference_tests()

        # Determine overall pass/fail
        all_threshold_passed = all(threshold_results.values()) if threshold_results else True
        all_inference_passed = all(inference_results.values()) if inference_results else True
        overall_passed = all_threshold_passed and all_inference_passed

        return EvaluationResult(
            passed=overall_passed,
            metrics=self.metrics,
            threshold_checks=threshold_results,
            inference_tests=inference_results,
            errors=errors,
        )


def run_evaluation(
    deployment_config_path: str = "configs/deployment.yaml",
    model_path: str = "models/model.pkl",
    metrics_path: str = "models/metrics.yaml",
    previous_metrics_path: str = None,
    output_path: str = "models/evaluation_report.yaml",
) -> EvaluationResult:
    """Main entry point for model evaluation.

    Args:
        deployment_config_path: Path to deployment config
        model_path: Path to trained model
        metrics_path: Path to metrics YAML
        previous_metrics_path: Path to previous metrics for regression check
        output_path: Path to save evaluation report

    Returns:
        EvaluationResult with evaluation details
    """
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 50)

    evaluator = ModelEvaluator(
        deployment_config_path=deployment_config_path,
        model_path=model_path,
        metrics_path=metrics_path,
        previous_metrics_path=previous_metrics_path,
    )

    result = evaluator.evaluate()

    # Log overall result
    logger.info("\n" + "=" * 50)
    if result.passed:
        logger.info("EVALUATION PASSED - Model approved for deployment")
    else:
        logger.info("EVALUATION FAILED - Model NOT approved for deployment")
    logger.info("=" * 50)

    # Save report
    report = {
        "passed": result.passed,
        "metrics": result.metrics,
        "threshold_checks": result.threshold_checks,
        "inference_tests": result.inference_tests,
        "errors": result.errors,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(report, f)
    logger.info(f"Evaluation report saved to {output_path}")

    return result


if __name__ == "__main__":
    result = run_evaluation()
    exit(0 if result.passed else 1)
