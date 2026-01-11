"""Airflow DAG for orchestrating the ML training pipeline."""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Import pipeline modules
import sys
sys.path.insert(0, "/opt/airflow")
from src import schema, features, train, evaluate


# Default arguments for the DAG
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def validate_schema_task(**context):
    """Validate data against schema."""
    result = schema.validate_data(
        data_path="data/raw/customers.csv",
        schema_path="configs/schema.yaml",
        output_path="data/validation_report.json",
    )

    if not result.is_valid:
        raise ValueError(f"Schema validation failed: {result.errors}")

    # Push stats to XCom
    context["ti"].xcom_push(key="validation_stats", value=result.stats)
    return result.is_valid


def run_feature_engineering_task(**context):
    """Run feature engineering pipeline."""
    features.run_feature_engineering(
        input_path="data/raw/customers.csv",
        output_path="data/processed/customers_featured.csv",
        config_path="configs/features.yaml",
        transformer_path="models/feature_engineer.pkl",
    )


def train_model_task(**context):
    """Train model and log to MLflow."""
    result = train.train_model(
        data_path="data/processed/customers_featured.csv",
        feature_config_path="configs/features.yaml",
        model_config_path="configs/model.yaml",
        output_model_path="models/model.pkl",
        metrics_path="models/metrics.yaml",
    )

    # Push result to XCom
    context["ti"].xcom_push(key="training_result", value=result)
    return result


def evaluate_model_task(**context):
    """Evaluate model against deployment criteria."""
    result = evaluate.run_evaluation(
        deployment_config_path="configs/deployment.yaml",
        model_path="models/model.pkl",
        metrics_path="models/metrics.yaml",
        previous_metrics_path=None,
        output_path="models/evaluation_report.yaml",
    )

    # Push passed status to XCom
    context["ti"].xcom_push(key="evaluation_passed", value=result.passed)
    return result.passed


def check_deployment_gate(**context):
    """Branch based on evaluation result."""
    ti = context["ti"]
    passed = ti.xcom_pull(task_ids="evaluate_model", key="evaluation_passed")

    if passed:
        return "deploy_model"
    else:
        return "notify_failure"


def deploy_model_task(**context):
    """Deploy model by copying to versioned filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_path = Path("models/model.pkl")
    dest_path = Path(f"models/model_{timestamp}.pkl")

    if source_path.exists():
        shutil.copy(source_path, dest_path)
        print(f"Model deployed to {dest_path}")
    else:
        raise FileNotFoundError(f"Model not found at {source_path}")


def notify_failure_task(**context):
    """Log that deployment was blocked."""
    print("DEPLOYMENT BLOCKED: Model did not pass evaluation criteria.")
    print("Please review the evaluation report at models/evaluation_report.yaml")


# Define the DAG
with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="End-to-end ML training pipeline with validation and deployment gates",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "churn"],
) as dag:

    # Task definitions
    validate = PythonOperator(
        task_id="validate_schema",
        python_callable=validate_schema_task,
        provide_context=True,
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering_task,
        provide_context=True,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
        provide_context=True,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_task,
        provide_context=True,
    )

    deployment_gate = BranchPythonOperator(
        task_id="deployment_gate",
        python_callable=check_deployment_gate,
        provide_context=True,
    )

    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model_task,
        provide_context=True,
    )

    notify_failure = PythonOperator(
        task_id="notify_failure",
        python_callable=notify_failure_task,
        provide_context=True,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # Define task dependencies
    validate >> feature_engineering >> train_task >> evaluate_task >> deployment_gate
    deployment_gate >> [deploy, notify_failure]
    [deploy, notify_failure] >> end
