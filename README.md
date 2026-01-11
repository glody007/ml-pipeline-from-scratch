# Production ML Pipeline Demo

A complete end-to-end machine learning pipeline demonstrating production best practices for customer churn prediction.

> **📖 Full Documentation**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture, diagrams, and configuration options.

## Overview

This project demonstrates:
- **Schema Validation**: Ensure data quality before processing
- **Feature Engineering**: Reproducible feature transformations
- **Model Training**: Multiple models with MLflow tracking
- **Evaluation Gate**: Quality thresholds before deployment
- **Airflow Orchestration**: Automated pipeline execution
- **DVC Versioning**: Track data and model versions
- **API Serving**: FastAPI for real-time predictions

## Project Structure

```
ml-pipeline-prod/
├── configs/
│   ├── schema.yaml          # Data schema definition
│   ├── features.yaml        # Feature engineering config
│   ├── model.yaml           # Model hyperparameters
│   └── deployment.yaml      # Quality gate thresholds
├── src/
│   ├── schema.py            # Data validation
│   ├── features.py          # Feature transformations
│   ├── train.py             # Model training with MLflow
│   ├── evaluate.py          # Deployment gate checks
│   └── api.py               # FastAPI serving endpoint
├── dags/
│   └── ml_pipeline_dag.py   # Airflow DAG
├── tests/
│   ├── test_schema.py
│   ├── test_features.py
│   └── test_model.py
├── scripts/
│   └── generate_sample_data.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── dvc.yaml
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py
```

## Running the Pipeline

### Using DVC

```bash
# Initialize DVC (first time only)
dvc init

# Run the full pipeline
dvc repro

# View metrics
dvc metrics show
```

### Using Airflow

```bash
# Start Airflow with Docker Compose
docker-compose up -d

# Access the Airflow UI
# Open http://localhost:8080
# Login: admin / admin

# Trigger the DAG from the UI or CLI
docker-compose exec airflow airflow dags trigger ml_training_pipeline
```

### Running Individual Steps

```bash
# Validate data
python src/schema.py

# Run feature engineering
python src/features.py

# Train models
python src/train.py

# Evaluate model
python src/evaluate.py
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schema.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## API

### Build and Run

```bash
# Build Docker image
docker build -t ml-pipeline-api .

# Run container
docker run -p 8000:8000 ml-pipeline-api
```

### Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "age": 35,
    "income": 50000,
    "gender": "M",
    "region": "North",
    "tenure_months": 24,
    "monthly_charges": 65.0,
    "total_charges": 1500.0
  }'

# Model info
curl http://localhost:8000/model-info
```

## Key Components

### Schema Validation

The schema validator (`src/schema.py`) ensures data quality by checking:
- Required columns exist
- Data types match expectations
- Values fall within valid ranges
- Categorical values are in allowed sets
- Nullable constraints are respected
- Unique constraints are enforced

### Feature Engineering

The feature engineer (`src/features.py`) creates:
- Derived features (ratios, calculations)
- Binary features (threshold-based flags)
- Binned features (age groups, tenure groups)
- Handles missing values and infinite values

### Model Training

The training module (`src/train.py`) provides:
- Multiple model comparison (Random Forest, Gradient Boosting, Logistic Regression)
- Cross-validation for robust evaluation
- MLflow experiment tracking
- Automatic best model selection

### Evaluation Gate

The evaluator (`src/evaluate.py`) checks:
- Minimum metric thresholds (accuracy, precision, recall, F1, ROC-AUC)
- Regression from previous model version
- Inference tests (latency, determinism, sample prediction)

### Orchestration

The Airflow DAG (`dags/ml_pipeline_dag.py`) orchestrates:
- Sequential pipeline stages
- Branching logic for deployment decisions
- XCom for passing data between tasks
- Failure notifications

### Versioning

DVC (`dvc.yaml`) tracks:
- Data file versions
- Model artifacts
- Metrics over time
- Pipeline dependencies
