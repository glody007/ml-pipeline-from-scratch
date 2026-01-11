# ML Pipeline Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Components](#components)
4. [Data Flow](#data-flow)
5. [Running the Pipeline](#running-the-pipeline)
6. [Configuration](#configuration)
7. [Model Versioning](#model-versioning)
8. [API Serving](#api-serving)
9. [Development vs Production](#development-vs-production)

---

## Overview

This is a production-grade ML pipeline for **Customer Churn Prediction**. It demonstrates:

- **Data Validation**: Schema-based data quality checks
- **Feature Engineering**: Reproducible feature transformations
- **Model Training**: Multiple models with experiment tracking
- **Evaluation Gate**: Quality thresholds before deployment
- **Model Serving**: REST API for real-time predictions
- **Orchestration**: Airflow for scheduling, DVC for versioning

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ML PIPELINE ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│   Schema     │────▶│   Feature    │────▶│   Model      │
│  (CSV)       │     │  Validation  │     │  Engineering │     │  Training    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │                    │                    │
                            ▼                    ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
                     │  Validation  │     │  Processed   │     │   MLflow     │
                     │  Report      │     │  Data        │     │  Tracking    │
                     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Deployed    │◀────│  Evaluation  │◀────│   Model      │◀────│   Model      │
│  Model       │     │  Gate        │     │   Artifacts  │     │  Registry    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SERVING LAYER                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                       │
│  │  FastAPI    │    │   /predict  │    │   /health   │                       │
│  │  Server     │───▶│   endpoint  │    │   endpoint  │                       │
│  └─────────────┘    └─────────────┘    └─────────────┘                       │
└──────────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION OPTIONS                               │
│                                                                               │
│  ┌─────────────────┐         ┌─────────────────┐         ┌────────────────┐  │
│  │    AIRFLOW      │         │      DVC        │         │    MANUAL      │  │
│  │  (Scheduling)   │         │  (Versioning)   │         │   (Scripts)    │  │
│  │                 │         │                 │         │                │  │
│  │  - DAG-based    │         │  - Git-based    │         │  - Direct      │  │
│  │  - UI dashboard │         │  - Reproducible │         │  - Debugging   │  │
│  │  - Retries      │         │  - Data version │         │  - Testing     │  │
│  └─────────────────┘         └─────────────────┘         └────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Directory Structure

```
ml-pipeline-prod/
├── configs/                 # Configuration files
│   ├── schema.yaml         # Data schema definition
│   ├── features.yaml       # Feature engineering config
│   ├── model.yaml          # Model hyperparameters
│   └── deployment.yaml     # Evaluation thresholds
│
├── src/                    # Source code
│   ├── schema.py          # Data validation
│   ├── features.py        # Feature transformations
│   ├── train.py           # Model training + MLflow
│   ├── evaluate.py        # Deployment gate
│   └── api.py             # FastAPI serving
│
├── dags/                   # Airflow DAGs
│   └── ml_pipeline_dag.py # Pipeline orchestration
│
├── data/                   # Data files (gitignored)
│   ├── raw/               # Raw input data
│   └── processed/         # Transformed data
│
├── models/                 # Model artifacts
│   ├── model.pkl          # Trained model
│   └── metrics.yaml       # Training metrics
│
├── mlruns/                 # MLflow experiment data
├── tests/                  # Unit tests
├── docker-compose.yaml     # Services definition
├── Dockerfile             # API container
├── requirements.txt       # Training dependencies
└── requirements-api.txt   # Serving dependencies
```

### Component Details

| Component | File | Purpose |
|-----------|------|---------|
| **Schema Validator** | `src/schema.py` | Validates data types, ranges, nullability |
| **Feature Engineer** | `src/features.py` | Creates derived, binary, binned features |
| **Model Trainer** | `src/train.py` | Trains models, logs to MLflow |
| **Evaluator** | `src/evaluate.py` | Checks thresholds, runs inference tests |
| **API Server** | `src/api.py` | Serves predictions via REST |
| **DAG** | `dags/ml_pipeline_dag.py` | Orchestrates pipeline in Airflow |

---

## Data Flow

```
1. INPUT
   └── data/raw/customers.csv
          │
          ▼
2. VALIDATION (schema.py)
   ├── Check columns exist
   ├── Validate data types
   ├── Check value ranges
   └── Output: data/validation_report.json
          │
          ▼
3. FEATURE ENGINEERING (features.py)
   ├── Derived: charges_per_month, income_ratio
   ├── Binary: is_high_value
   ├── Binned: age_group, tenure_group
   └── Output: data/processed/customers_featured.csv
          │
          ▼
4. TRAINING (train.py)
   ├── Load processed data
   ├── Train: RandomForest, GradientBoosting, LogisticRegression
   ├── Cross-validation
   ├── Log to MLflow (metrics, params, model)
   ├── Register to Model Registry
   └── Output: models/model.pkl, models/metrics.yaml
          │
          ▼
5. EVALUATION (evaluate.py)
   ├── Check metric thresholds
   ├── Check regression from previous
   ├── Run inference tests (latency, determinism)
   └── Output: models/evaluation_report.yaml
          │
          ▼
6. SERVING (api.py)
   ├── Load model (file or MLflow)
   ├── POST /predict → prediction
   └── GET /health → status
```

---

## Running the Pipeline

### Option 1: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d --build

# Services:
# - Airflow UI: http://localhost:8080 (admin/password)
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Option 2: Airflow Only

```bash
# Start Airflow
docker-compose up airflow -d

# Open UI, trigger DAG: ml_training_pipeline
# Login: admin / password
```

### Option 3: DVC Pipeline

```bash
# Activate virtualenv
source venv/bin/activate

# Run full pipeline
dvc repro

# View metrics
dvc metrics show

# Compare versions
dvc diff
```

### Option 4: Manual Execution

```bash
source venv/bin/activate

# Step 1: Generate sample data
python scripts/generate_sample_data.py

# Step 2: Validate data
python src/schema.py

# Step 3: Feature engineering
python src/features.py

# Step 4: Train models
python src/train.py

# Step 5: Evaluate
python src/evaluate.py

# Step 6: Serve
uvicorn src.api:app --port 8000
```

---

## Configuration

### configs/schema.yaml
```yaml
schema:
  columns:
    customer_id:
      type: string
      nullable: false
      unique: true
    age:
      type: integer
      min: 18
      max: 120
    # ... more columns
```

### configs/features.yaml
```yaml
features:
  numeric:
    - age
    - income
    - tenure_months
  categorical:
    - gender
    - region
  derived:
    charges_per_month:
      formula: "total_charges / (tenure_months + 1)"
  target: churn
```

### configs/model.yaml
```yaml
experiment:
  name: churn_prediction

models:
  random_forest:
    class: sklearn.ensemble.RandomForestClassifier
    params:
      n_estimators: 100
      max_depth: 10
```

### configs/deployment.yaml
```yaml
thresholds:
  minimum:
    accuracy: 0.75
    precision: 0.70
    recall: 0.65
    f1: 0.70
    roc_auc: 0.75
```

---

## Model Versioning

### MLflow (Experiment Tracking)

```
┌─────────────────────────────────────────────────────────────┐
│                    MLFLOW ARCHITECTURE                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Experiment  │───▶│     Runs     │───▶│   Artifacts  │   │
│  │  churn_pred  │    │  - rf_v1     │    │  - model/    │   │
│  │              │    │  - gb_v1     │    │  - metrics   │   │
│  │              │    │  - lr_v1     │    │              │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                             │                                │
│                             ▼                                │
│                    ┌──────────────┐                         │
│                    │   Model      │                         │
│                    │   Registry   │                         │
│                    │              │                         │
│                    │ churn_pred/1 │                         │
│                    │ churn_pred/2 │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

**View MLflow UI:**
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000
```

**Load model from registry:**
```python
import mlflow
model = mlflow.sklearn.load_model("models:/churn_predictor/latest")
```

### DVC (Data/Model Versioning)

```
┌─────────────────────────────────────────────────────────────┐
│                     DVC ARCHITECTURE                         │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │    Git       │    │    DVC       │    │   Remote     │   │
│  │  (metadata)  │◀──▶│  (tracking)  │◀──▶│  (storage)   │   │
│  │              │    │              │    │              │   │
│  │ .dvc files   │    │ .dvc/cache   │    │  S3/GCS     │   │
│  │ dvc.lock     │    │              │    │              │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Commands:**
```bash
dvc add data/raw/customers.csv  # Track file
dvc push                        # Upload to remote
dvc pull                        # Download from remote
dvc checkout                    # Restore version
```

---

## API Serving

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Make prediction |
| GET | `/docs` | Swagger UI |

### Request/Response

**POST /predict**
```json
// Request
{
  "customer_id": "CUST_001",
  "age": 35,
  "income": 50000,
  "gender": "M",
  "region": "North",
  "tenure_months": 24,
  "monthly_charges": 65.0,
  "total_charges": 1500.0
}

// Response
{
  "customer_id": "CUST_001",
  "prediction": 0,
  "churn_probability": 0.23,
  "confidence": 0.77
}
```

### Model Source Configuration

```yaml
# docker-compose.yaml
environment:
  # Option 1: Load from file
  - MODEL_SOURCE=file
  - MODEL_PATH=models/model.pkl

  # Option 2: Load from MLflow
  - MODEL_SOURCE=mlflow
  - MLFLOW_MODEL_URI=models:/churn_predictor/latest
```

### Volume Requirements by Model Source

| MODEL_SOURCE | MLflow Setup | Volumes Needed |
|--------------|--------------|----------------|
| `file` | N/A | `./models:/app/models` |
| `mlflow` | File-based (`file:///app/mlruns`) | `./models` + `./mlruns:/app/mlruns` |
| `mlflow` | Remote server (`http://mlflow:5000`) | `./models` only (no mlruns) |

**Development (file-based MLflow):**
```yaml
volumes:
  - ./models:/app/models
  - ./mlruns:/app/mlruns    # Needed for local MLflow registry
  - ./src:/app/src          # Hot reload
```

**Production (remote MLflow server):**
```yaml
environment:
  - MODEL_SOURCE=mlflow
  - MLFLOW_TRACKING_URI=http://mlflow-server:5000
  - MLFLOW_MODEL_URI=models:/churn_predictor/Production
volumes:
  - ./models:/app/models    # Fallback only, no mlruns needed
```

---

## Development vs Production

### Docker Build Targets

```dockerfile
# Development: hot reload, full deps
docker build --target development -t api:dev .

# Production: minimal image, optimized
docker build --target production -t api:prod .
```

### Comparison

| Aspect | Development | Production |
|--------|-------------|------------|
| **Image size** | ~800MB | ~300MB |
| **Hot reload** | ✅ Yes (`--reload`) | ❌ No |
| **Source mount** | ✅ Volumes | ❌ Copied into image |
| **Debug tools** | ✅ Included | ❌ Minimal |
| **Build target** | `development` | `production` |
| **Model source** | `file` (local pkl) | `mlflow` (registry) |
| **MLflow** | File-based (`./mlruns`) | Remote server |
| **Secrets** | Hardcoded/env | Secrets manager |

### docker-compose.yaml: Development vs Production

```yaml
# ============= DEVELOPMENT =============
api:
  build:
    context: .
    target: development           # Full deps, hot reload
  environment:
    - MODEL_SOURCE=file           # Load from local file
    - MODEL_PATH=models/model.pkl
  volumes:
    - ./models:/app/models
    - ./src:/app/src              # Mount source for hot reload
    - ./configs:/app/configs

# ============= PRODUCTION =============
api:
  build:
    context: .
    target: production            # Minimal image
  environment:
    - MODEL_SOURCE=mlflow         # Load from registry
    - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    - MLFLOW_MODEL_URI=models:/churn_predictor/Production
  volumes:
    - ./models:/app/models        # Fallback only
    # No ./src mount - code baked into image
    # No ./mlruns - using remote MLflow
```

### Why Separate Training and Serving?

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEPARATION OF CONCERNS                        │
├─────────────────────────────┬───────────────────────────────────┤
│         TRAINING            │            SERVING                 │
├─────────────────────────────┼───────────────────────────────────┤
│ Runs: Batch, scheduled      │ Runs: Always on, real-time        │
│ Compute: CPU/GPU heavy      │ Compute: Lightweight              │
│ Duration: Minutes to hours  │ Duration: Milliseconds            │
│ Scaling: Vertical           │ Scaling: Horizontal               │
│ Dependencies: Heavy (MLflow,│ Dependencies: Light (FastAPI,     │
│   sklearn, pandas, etc.)    │   joblib, sklearn)                │
│ Container: Airflow          │ Container: API                    │
│ Code: train.py, evaluate.py │ Code: api.py, features.py         │
└─────────────────────────────┴───────────────────────────────────┘
```

**Benefits:**
- API container stays small and fast
- Training can use GPU instances
- Scale serving independently
- Update model without redeploying API (MLflow)
- Different SLAs for training vs serving

### Production Checklist

- [ ] Change `target: production` in docker-compose
- [ ] Remove `./src` volume mount
- [ ] Set `MODEL_SOURCE=mlflow`
- [ ] Configure remote MLflow server
- [ ] Set up DVC remote storage (S3/GCS)
- [ ] Add monitoring/alerting (Prometheus/Grafana)
- [ ] Configure secrets management (Vault/AWS Secrets)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Add health check endpoints monitoring
- [ ] Configure auto-scaling for API
- [ ] Set up logging aggregation (ELK/CloudWatch)

---

## Quick Reference

```bash
# Start everything
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Rebuild API only
docker-compose up -d --build api

# Run tests
pytest tests/ -v

# Check MLflow
mlflow ui --port 5000

# Run DVC pipeline
dvc repro

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"TEST","age":35,"income":50000,"gender":"M","region":"North","tenure_months":24,"monthly_charges":65,"total_charges":1500}'
```
