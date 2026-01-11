"""FastAPI serving endpoint for model predictions."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "file")  # "file" or "mlflow"
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/churn_predictor/latest")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using trained ML model",
    version="1.0.0",
)

# Global model variable
MODEL = None
MODEL_LOADED_AT = None
MODEL_INFO = {}


class CustomerInput(BaseModel):
    """Input schema for customer data."""
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., ge=18, le=120, description="Customer age")
    income: Optional[float] = Field(None, ge=0, description="Annual income")
    gender: str = Field(..., description="Gender (M, F, Other)")
    region: str = Field(..., description="Region (North, South, East, West)")
    tenure_months: int = Field(..., ge=0, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: Optional[float] = Field(None, ge=0, description="Total charges to date")


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    customer_id: str
    prediction: int
    churn_probability: float
    confidence: float


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response schema for model info."""
    source: str
    uri: str
    loaded_at: Optional[str]


@app.on_event("startup")
async def load_model():
    """Load model at startup from file or MLflow."""
    global MODEL, MODEL_LOADED_AT, MODEL_INFO

    try:
        if MODEL_SOURCE == "mlflow":
            # Load from MLflow Model Registry
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"Loading model from MLflow: {MLFLOW_MODEL_URI}")
            MODEL = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
            MODEL_INFO = {"source": "mlflow", "uri": MLFLOW_MODEL_URI}
            logger.info(f"Model loaded from MLflow: {MLFLOW_MODEL_URI}")
        else:
            # Load from local file
            import joblib
            if Path(MODEL_PATH).exists():
                MODEL = joblib.load(MODEL_PATH)
                MODEL_INFO = {"source": "file", "path": MODEL_PATH}
                logger.info(f"Model loaded from file: {MODEL_PATH}")
            else:
                logger.warning(f"Model file not found at {MODEL_PATH}")
                return

        MODEL_LOADED_AT = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL = None


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CustomerInput) -> PredictionResponse:
    """Make a churn prediction for a customer.

    Args:
        input_data: Customer data for prediction

    Returns:
        Prediction result with probability and confidence
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Convert input to DataFrame
    df = pd.DataFrame([input_data.model_dump()])

    # Make prediction
    try:
        prediction = MODEL.predict(df)[0]
        probabilities = MODEL.predict_proba(df)[0]
        churn_probability = probabilities[1]
        confidence = max(probabilities)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    latency_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Prediction for {input_data.customer_id}: "
        f"churn={prediction}, prob={churn_probability:.3f}, "
        f"latency={latency_ms:.1f}ms"
    )

    return PredictionResponse(
        customer_id=input_data.customer_id,
        prediction=int(prediction),
        churn_probability=float(churn_probability),
        confidence=float(confidence),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        Health status and model load status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Get information about the loaded model.

    Returns:
        Model source, URI, and load timestamp
    """
    return ModelInfoResponse(
        source=MODEL_INFO.get("source", "unknown"),
        uri=MODEL_INFO.get("uri") or MODEL_INFO.get("path", "unknown"),
        loaded_at=MODEL_LOADED_AT,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
