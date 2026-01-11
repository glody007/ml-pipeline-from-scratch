"""Unit tests for trained model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def model():
    """Load the trained model."""
    model_path = Path("models/model.pkl")
    if not model_path.exists():
        pytest.skip("Model file not found. Run training first.")
    return joblib.load(model_path)


@pytest.fixture
def sample_input():
    """Create sample input DataFrame."""
    return pd.DataFrame({
        "customer_id": ["TEST_001"],
        "age": [35],
        "income": [50000.0],
        "gender": ["M"],
        "region": ["North"],
        "tenure_months": [24],
        "monthly_charges": [65.0],
        "total_charges": [1500.0],
    })


class TestModel:
    """Test suite for trained model."""

    def test_model_loads(self, model):
        """Model should load successfully."""
        assert model is not None

    def test_model_predicts(self, model, sample_input):
        """Model should make predictions with correct values."""
        prediction = model.predict(sample_input)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]

    def test_model_predicts_proba(self, model, sample_input):
        """Model should return probabilities that sum to 1."""
        proba = model.predict_proba(sample_input)

        assert proba.shape == (1, 2)
        assert np.isclose(proba.sum(axis=1), 1.0).all()
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_model_handles_missing(self, model, sample_input):
        """Model should handle None values in nullable fields."""
        df = sample_input.copy()
        df.loc[0, "income"] = None
        df.loc[0, "total_charges"] = None

        # Should not raise an error
        prediction = model.predict(df)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]

    def test_model_handles_unknown_category(self, model, sample_input):
        """Model should handle unseen category values."""
        df = sample_input.copy()
        df.loc[0, "region"] = "Unknown_Region"

        # Should not raise an error (OneHotEncoder has handle_unknown='ignore')
        prediction = model.predict(df)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]

    def test_prediction_deterministic(self, model, sample_input):
        """Two predictions on same input should be equal."""
        pred1 = model.predict(sample_input)
        pred2 = model.predict(sample_input)

        assert np.array_equal(pred1, pred2)

    def test_batch_prediction(self, model, sample_input):
        """Model should handle batch predictions."""
        df = pd.concat([sample_input] * 10, ignore_index=True)
        df["customer_id"] = [f"TEST_{i:03d}" for i in range(10)]

        prediction = model.predict(df)

        assert len(prediction) == 10
        assert all(p in [0, 1] for p in prediction)

    def test_proba_matches_prediction(self, model, sample_input):
        """Predicted class should match highest probability."""
        prediction = model.predict(sample_input)[0]
        proba = model.predict_proba(sample_input)[0]

        predicted_class = np.argmax(proba)
        assert prediction == predicted_class
