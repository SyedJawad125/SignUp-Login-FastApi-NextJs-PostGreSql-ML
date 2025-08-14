from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from pathlib import Path

router = APIRouter(
    prefix="/api/churn",
    tags=["Customer Churn Prediction"]
)

# Paths
MODEL_DIR = Path(__file__).parent.parent / "ml_models"
CHURN_MODEL_PATH = MODEL_DIR / "churn_pipeline.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

# Load model & features
try:
    pipeline = joblib.load(CHURN_MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    print("✅ Churn model loaded successfully")
except Exception as e:
    pipeline = None
    print(f"⚠️ Error loading churn model: {e}")

@router.post("/predict")
async def predict_churn(data: dict):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Arrange data
        df = pd.DataFrame([data])[features]
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Predict
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0][1]

        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "probability": float(probability),
            "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
            "features_used": {k: v for k, v in data.items() if k in features}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
