from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from pathlib import Path
import numpy as np

router = APIRouter(
    prefix="/api/churn",
    tags=["Customer Churn Prediction"]
)

# Load artifacts
pipeline_path = Path(__file__).parent.parent / "models" / "churn_pipeline.pkl"
features_path = Path(__file__).parent.parent / "models" / "features.pkl"

try:
    pipeline = joblib.load(pipeline_path)
    features = joblib.load(features_path)
    print("✅ Model pipeline loaded successfully")
except Exception as e:
    pipeline = None
    print(f"⚠️ Error loading model pipeline: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

@router.post("/predict")
async def predict_churn(data: dict):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs"
        )
    
    try:
        # Create DataFrame with correct feature order
        df = pd.DataFrame([data])[features]
        
        # Convert TotalCharges to numeric (handled in pipeline but good to validate)
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
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )