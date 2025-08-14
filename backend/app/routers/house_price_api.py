# app/routers/house_price_model.py
from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from pathlib import Path

router = APIRouter(
    prefix="/api/house-price",
    tags=["Machine Learning"]
)

# Path to ml_models folder at project root
model_path = Path(__file__).parent.parent / "ml_models" / "house_price_model.pkl"

try:
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    model = None
    print(f"⚠️ Model loading failed: {str(e)}")
    print(f"Expected model at: {model_path}")

@router.post("/predict")
async def predict_house_price(data: dict):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Required features
        required_features = ["size_sqft", "bedrooms", "bathrooms", "location", "age_years"]
        df = pd.DataFrame([data])

        # Validate input
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing}"
            )

        # Predict
        prediction = model.predict(df)[0]
        return {
            "predicted_price": round(prediction, 2),
            "currency": "USD",
            "features_used": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
