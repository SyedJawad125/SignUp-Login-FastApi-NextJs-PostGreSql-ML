from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/churn",
    tags=["Customer Churn Prediction"]
)

# Paths
MODEL_DIR = Path(__file__).parent.parent / "ml_models"
CHURN_MODEL_PATH = MODEL_DIR / "churn_pipeline.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

# Initialize as None
pipeline = None
features = None

# Try to load model
try:
    if CHURN_MODEL_PATH.exists() and FEATURES_PATH.exists():
        pipeline = joblib.load(CHURN_MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        logger.info("✅ Churn model loaded successfully")
        logger.info(f"Features: {features}")
    else:
        logger.warning("⚠️ Model files not found. Please train the model first.")
except Exception as e:
    logger.error(f"❌ Error loading churn model: {e}")
    pipeline = None
    features = None

@router.post("/predict")
async def predict_churn(data: dict):
    if pipeline is None or features is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first using /api/churn/train endpoint."
        )

    try:
        # Convert SeniorCitizen to string if it exists (categorical feature)
        if 'SeniorCitizen' in data:
            data['SeniorCitizen'] = str(data['SeniorCitizen'])
        
        # Create DataFrame with expected features
        df = pd.DataFrame([data])
        
        # Ensure all expected features are present
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing features: {list(missing_features)}. Required features: {features}"
            )
        
        # Select only the features the model expects
        df = df[features]
        
        # Convert TotalCharges to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Handle NaN values
        if df.isnull().any().any():
            raise HTTPException(
                status_code=400, 
                detail="Invalid data: NaN values detected after preprocessing"
            )

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
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Add training endpoint
@router.post("/train")
async def train_model():
    try:
        from app.models.churn_model import train_churn_model
        train_churn_model()
        
        # Reload the model
        global pipeline, features
        pipeline = joblib.load(CHURN_MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        
        return {"status": "success", "message": "Model trained and loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/status")
async def model_status():
    return {
        "model_loaded": pipeline is not None,
        "features_loaded": features is not None,
        "features": features if features else []
    }