# app/routers/car_price_api.py
from fastapi import APIRouter
from pydantic import BaseModel
import pickle
import numpy as np
import os

router = APIRouter(prefix="/car-price", tags=["Car Price Prediction"])

# Path to ml_models folder
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models")

# Load model and scaler
with open(os.path.join(MODEL_DIR, "car_price_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

class CarFeatures(BaseModel):
    Horsepower: float
    Age: float
    Mileage: float

@router.post("/predict")
def predict_price(features: CarFeatures):
    input_data = np.array([[features.Horsepower, features.Age, features.Mileage]])
    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]
    return {"predicted_price": round(predicted_price, 2)}
