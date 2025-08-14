# app/models/house_price_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
import os

def train_and_save_model():
    # Dataset path
    dataset_path = Path(__file__).parent.parent / "datasets" / "train.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Features & target
    features = ["size_sqft", "bedrooms", "bathrooms", "location", "age_years"]
    X = df[features]
    y = df["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure ml_models directory exists at project root
    model_dir = Path(__file__).parent.parent / "ml_models"
    model_dir.mkdir(exist_ok=True)

    # Save model
    model_path = model_dir / "house_price_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model successfully saved to: {model_path}")

    return model

if __name__ == "__main__":
    train_and_save_model()
