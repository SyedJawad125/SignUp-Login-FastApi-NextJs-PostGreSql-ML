# # car_price_model.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# import pickle

# # Expanded dataset
# data = {
#     'Horsepower': [130, 250, 190, 300, 210, 220, 180, 160, 280, 200, 150, 170],
#     'Age': [5, 3, 8, 2, 6, 4, 7, 10, 1, 5, 9, 6],
#     'Mileage': [50000, 20000, 30000, 15000, 40000, 25000, 60000, 80000, 12000, 45000, 90000, 70000],
#     'Price': [20000, 35000, 25000, 45000, 28000, 30000, 22000, 18000, 50000, 26000, 15000, 21000]
# }
# df = pd.DataFrame(data)

# # Features and target
# X = df[['Horsepower', 'Age', 'Mileage']]
# y = df['Price']

# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Save model and scaler
# with open("car_price_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# print("✅ Car price model & scaler saved.")




# app/models/car_price_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_car_price_model():
    # Expanded dataset
    data = {
        'Horsepower': [130, 250, 190, 300, 210, 220, 180, 160, 280, 200, 150, 170],
        'Age': [5, 3, 8, 2, 6, 4, 7, 10, 1, 5, 9, 6],
        'Mileage': [50000, 20000, 30000, 15000, 40000, 25000, 60000, 80000, 12000, 45000, 90000, 70000],
        'Price': [20000, 35000, 25000, 45000, 28000, 30000, 22000, 18000, 50000, 26000, 15000, 21000]
    }
    df = pd.DataFrame(data)

    # Features & target
    X = df[['Horsepower', 'Age', 'Mileage']]
    y = df['Price']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create ml_models directory at root
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model and scaler
    with open(os.path.join(MODEL_DIR, "car_price_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Model and scaler saved in {MODEL_DIR}")

if __name__ == "__main__":
    train_car_price_model()
