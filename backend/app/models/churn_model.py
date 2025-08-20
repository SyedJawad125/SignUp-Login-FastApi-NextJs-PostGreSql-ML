import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path

# Directory for ML models
MODEL_DIR = Path(__file__).parent.parent / "ml_models"
MODEL_DIR.mkdir(exist_ok=True)

CHURN_MODEL_PATH = MODEL_DIR / "churn_pipeline.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

def train_churn_model():
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "datasets" / "telco_customer_churn_big.csv"
    df = pd.read_csv(dataset_path)

    # Data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Target encoding
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Features
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    features = numerical_cols + categorical_cols

    X = df[features]
    y = df['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        # Evaluation
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        print(f"\n{name} Evaluation:")
        print(classification_report(y_test, y_pred, digits=4))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Pick best model (Random Forest)
    best_model = trained_models['Random Forest']

    # Save artifacts
    joblib.dump(best_model, CHURN_MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)

    print(f"\n✅ Model training complete. Saved to {CHURN_MODEL_PATH} and {FEATURES_PATH}")

if __name__ == "__main__":
    train_churn_model()
