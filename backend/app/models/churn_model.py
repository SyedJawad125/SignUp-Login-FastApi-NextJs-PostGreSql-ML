import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from pathlib import Path

def train_churn_model():
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "datasets" / "telco_churn.csv"
    df = pd.read_csv(dataset_path)
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Convert target to binary
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Feature selection
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod']
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    features = numerical_cols + categorical_cols
    X = df[features]
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Train models with pipeline
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced')
    }
    
    trained_models = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        print(f"\n{name} Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.2f}")
    
    # Save best model (Random Forest) and preprocessing pipeline
    model_dir = Path(__file__).parent
    model_dir.mkdir(exist_ok=True)
    
    # Save the entire pipeline (includes preprocessing)
    joblib.dump(trained_models['Random Forest'], model_dir / "churn_pipeline.pkl")
    joblib.dump(features, model_dir / "features.pkl")
    
    print("\n✅ Model training complete")

if __name__ == "__main__":
    train_churn_model()