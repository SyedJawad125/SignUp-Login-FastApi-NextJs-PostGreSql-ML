import pickle
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# Paths
ml_dir = Path(__file__).parent.parent / "ml_models"
ml_dir.mkdir(exist_ok=True)
model_path = ml_dir / "pca_model.pkl"

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# If model doesn't exist, create and save it
if not model_path.exists():
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    with open(model_path, "wb") as f:
        pickle.dump(pca, f)

    print(f"✅ PCA model trained and saved at {model_path}")
else:
    with open(model_path, "rb") as f:
        pca = pickle.load(f)
    X_pca = pca.transform(X)
    print(f"✅ PCA model loaded from {model_path}")

# Store PCA results
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Target"] = y

def get_pca_results():
    """Return PCA results on the Iris dataset."""
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_data": df_pca.to_dict(orient="records"),
    }

def transform_new_data(data):
    """Transform new data using fitted PCA."""
    transformed = pca.transform(data)
    return transformed.tolist()
