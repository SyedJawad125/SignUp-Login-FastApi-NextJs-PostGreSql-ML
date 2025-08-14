from fastapi import APIRouter, Response
from pydantic import BaseModel
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import io
from app.models import pca_model

router = APIRouter(prefix="/pca", tags=["PCA"])

class Features(BaseModel):
    data: List[List[float]]

@router.get("/results")
def get_pca_results():
    """Return PCA results on full Iris dataset"""
    return pca_model.get_pca_results()

@router.post("/transform")
def pca_transform(features: Features):
    """Apply PCA transform to custom data"""
    return {"transformed_data": pca_model.transform_new_data(features.data)}

@router.get("/plot")
def pca_plot():
    """Return PCA plot as PNG"""
    df_pca = pd.DataFrame(pca_model.get_pca_results()["pca_data"])

    plt.figure(figsize=(8, 6))
    colors = ["red", "green", "blue"]
    targets = [0, 1, 2]
    for target, color in zip(targets, colors):
        subset = df_pca[df_pca["Target"] == target]
        plt.scatter(subset["PC1"], subset["PC2"], color=color, label=f"Class {target}")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA on Iris Dataset")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
