from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.model import CNNModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["CNN Image Classifier"])

# Global model instance
cnn_model = CNNModel()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Pydantic models for request/response
class ModelCreateRequest(BaseModel):
    model_type: str = "simple"  # simple, vgg, resnet

class TrainingRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 32

class ModelResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Health check endpoint
@router.get("/health", summary="Health Check")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "message": "CNN Image Classifier API is running"}

# Model management endpoints
@router.post("/model/create", response_model=ModelResponse, summary="Create CNN Model")
async def create_model(request: ModelCreateRequest):
    """
    Create a new CNN model with specified architecture
    
    - **model_type**: Type of CNN architecture (simple, vgg, resnet)
    """
    try:
        logger.info(f"Creating {request.model_type} model...")
        
        # Run model creation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            cnn_model.create_model, 
            request.model_type
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message=result["message"],
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/model/info", response_model=ModelResponse, summary="Get Model Information")
async def get_model_info():
    """Get information about the current model"""
    try:
        result = cnn_model.get_model_info()
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message="Model information retrieved successfully",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/model/load", response_model=ModelResponse, summary="Load Saved Model")
async def load_model(model_path: Optional[str] = None):
    """Load a previously saved model"""
    try:
        result = cnn_model.load_model(model_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message=result["message"]
        )
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Training endpoints
@router.post("/model/train", response_model=ModelResponse, summary="Train CNN Model")
async def train_model(request: TrainingRequest):
    """
    Train the CNN model with synthetic data
    
    - **epochs**: Number of training epochs (default: 10)
    - **batch_size**: Batch size for training (default: 32)
    """
    try:
        logger.info(f"Starting training for {request.epochs} epochs...")
        
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            cnn_model.train_model,
            request.epochs,
            request.batch_size
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message=result["message"],
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Prediction endpoints
@router.post("/predict/image", response_model=ModelResponse, summary="Predict Image Class")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image is a cat or dog
    
    - **file**: Image file to classify (JPG, PNG, etc.)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            cnn_model.predict_image,
            image_data
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message="Image prediction completed",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/predict/random", response_model=ModelResponse, summary="Generate Random Prediction")
async def predict_random():
    """
    Generate a random test image and predict its class
    """
    try:
        # Run random prediction in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            cnn_model.generate_random_prediction
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message="Random prediction generated",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error generating random prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Utility endpoints
@router.get("/models/architectures", summary="Get Available Architectures")
async def get_architectures():
    """Get list of available CNN architectures"""
    return {
        "architectures": [
            {
                "name": "simple",
                "description": "Simple LeNet-inspired CNN with 2 convolutional layers",
                "features": ["Basic convolution", "Max pooling", "Dropout regularization"]
            },
            {
                "name": "vgg",
                "description": "VGG-style CNN with deeper architecture and smaller filters",
                "features": ["Multiple 3x3 convolutions", "Deeper network", "Better feature learning"]
            },
            {
                "name": "resnet",
                "description": "ResNet-inspired CNN with skip connections",
                "features": ["Skip connections", "Batch normalization", "Gradient flow optimization"]
            }
        ]
    }

@router.get("/concepts/cnn", summary="CNN Learning Concepts")
async def get_cnn_concepts():
    """Get educational information about CNN concepts"""
    return {
        "concepts": {
            "convolution": {
                "description": "Feature detection using learnable filters/kernels",
                "key_points": [
                    "Filters slide across input to detect patterns",
                    "Each filter learns different features (edges, textures, etc.)",
                    "Convolution preserves spatial relationships"
                ]
            },
            "pooling": {
                "description": "Downsampling operation to reduce spatial dimensions",
                "types": {
                    "max_pooling": "Takes maximum value in each window",
                    "average_pooling": "Takes average value in each window"
                },
                "benefits": ["Reduces computation", "Provides translation invariance"]
            },
            "padding": {
                "description": "Adding pixels around input to control output size",
                "types": {
                    "valid": "No padding, output size shrinks",
                    "same": "Padding added to keep same output size"
                }
            },
            "stride": {
                "description": "Step size for moving filters across input",
                "effects": [
                    "Stride 1: Move one pixel at a time (detailed features)",
                    "Stride 2: Move two pixels at a time (faster, less detail)"
                ]
            }
        }
    }

@router.delete("/model/reset", summary="Reset Model")
async def reset_model():
    """Reset the current model (clear from memory)"""
    try:
        global cnn_model
        cnn_model = CNNModel()  # Create new instance
        
        return {
            "status": "success",
            "message": "Model reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Error handlers
@router.get("/test/error", include_in_schema=False)
async def test_error():
    """Test endpoint to trigger error handling"""
    raise HTTPException(status_code=500, detail="This is a test error")