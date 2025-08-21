from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os

import app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/cnn", tags=["CNN Image Classifier"])

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Pydantic models for request/response
class ModelCreateRequest(BaseModel):
    model_type: str = "simple"  # simple, vgg, resnet
    target_size: Tuple[int, int] = (150, 150)  # Image size for training
    batch_size: int = 32

class TrainingRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 32

class ModelResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Global variable to store the model instance - will be injected from main.py
cnn_model = None

def set_cnn_model(model_instance):
    """Set the CNN model instance from main.py"""
    global cnn_model
    cnn_model = model_instance
    logger.info("CNN model instance set in router")

def get_cnn_model():
    """Get the CNN model instance"""
    if cnn_model is None:
        raise HTTPException(status_code=500, detail="CNN model not initialized")
    return cnn_model

# Health check endpoint
@router.get("/health", summary="Health Check")
async def health_check():
    """Check if the API is running"""
    model_status = "not_initialized"
    dataset_status = "unknown"
    
    if cnn_model:
        model_status = "loaded" if cnn_model.is_loaded else "not_loaded"
        trained_status = "trained" if cnn_model.is_trained else "not_trained"
        
        # Check dataset availability
        dataset_path = cnn_model.dataset_path
        train_path = os.path.join(dataset_path, 'train')
        dataset_status = "available" if os.path.exists(train_path) else "missing"
        
        return {
            "status": "healthy", 
            "message": "CNN Image Classifier API is running",
            "model_status": model_status,
            "training_status": trained_status,
            "dataset_status": dataset_status,
            "dataset_path": dataset_path
        }
    
    return {
        "status": "healthy", 
        "message": "CNN Image Classifier API is running",
        "model_status": model_status,
        "dataset_status": dataset_status
    }

# Model management endpoints
@router.post("/model/create", response_model=ModelResponse, summary="Create CNN Model")
async def create_model(request: ModelCreateRequest):
    """
    Create a new CNN model with specified architecture and setup data generators
    
    - **model_type**: Type of CNN architecture (simple, vgg, resnet)
    - **target_size**: Input image size as [width, height] (default: [150, 150])
    - **batch_size**: Batch size for data generators (default: 32)
    """
    try:
        model = get_cnn_model()
        logger.info(f"Creating {request.model_type} model with target size {request.target_size}...")
        
        # Validate target size
        if len(request.target_size) != 2 or any(s <= 0 for s in request.target_size):
            raise HTTPException(status_code=400, detail="target_size must be a tuple of two positive integers")
        
        # Run model creation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            model.create_model, 
            request.model_type,
            request.target_size,
            request.batch_size
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message=result["message"],
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/model/info", response_model=ModelResponse, summary="Get Model Information")
async def get_model_info():
    """Get information about the current model"""
    try:
        model = get_cnn_model()
        result = model.get_model_info()
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message="Model information retrieved successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/model/load", response_model=ModelResponse, summary="Load Saved Model")
async def load_model(model_path: Optional[str] = Query(None, description="Path to model file")):
    """Load a previously saved model"""
    try:
        model = get_cnn_model()
        
        # Run loading in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            model.load_model,
            model_path
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message=result["message"],
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Training endpoints
@router.post("/model/train", response_model=ModelResponse, summary="Train CNN Model")
async def train_model(request: TrainingRequest):
    """
    Train the CNN model with real dataset from app/dataset folder
    
    - **epochs**: Number of training epochs (default: 10)
    - **batch_size**: Batch size for training (default: 32)
    """
    try:
        model = get_cnn_model()
        logger.info(f"Starting training for {request.epochs} epochs with batch size {request.batch_size}...")
        
        # Check if model is created
        if not model.is_loaded:
            raise HTTPException(status_code=400, detail="No model found. Please create a model first.")
        
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            model.train_model,
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/model/evaluate", response_model=ModelResponse, summary="Evaluate Model on Test Data")
async def evaluate_model():
    """
    Evaluate the trained model on test dataset
    """
    try:
        model = get_cnn_model()
        
        # Check if model is trained
        if not model.is_trained:
            raise HTTPException(status_code=400, detail="No trained model found. Please train a model first.")
        
        # Run evaluation in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            model.evaluate_model
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ModelResponse(
            status="success",
            message="Model evaluation completed",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Prediction endpoints - FIXED
@router.post("/predict/image", response_model=ModelResponse, summary="Predict Image Class")
async def predict_image(file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)")):
    """
    Predict whether an uploaded image is a cat or dog
    
    - **file**: Image file to classify (JPG, PNG, etc.)
    """
    try:
        model = get_cnn_model()
        
        # Check if model is available for prediction
        if not model.is_loaded:
            raise HTTPException(status_code=400, detail="No model found. Please create and train a model first.")
        
        # Check if file was actually uploaded
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Validate file type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        try:
            image_data = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Validate file size (e.g., max 10MB)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large. Maximum 10MB allowed.")
        
        logger.info(f"Processing image file: {file.filename}, size: {len(image_data)} bytes")
        
        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            model.predict_image,
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

@router.post("/predict/path", response_model=ModelResponse, summary="Predict Image from File Path")
async def predict_image_path(image_path: str = Form(..., description="Full path to the image file")):
    """
    Predict whether an image at given file path is a cat or dog
    
    - **image_path**: Full path to the image file
    """
    try:
        model = get_cnn_model()
        
        # Check if model is available for prediction
        if not model.is_loaded:
            raise HTTPException(status_code=400, detail="No model found. Please create and train a model first.")
        
        # Validate file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        
        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            model.predict_from_path,
            image_path
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
        logger.error(f"Error predicting image from path: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Dataset information endpoints
@router.get("/dataset/info", summary="Get Dataset Information")
async def get_dataset_info():
    """Get information about the dataset"""
    try:
        model = get_cnn_model()
        dataset_path = model.dataset_path
        
        info = {
            "dataset_path": dataset_path,
            "folders": {},
            "total_images": 0
        }
        
        # Check each folder
        for folder_name in ['train', 'validation', 'test']:
            folder_path = os.path.join(dataset_path, folder_name)
            if os.path.exists(folder_path):
                # Check for subfolders
                subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                folder_info = {"subfolders": {}, "total": 0}
                
                for subfolder in subfolders:
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.exists(subfolder_path):
                        try:
                            files = [f for f in os.listdir(subfolder_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
                            count = len(files)
                            folder_info["subfolders"][subfolder] = count
                            folder_info["total"] += count
                        except PermissionError:
                            folder_info["subfolders"][subfolder] = "permission_denied"
                
                info["folders"][folder_name] = folder_info
                info["total_images"] += folder_info["total"]
            else:
                info["folders"][folder_name] = {
                    "status": "not_found",
                    "path": folder_path
                }
        
        return {
            "status": "success",
            "message": "Dataset information retrieved",
            "data": info
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Utility endpoints
@router.get("/models/architectures", summary="Get Available Architectures")
async def get_architectures():
    """Get list of available CNN architectures"""
    return {
        "architectures": [
            {
                "name": "simple",
                "description": "Simple CNN with 3 convolutional layers - good for learning and fast training",
                "features": ["3 Conv layers", "Max pooling", "Dropout regularization", "Fast training"],
                "recommended_for": "Quick experiments and learning"
            },
            {
                "name": "vgg",
                "description": "VGG-style CNN with deeper architecture - better accuracy but slower",
                "features": ["Multiple 3x3 convolutions", "4 Conv blocks", "Better feature learning", "Higher accuracy"],
                "recommended_for": "Better accuracy on complex images"
            },
            {
                "name": "resnet",
                "description": "ResNet-inspired CNN with skip connections - best performance",
                "features": ["Skip connections", "Batch normalization", "Gradient flow optimization", "Best performance"],
                "recommended_for": "Production use and best results"
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
            "data_augmentation": {
                "description": "Techniques to increase dataset size and variety",
                "techniques": [
                    "Rotation: Rotate images by small angles",
                    "Shifting: Move images horizontally/vertically",
                    "Flipping: Mirror images horizontally",
                    "Zooming: Zoom in/out of images",
                    "Shearing: Apply shear transformation"
                ],
                "benefits": ["Prevents overfitting", "Improves generalization"]
            },
            "binary_classification": {
                "description": "Classification between two classes (Cat vs Dog)",
                "output": "Single neuron with sigmoid activation (0-1 probability)",
                "loss_function": "Binary crossentropy",
                "interpretation": "Output > 0.5 = Dog, Output < 0.5 = Cat"
            }
        }
    }

@router.delete("/model/reset", summary="Reset Model")
async def reset_model():
    """Reset the current model (clear from memory)"""
    try:
        model = get_cnn_model()
        
        # Reset model state
        model.model = None
        model.model_type = None
        model.is_loaded = False
        model.is_trained = False
        model.train_generator = None
        model.validation_generator = None
        model.test_generator = None
        
        return ModelResponse(
            status="success",
            message="Model reset successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Development and testing endpoints
@router.get("/test/dataset-paths", include_in_schema=False, summary="Test Dataset Paths")
async def test_dataset_paths():
    """Development endpoint to check if dataset paths exist"""
    try:
        model = get_cnn_model()
        base_path = model.dataset_path
        
        paths_to_check = [
            'train/cats',
            'train/dogs', 
            'validation/cats',
            'validation/dogs',
            'test/cats',
            'test/dogs'
        ]
        
        results = {}
        for path in paths_to_check:
            full_path = os.path.join(base_path, path)
            results[path] = {
                "exists": os.path.exists(full_path),
                "full_path": full_path
            }
            
            if os.path.exists(full_path):
                try:
                    files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
                    results[path]["image_count"] = len(files)
                    results[path]["sample_files"] = files[:3]  # Show first 3 files
                except PermissionError:
                    results[path]["error"] = "Permission denied"
                except Exception as e:
                    results[path]["error"] = str(e)
        
        return {"dataset_paths": results}
        
    except Exception as e:
        return {"error": str(e)}

# Simple test endpoint
@router.get("/test/simple", include_in_schema=False)
async def test_simple():
    """Simple test endpoint"""
    return {"message": "Router is working!", "timestamp": "2025-01-01"}

# Error handlers for testing
@router.get("/test/error", include_in_schema=False)
async def test_error():
    """Test endpoint to trigger error handling"""
    raise HTTPException(status_code=500, detail="This is a test error")


@router.get("/api/test")
async def test_endpoint():
    return {"message": "API is working", "endpoints": {
        "cnn_create": "/api/cnn/model/create",
        "cnn_train": "/api/cnn/model/train",
        "cnn_predict": "/api/cnn/predict/image"
    }}


@router.get("/model/debug")
async def debug_model():
    """Debug model loading and state"""
    model = get_cnn_model()
    
    # Check model file
    model_path = model.model_path
    file_exists = os.path.exists(model_path)
    file_size = os.path.getsize(model_path) if file_exists else 0
    
    # Check model state
    model_exists = model.model is not None
    can_predict = False
    
    if model_exists:
        try:
            # Test if model can make predictions
            test_input = np.random.random((1, *model.input_shape))
            prediction = model.model.predict(test_input, verbose=0)
            can_predict = True
        except Exception as e:
            can_predict = False
    
    return {
        "model_file": {
            "path": model_path,
            "exists": file_exists,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2) if file_exists else 0
        },
        "model_state": {
            "model_exists": model_exists,
            "is_loaded": model.is_loaded,
            "is_trained": model.is_trained,
            "can_predict": can_predict
        },
        "input_shape": model.input_shape
    }   