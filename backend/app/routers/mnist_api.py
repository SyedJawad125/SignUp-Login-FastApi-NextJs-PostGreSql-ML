"""
MNIST API Endpoints
==================
FastAPI router with all prediction endpoints
Save as: app/routers/endpoints.py
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging

# Import our model
from app.models.mnist_model import mnist_model

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/mnist", tags=["MNIST Prediction"])

# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ============================================================================

class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: List[float]
    timestamp: str

class ImageArray(BaseModel):
    image_data: List[List[float]]  # 28x28 array

class Base64Image(BaseModel):
    image: str  # base64 encoded image

class TrainingRequest(BaseModel):
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 128
    validation_split: Optional[float] = 0.1

class TrainingStatus(BaseModel):
    status: str
    message: str
    accuracy: Optional[float] = None

class TestSampleResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: List[float]
    timestamp: str
    test_info: dict

class BatchTestResponse(BaseModel):
    batch_size: int
    accuracy: float
    correct_predictions: int
    total_predictions: int
    results: List[dict]
    timestamp: str

# ============================================================================
# HEALTH AND INFO ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": mnist_model.is_loaded,
        "timestamp": datetime.now().isoformat(),
        "service": "MNIST Digit Classification API"
    }

@router.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        info = mnist_model.get_model_info()
        info["timestamp"] = datetime.now().isoformat()
        return info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@router.post("/predict/array", response_model=PredictionResponse)
async def predict_from_array(image_request: ImageArray):
    """
    Predict digit from 28x28 numerical array
    
    Expected format:
    {
        "image_data": [
            [0.0, 0.1, 0.2, ...], // 28 values
            [0.0, 0.1, 0.2, ...], // 28 values
            // ... 28 rows total
        ]
    }
    """
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        # Preprocess image
        img = mnist_model.preprocess_image_array(image_request.image_data)
        
        # Make prediction
        result = mnist_model.predict_single(img)
        
        return PredictionResponse(
            **result,
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(image_request: Base64Image):
    """
    Predict digit from base64 encoded image
    
    Expected format:
    {
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
    """
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        # Preprocess image
        img = mnist_model.preprocess_base64_image(image_request.image)
        
        # Make prediction
        result = mnist_model.predict_single(img)
        
        return PredictionResponse(
            **result,
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/file", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image file
    Supports: PNG, JPG, JPEG, BMP, GIF
    """
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        # Check file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file data
        file_data = await file.read()
        
        # Preprocess image
        img = mnist_model.preprocess_uploaded_file(file_data)
        
        # Make prediction
        result = mnist_model.predict_single(img)
        
        return PredictionResponse(
            **result,
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================================================
# TESTING ENDPOINTS
# ============================================================================

@router.get("/test/random", response_model=TestSampleResponse)
async def test_random_sample():
    """Test prediction on a random sample from test dataset"""
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        result = mnist_model.get_test_sample()
        result["timestamp"] = datetime.now().isoformat()
        
        return TestSampleResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")

@router.get("/test/sample/{index}", response_model=TestSampleResponse)
async def test_specific_sample(index: int):
    """Test prediction on a specific test sample by index"""
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        result = mnist_model.get_test_sample(index)
        result["timestamp"] = datetime.now().isoformat()
        
        return TestSampleResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")

@router.get("/test/batch/{count}", response_model=BatchTestResponse)
async def test_batch_samples(count: int):
    """
    Test model on multiple random samples
    Maximum batch size: 1000
    """
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        if count > 1000:
            raise HTTPException(status_code=400, detail="Maximum batch size is 1000")
        
        if count < 1:
            raise HTTPException(status_code=400, detail="Batch size must be at least 1")
        
        result = mnist_model.evaluate_batch(count)
        result["timestamp"] = datetime.now().isoformat()
        
        return BatchTestResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch test error: {str(e)}")

# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/model/train", response_model=TrainingStatus)
async def retrain_model(background_tasks: BackgroundTasks, training_request: TrainingRequest):
    """
    Retrain the model with specified parameters
    Training runs in background, check logs for progress
    """
    def train_in_background(epochs: int, batch_size: int, validation_split: float):
        try:
            logger.info(f"Starting background training with {epochs} epochs...")
            accuracy = mnist_model.train_model(epochs, batch_size, validation_split)
            logger.info(f"Background training completed! Final accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.error(f"Background training failed: {str(e)}")
    
    # Validate parameters
    if training_request.epochs < 1 or training_request.epochs > 100:
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 100")
    
    if training_request.batch_size < 1 or training_request.batch_size > 1024:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 1024")
    
    if training_request.validation_split < 0.0 or training_request.validation_split > 0.5:
        raise HTTPException(status_code=400, detail="Validation split must be between 0.0 and 0.5")
    
    # Start training in background
    background_tasks.add_task(
        train_in_background,
        training_request.epochs,
        training_request.batch_size,
        training_request.validation_split
    )
    
    return TrainingStatus(
        status="started",
        message=f"Training started with {training_request.epochs} epochs. Monitor logs for progress."
    )

@router.post("/model/reload")
async def reload_model():
    """Reload the model from saved file"""
    try:
        success = mnist_model.load_model()
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="No saved model found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

# ============================================================================
# BATCH PREDICTION ENDPOINTS
# ============================================================================

@router.post("/predict/batch/arrays")
async def predict_batch_arrays(images: List[ImageArray]):
    """
    Predict multiple images from arrays at once
    Maximum batch size: 100
    """
    try:
        if not mnist_model.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        if len(images) > 100:
            raise HTTPException(status_code=400, detail="Maximum batch size is 100")
        
        # Preprocess all images
        processed_images = []
        for img_data in images:
            img = mnist_model.preprocess_image_array(img_data.image_data)
            processed_images.append(img)
        
        # Convert to numpy array
        import numpy as np
        batch_images = np.array(processed_images)
        
        # Make batch prediction
        results = mnist_model.predict_batch(batch_images)
        
        # Add timestamp to each result
        timestamp = datetime.now().isoformat()
        for result in results:
            result["timestamp"] = timestamp
        
        return {
            "batch_size": len(images),
            "results": results,
            "timestamp": timestamp
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/")
async def api_root():
    """API root with available endpoints information"""
    return {
        "service": "MNIST Digit Classification API",
        "version": "1.0.0",
        "model_loaded": mnist_model.is_loaded,
        "endpoints": {
            "prediction": {
                "POST /predict/array": "Predict from 28x28 numerical array",
                "POST /predict/base64": "Predict from base64 encoded image",
                "POST /predict/file": "Predict from uploaded image file",
                "POST /predict/batch/arrays": "Predict multiple arrays at once"
            },
            "testing": {
                "GET /test/random": "Test with random sample from dataset",
                "GET /test/sample/{index}": "Test with specific test sample",
                "GET /test/batch/{count}": "Test multiple samples at once"
            },
            "model_management": {
                "GET /model/info": "Get model information and performance",
                "POST /model/train": "Retrain the model",
                "POST /model/reload": "Reload model from file"
            },
            "utility": {
                "GET /health": "Health check",
                "GET /": "This endpoint"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/stats")
async def get_api_stats():
    """Get API statistics and model performance"""
    try:
        if not mnist_model.is_loaded:
            return {
                "model_loaded": False,
                "message": "Model not loaded",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get model info
        model_info = mnist_model.get_model_info()
        
        # Test on small batch for current performance
        test_result = mnist_model.evaluate_batch(100)
        
        return {
            "model_loaded": True,
            "model_performance": {
                "test_accuracy": model_info["performance"]["test_accuracy"],
                "test_loss": model_info["performance"]["test_loss"]
            },
            "current_performance": {
                "sample_accuracy": test_result["accuracy"],
                "sample_size": test_result["batch_size"]
            },
            "model_details": {
                "total_parameters": model_info["model_summary"]["total_params"],
                "layers": model_info["model_summary"]["layers"]
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")