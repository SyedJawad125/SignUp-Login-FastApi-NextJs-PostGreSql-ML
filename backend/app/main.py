from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Enable SQLAlchemy logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Import Base and engine from database (use absolute import)
from app.database import engine, Base

# Import all models explicitly
from app.models.user import User
from app.models.role import Role
from app.models.permission import Permission
from app.models.image_category import ImageCategory
from app.models.image import Image
from app.models.mnist_model import initialize_model

# Import routers
from app.routers import (
    employee, auth, house_price_api, user, 
    role, permission, image_category, image, 
    churn_router, pca_api, car_price_api, mnist_api # Added house_price_model router
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================================================
# Application Lifespan Management
# ========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    logger.info("🚀 Starting MNIST API application...")

    try:
        # Initialize MNIST model
        logger.info("📦 Initializing MNIST model...")
        model = initialize_model()
        logger.info("✅ Model initialized successfully!")

        if model.is_loaded:
            model_info = model.get_model_info()
            logger.info(f"📊 Model accuracy: {model_info['performance']['test_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {str(e)}")
        logger.info("🔄 Application will continue, but model endpoints may not work")

    logger.info("🎉 Application startup completed!")
    yield
    logger.info("🛑 Shutting down MNIST API application...")
    logger.info("✅ Shutdown completed!")

# ========================================================================
# Create FastAPI App
# ========================================================================

app = FastAPI(
    title="MNIST Digit Classification API",
    version="1.0.0",
    description="Deep Learning API for Handwritten Digit Recognition using MNIST dataset.",
    lifespan=lifespan
)
app = FastAPI(
    # title="HRM System with House Price Prediction",
    # version="1.0.0",
    # description="An API for managing HRM features with machine learning capabilities",
    title="HRM System with House Price Prediction + MNIST Digit Classification",
    version="1.0.0",
    description=(
        "An API for managing HRM features with machine learning capabilities, "
        "including House Price Prediction, Churn Prediction, Car Price Prediction, "
        "and MNIST Handwritten Digit Recognition."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Employees",
            "description": "Employee profile management"
        },
        {
            "name": "Users",
            "description": "User login and registration"
        },
        {
            "name": "Roles",
            "description": "Roles profile management"
        },
        {
            "name": "Image Categories",
            "description": "Image categories management"
        },
        {
            "name": "Machine Learning",
            "description": "House price prediction model"
        }
    ]
)

# CORS settings to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os
# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/api/hello")
def read_root():
    return {"message": "Hello from FastAPI!"}

# Include all routers
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(employee.router)
app.include_router(role.router)
app.include_router(permission.router)
app.include_router(image_category.router)
app.include_router(image.router)
app.include_router(house_price_api.router)  # Added ML router
app.include_router(churn_router.router)  # Added ML router
app.include_router(pca_api.router)
app.include_router(car_price_api.router)
app.include_router(mnist_api.router)


@app.on_event("startup")
def startup_event():
    print("Creating database tables...")
    print(f"Engine URL: {engine.url}")
    print(f"Tables in metadata: {Base.metadata.tables.keys()}")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

    # Load ML model on startup
    try:
        from app.models.house_price_model import train_and_save_model
        print("Initializing ML model...")
        train_and_save_model()
        print("ML model ready")
    except Exception as e:
        print(f"Error loading ML model: {str(e)}")

    try:
        from app.models.churn_model import train_churn_model
        print("Initializing churn prediction model...")
        train_churn_model()
    except Exception as e:
        print(f"Error loading churn model: {str(e)}")

    # Import training function
    try:
        from app.models.car_price_model import train_car_price_model
        print("Initializing car price prediction model...")
        train_car_price_model()
    except Exception as e:
        print(f"❌ Error loading car price model: {str(e)}")

@app.get("/")
def root():
    return {"message": "Welcome to HRM API with Machine Learning capabilities"}

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API"}

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.get("/test")
def test():
    return {"status": "working"}