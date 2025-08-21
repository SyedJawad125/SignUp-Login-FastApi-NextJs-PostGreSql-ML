# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# # Enable SQLAlchemy logging
# import logging
# logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# # Import Base and engine from database (use absolute import)
# from app.database import engine, Base

# # Import all models explicitly
# from app.models.user import User
# from app.models.role import Role
# from app.models.permission import Permission
# from app.models.image_category import ImageCategory
# from app.models.image import Image
# from app.models.mnist_model import initialize_model

# # Import routers
# from app.routers import (
#     employee, auth, house_price_api, user, 
#     role, permission, image_category, image, 
#     churn_router, pca_api, car_price_api, mnist_api # Added house_price_model router
# )


# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # ========================================================================
# # Application Lifespan Management
# # ========================================================================

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Handle startup and shutdown events"""
#     logger.info("🚀 Starting MNIST API application...")

#     try:
#         # Initialize MNIST model
#         logger.info("📦 Initializing MNIST model...")
#         model = initialize_model()
#         logger.info("✅ Model initialized successfully!")

#         if model.is_loaded:
#             model_info = model.get_model_info()
#             logger.info(f"📊 Model accuracy: {model_info['performance']['test_accuracy']:.4f}")
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize model: {str(e)}")
#         logger.info("🔄 Application will continue, but model endpoints may not work")

#     logger.info("🎉 Application startup completed!")
#     yield
#     logger.info("🛑 Shutting down MNIST API application...")
#     logger.info("✅ Shutdown completed!")

# # ========================================================================
# # Create FastAPI App
# # ========================================================================

# app = FastAPI(
#     title="MNIST Digit Classification API",
#     version="1.0.0",
#     description="Deep Learning API for Handwritten Digit Recognition using MNIST dataset.",
#     lifespan=lifespan
# )
# app = FastAPI(
#     # title="HRM System with House Price Prediction",
#     # version="1.0.0",
#     # description="An API for managing HRM features with machine learning capabilities",
#     title="HRM System with House Price Prediction + MNIST Digit Classification",
#     version="1.0.0",
#     description=(
#         "An API for managing HRM features with machine learning capabilities, "
#         "including House Price Prediction, Churn Prediction, Car Price Prediction, "
#         "and MNIST Handwritten Digit Recognition."
#     ),
#     lifespan=lifespan,
#     openapi_tags=[
#         {
#             "name": "Employees",
#             "description": "Employee profile management"
#         },
#         {
#             "name": "Users",
#             "description": "User login and registration"
#         },
#         {
#             "name": "Roles",
#             "description": "Roles profile management"
#         },
#         {
#             "name": "Image Categories",
#             "description": "Image categories management"
#         },
#         {
#             "name": "Machine Learning",
#             "description": "House price prediction model"
#         }
#     ]
# )

# # CORS settings to allow requests from frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Next.js frontend origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# from fastapi.staticfiles import StaticFiles
# import os
# # Mount static files
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# @app.get("/api/hello")
# def read_root():
#     return {"message": "Hello from FastAPI!"}

# # Include all routers
# app.include_router(auth.router)
# app.include_router(user.router)
# app.include_router(employee.router)
# app.include_router(role.router)
# app.include_router(permission.router)
# app.include_router(image_category.router)
# app.include_router(image.router)
# app.include_router(house_price_api.router)  # Added ML router
# app.include_router(churn_router.router)  # Added ML router
# app.include_router(pca_api.router)
# app.include_router(car_price_api.router)
# app.include_router(mnist_api.router)


# @app.on_event("startup")
# def startup_event():
#     print("Creating database tables...")
#     print(f"Engine URL: {engine.url}")
#     print(f"Tables in metadata: {Base.metadata.tables.keys()}")
#     Base.metadata.create_all(bind=engine)
#     print("Database tables created.")

#     # Load ML model on startup
#     try:
#         from app.models.house_price_model import train_and_save_model
#         print("Initializing ML model...")
#         train_and_save_model()
#         print("ML model ready")
#     except Exception as e:
#         print(f"Error loading ML model: {str(e)}")

#     try:
#         from app.models.churn_model import train_churn_model
#         print("Initializing churn prediction model...")
#         train_churn_model()
#     except Exception as e:
#         print(f"Error loading churn model: {str(e)}")

#     # Import training function
#     try:
#         from app.models.car_price_model import train_car_price_model
#         print("Initializing car price prediction model...")
#         train_car_price_model()
#     except Exception as e:
#         print(f"❌ Error loading car price model: {str(e)}")

# @app.get("/")
# def root():
#     return {"message": "Welcome to HRM API with Machine Learning capabilities"}

# @app.get("/")
# def read_root():
#     return {"message": "Customer Churn Prediction API"}

# @app.get("/ping")
# async def health_check():
#     return {"status": "healthy"}

# @app.get("/test")
# def test():
#     return {"status": "working"}




# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import logging
# import os

# # Enable SQLAlchemy logging
# logging.basicConfig()
# logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

# # Import Base and engine
# from app.database import engine, Base

# # Import models
# from app.models.user import User
# from app.models.role import Role
# from app.models.permission import Permission
# from app.models.image_category import ImageCategory
# from app.models.image import Image
# from app.models.cnn_model import CNNModel   # ✅ CNN model

# # Import routers
# from app.routers import (
#     employee, auth, house_price_api, user,
#     role, permission, image_category, image,
#     churn_router, pca_api, car_price_api, cnn_model_api
# )

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ========================================================================
# # Global CNN Model Instance
# # ========================================================================
# cnn_model = CNNModel()
# cnn_model.is_loaded = False   # 👈 ensure attribute exists globally


# # ========================================================================
# # Application Lifespan (startup/shutdown for MNIST CNN model)
# # ========================================================================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("🚀 Starting API application...")

#     # Initialize CNN MNIST model
#     try:
#         logger.info("📦 Loading MNIST CNN model...")
#         model_path = "app/models/mnist_cnn.pth"

#         if os.path.exists(model_path):
#             cnn_model.load_model(model_path)
#             cnn_model.is_loaded = True
#             logger.info("✅ CNN model loaded successfully.")

#             # Optional: print model info if available
#             if hasattr(cnn_model, "get_model_info"):
#                 model_info = cnn_model.get_model_info()
#                 logger.info(
#                     f"📊 Test accuracy: {model_info['performance'].get('test_accuracy', 'N/A')}"
#                 )
#         else:
#             logger.warning("⚠️ No pretrained CNN weights found. Train first before prediction.")
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize CNN model: {str(e)}")
#         logger.info("⚠️ App will continue, but MNIST endpoints may not work")

#     logger.info("🎉 Startup completed!")
#     yield
#     logger.info("🛑 Shutting down API...")
#     logger.info("✅ Shutdown completed!")


# # ========================================================================
# # Create FastAPI App
# # ========================================================================
# app = FastAPI(
#     title="HRM System + ML Models (House Price, Churn, Car Price, MNIST CNN)",
#     version="1.0.0",
#     description=(
#         "An API for HRM features with integrated ML models: "
#         "House Price Prediction, Churn Prediction, Car Price Prediction, "
#         "and MNIST CNN Handwritten Digit Recognition."
#     ),
#     lifespan=lifespan,
#     openapi_tags=[
#         {"name": "Employees", "description": "Employee profile management"},
#         {"name": "Users", "description": "User login and registration"},
#         {"name": "Roles", "description": "Role management"},
#         {"name": "Image Categories", "description": "Image categories management"},
#         {"name": "Machine Learning", "description": "ML models (House, Churn, Car, MNIST CNN)"}
#     ]
# )

# # ========================================================================
# # Middleware & Static
# # ========================================================================
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Next.js frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# # ========================================================================
# # Basic Routes
# # ========================================================================
# @app.get("/api/hello")
# def hello():
#     return {"message": "Hello from FastAPI!"}

# @app.get("/")
# def root():
#     return {"message": "Welcome to HRM API with Machine Learning capabilities"}

# @app.get("/ping")
# async def health_check():
#     return {"status": "healthy"}

# @app.get("/test")
# def test():
#     return {"status": "working"}

# # ========================================================================
# # Routers
# # ========================================================================
# app.include_router(auth.router)
# app.include_router(user.router)
# app.include_router(employee.router)
# app.include_router(role.router)
# app.include_router(permission.router)
# app.include_router(image_category.router)
# app.include_router(image.router)
# app.include_router(house_price_api.router)
# app.include_router(churn_router.router)
# app.include_router(pca_api.router)
# app.include_router(car_price_api.router)
# app.include_router(cnn_model_api.router)

# # ========================================================================
# # Startup Events (non-CNN ML models)
# # ========================================================================
# @app.on_event("startup")
# def startup_event():
#     print("Creating database tables...")
#     print(f"Engine URL: {engine.url}")
#     print(f"Tables in metadata: {Base.metadata.tables.keys()}")
#     Base.metadata.create_all(bind=engine)
#     print("✅ Database tables created.")

#     # Initialize other ML models
#     try:
#         from app.models.house_price_model import train_and_save_model
#         print("Initializing House Price model...")
#         train_and_save_model()
#         print("✅ House Price model ready")
#     except Exception as e:
#         print(f"❌ Error loading House Price model: {str(e)}")

#     try:
#         from app.models.churn_model import train_churn_model
#         print("Initializing Churn model...")
#         train_churn_model()
#         print("✅ Churn model ready")
#     except Exception as e:
#         print(f"❌ Error loading Churn model: {str(e)}")

#     try:
#         from app.models.car_price_model import train_car_price_model
#         print("Initializing Car Price model...")
#         train_car_price_model()
#         print("✅ Car Price model ready")
#     except Exception as e:
#         print(f"❌ Error loading Car Price model: {str(e)}")






# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import logging
# import os

# # Enable SQLAlchemy logging
# logging.basicConfig()
# logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

# # Import Base and engine
# from app.database import engine, Base

# # Import models
# from app.models.user import User
# from app.models.role import Role
# from app.models.permission import Permission
# from app.models.image_category import ImageCategory
# from app.models.image import Image
# from app.models.cnn_model import CNNModel   # ✅ CNN model

# # Import routers
# from app.routers import (
#     employee, auth, house_price_api, user,
#     role, permission, image_category, image,
#     churn_router, pca_api, car_price_api,mnist_api, cnn_model_api
# )

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ========================================================================
# # Global CNN Model Instance
# # ========================================================================
# cnn_model = None

# # ========================================================================
# # Application Lifespan (startup/shutdown for CNN model)
# # ========================================================================
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global cnn_model
#     logger.info("🚀 Starting API application...")

#     # Initialize CNN model (Cats vs Dogs)
#     try:
#         logger.info("📦 Initializing CNN model...")
        
#         # Create the CNN model instance
#         cnn_model = CNNModel()
        
#         # Inject the model instance into the router
#         cnn_model_api.set_cnn_model(cnn_model)
        
#         # Create the model first
#         create_result = cnn_model.create_model("simple")
#         if create_result["status"] == "error":
#             logger.warning(f"⚠️ Failed to create CNN model: {create_result['message']}")
#         else:
#             logger.info(f"✅ CNN model created: {create_result['message']}")
            
#             # Try to load pre-trained weights if available
#             model_path = "app/ml_models/cnn_cats_dogs.h5"
            
#             # Ensure the directory exists
#             os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
#             if os.path.exists(model_path):
#                 load_result = cnn_model.load_model(model_path)
#                 if load_result["status"] == "success":
#                     logger.info("✅ Pre-trained CNN model loaded successfully")
#                 else:
#                     logger.warning(f"⚠️ Could not load pre-trained model: {load_result['message']}")
#             else:
#                 logger.info("ℹ️ No pre-trained model found. Train the model first using the API.")
                
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize CNN model: {str(e)}")
#         logger.info("⚠️ App will continue, but CNN endpoints may not work")

#     logger.info("🎉 Startup completed!")
#     yield
#     logger.info("🛑 Shutting down API...")

# # ========================================================================
# # Create FastAPI App
# # ========================================================================
# app = FastAPI(
#     title="HRM System + ML Models (House Price, Churn, Car Price, CNN Image Classification)",
#     version="1.0.0",
#     description=(
#         "An API for HRM features with integrated ML models: "
#         "House Price Prediction, Churn Prediction, Car Price Prediction, "
#         "and CNN Image Classification (Cats vs Dogs)."
#     ),
#     lifespan=lifespan,
#     openapi_tags=[
#         {"name": "Employees", "description": "Employee profile management"},
#         {"name": "Users", "description": "User login and registration"},
#         {"name": "Roles", "description": "Role management"},
#         {"name": "Image Categories", "description": "Image categories management"},
#         {"name": "Machine Learning", "description": "ML models (House, Churn, Car, CNN Image Classification)"},
#         {"name": "CNN Image Classifier", "description": "CNN model for cat/dog image classification"}
#     ]
# )

# # ========================================================================
# # Middleware & Static
# # ========================================================================
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Next.js frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# # ========================================================================
# # Basic Routes
# # ========================================================================
# @app.get("/api/hello")
# def hello():
#     return {"message": "Hello from FastAPI!"}

# @app.get("/")
# def root():
#     return {"message": "Welcome to HRM API with Machine Learning capabilities"}

# @app.get("/ping")
# async def health_check():
#     cnn_status = "loaded" if cnn_model and cnn_model.is_loaded else "not_loaded"
#     return {
#         "status": "healthy",
#         "cnn_model_status": cnn_status
#     }

# @app.get("/test")
# def test():
#     return {"status": "working"}

# # ========================================================================
# # Routers
# # ========================================================================
# app.include_router(auth.router)
# app.include_router(user.router)
# app.include_router(employee.router)
# app.include_router(role.router)
# app.include_router(permission.router)
# app.include_router(image_category.router)
# app.include_router(image.router)
# app.include_router(house_price_api.router)
# app.include_router(churn_router.router)
# app.include_router(pca_api.router)
# app.include_router(car_price_api.router)
# app.include_router(mnist_api.router)
# app.include_router(cnn_model_api.router)

# # ========================================================================
# # Startup Events (non-CNN ML models)
# # ========================================================================
# @app.on_event("startup")
# def startup_event():
#     print("Creating database tables...")
#     print(f"Engine URL: {engine.url}")
#     print(f"Tables in metadata: {Base.metadata.tables.keys()}")
#     Base.metadata.create_all(bind=engine)
#     print("✅ Database tables created.")

#     # Initialize other ML models
#     try:
#         from app.models.house_price_model import train_and_save_model
#         print("Initializing House Price model...")
#         train_and_save_model()
#         print("✅ House Price model ready")
#     except Exception as e:
#         print(f"❌ Error loading House Price model: {str(e)}")

#     try:
#         from app.models.churn_model import train_churn_model
#         print("Initializing Churn model...")
#         train_churn_model()
#         print("✅ Churn model ready")
#     except Exception as e:
#         print(f"❌ Error loading Churn model: {str(e)}")

#     try:
#         from app.models.car_price_model import train_car_price_model
#         print("Initializing Car Price model...")
#         train_car_price_model()
#         print("✅ Car Price model ready")
#     except Exception as e:
#         print(f"❌ Error loading Car Price model: {str(e)}")

#     # Add this in the startup_event() function
#     try:
#         from app.models.mnist_model import initialize_model
#         print("Initializing MNIST model...")
#         mnist_model = initialize_model()
#         print("✅ MNIST model ready")
#     except Exception as e:
#         print(f"❌ Error loading MNIST model: {str(e)}")

# # ========================================================================
# # Additional CNN Model Utilities
# # ========================================================================
# @app.get("/api/cnn/status", tags=["CNN Image Classifier"])
# async def get_cnn_status():
#     """Get the current status of the CNN model"""
#     if cnn_model is None:
#         return {
#             "status": "not_initialized",
#             "message": "CNN model has not been initialized"
#         }
    
#     return {
#         "status": "initialized" if cnn_model.is_loaded else "created_but_not_loaded",
#         "model_type": cnn_model.model_type,
#         "is_loaded": cnn_model.is_loaded,
#         "message": "CNN model is ready for use" if cnn_model.is_loaded else "CNN model needs training or loading"
#     }



# main.py - Fixed version
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os

# Enable SQLAlchemy logging
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

# Import Base and engine
from app.database import engine, Base

# Import models
from app.models.user import User
from app.models.role import Role
from app.models.permission import Permission
from app.models.image_category import ImageCategory
from app.models.image import Image

# Import routers - FIXED: Import the actual CNN model class and router
from app.routers import (
    employee, auth, house_price_api, user,
    role, permission, image_category, image,
    churn_router, pca_api, car_price_api, mnist_api
)

# Import the CNN model and router correctly
from app.models.cnn_model import CNNModel
from app.routers import cnn_model_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========================================================================
# Global CNN Model Instance
# ========================================================================
cnn_model = None

# ========================================================================
# Application Lifespan (startup/shutdown for CNN model)
# ========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cnn_model
    logger.info("🚀 Starting API application...")

    # Initialize CNN model (Cats vs Dogs)
    try:
        logger.info("📦 Initializing CNN model...")
        
        # Create the CNN model instance
        cnn_model = CNNModel()
        
        # Inject the model instance into the router
        cnn_model_api.set_cnn_model(cnn_model)
        
        # Try to load pre-trained weights if available
        model_path = "app/ml_models/cnn_cats_dogs.h5"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            load_result = cnn_model.load_model(model_path)
            if load_result["status"] == "success":
                logger.info("✅ Pre-trained CNN model loaded successfully")
            else:
                logger.warning(f"⚠️ Could not load pre-trained model: {load_result['message']}")
                logger.info("ℹ️ No pre-trained model found. Train the model first using the API.")
        else:
            logger.info("ℹ️ No pre-trained model found. Train the model first using the API.")
                
    except Exception as e:
        logger.error(f"❌ Failed to initialize CNN model: {str(e)}")
        logger.info("⚠️ App will continue, but CNN endpoints may not work")

    logger.info("🎉 Startup completed!")
    yield
    logger.info("🛑 Shutting down API...")

# ========================================================================
# Create FastAPI App
# ========================================================================
app = FastAPI(
    title="HRM System + ML Models (House Price, Churn, Car Price, CNN Image Classification)",
    version="1.0.0",
    description=(
        "An API for HRM features with integrated ML models: "
        "House Price Prediction, Churn Prediction, Car Price Prediction, "
        "and CNN Image Classification (Cats vs Dogs)."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Employees", "description": "Employee profile management"},
        {"name": "Users", "description": "User login and registration"},
        {"name": "Roles", "description": "Role management"},
        {"name": "Image Categories", "description": "Image categories management"},
        {"name": "Machine Learning", "description": "ML models (House, Churn, Car, CNN Image Classification)"},
        {"name": "CNN Image Classifier", "description": "CNN model for cat/dog image classification"}
    ]
)

# ========================================================================
# Middleware & Static
# ========================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ========================================================================
# Basic Routes
# ========================================================================
@app.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI!"}

@app.get("/")
def root():
    return {"message": "Welcome to HRM API with Machine Learning capabilities"}

@app.get("/ping")
async def health_check():
    cnn_status = "loaded" if cnn_model and cnn_model.is_loaded else "not_loaded"
    return {
        "status": "healthy",
        "cnn_model_status": cnn_status
    }

@app.get("/test")
def test():
    return {"status": "working"}

# ========================================================================
# Routers - FIXED: Include all routers with proper prefixes
# ========================================================================
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(employee.router)
app.include_router(role.router)
app.include_router(permission.router)
app.include_router(image_category.router)
app.include_router(image.router)
app.include_router(house_price_api.router)
app.include_router(churn_router.router)
app.include_router(pca_api.router)
app.include_router(car_price_api.router)
app.include_router(mnist_api.router)
app.include_router(cnn_model_api.router)  # This should now work

# ========================================================================
# Startup Events (non-CNN ML models)
# ========================================================================
@app.on_event("startup")
def startup_event():
    print("Creating database tables...")
    print(f"Engine URL: {engine.url}")
    print(f"Tables in metadata: {Base.metadata.tables.keys()}")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created.")

    # Initialize other ML models
    try:
        from app.models.house_price_model import train_and_save_model
        print("Initializing House Price model...")
        train_and_save_model()
        print("✅ House Price model ready")
    except Exception as e:
        print(f"❌ Error loading House Price model: {str(e)}")

    try:
        from app.models.churn_model import train_churn_model
        print("Initializing Churn model...")
        train_churn_model()
        print("✅ Churn model ready")
    except Exception as e:
        print(f"❌ Error loading Churn model: {str(e)}")

    try:
        from app.models.car_price_model import train_car_price_model
        print("Initializing Car Price model...")
        train_car_price_model()
        print("✅ Car Price model ready")
    except Exception as e:
        print(f"❌ Error loading Car Price model: {str(e)}")

    # Add this in the startup_event() function
    try:
        from app.models.mnist_model import initialize_model
        print("Initializing MNIST model...")
        mnist_model = initialize_model()
        print("✅ MNIST model ready")
    except Exception as e:
        print(f"❌ Error loading MNIST model: {str(e)}")

# ========================================================================
# Additional CNN Model Utilities
# ========================================================================
@app.get("/api/cnn/status", tags=["CNN Image Classifier"])
async def get_cnn_status():
    """Get the current status of the CNN model"""
    if cnn_model is None:
        return {
            "status": "not_initialized",
            "message": "CNN model has not been initialized"
        }
    
    return {
        "status": "initialized" if cnn_model.is_loaded else "created_but_not_loaded",
        "model_type": cnn_model.model_type,
        "is_loaded": cnn_model.is_loaded,
        "is_trained": cnn_model.is_trained,
        "message": "CNN model is ready for use" if cnn_model.is_loaded else "CNN model needs training or loading"
    }