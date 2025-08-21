# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import os
# from typing import Tuple, Dict, Any
# import logging
# from PIL import Image
# import io
# import base64
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class CNNModel:
#     def __init__(self):
#         self.model = None
#         self.model_type = None
#         self.input_shape = (64, 64, 3)  # You can change this to (150, 150, 3) for better results
#         self.num_classes = 2
#         self.class_names = ['cats', 'dogs']  # Updated to match folder names
#         self.model_path = "app/ml_models/cnn_cats_dogs.h5"
#         self.dataset_path = "app/dataset"  # Path to your dataset
#         self.is_loaded = False
#         self.is_trained = False
        
#         # Data generators for real dataset
#         self.train_generator = None
#         self.validation_generator = None
#         self.test_generator = None
        
#     def setup_data_generators(self, batch_size: int = 32, target_size: Tuple[int, int] = (64, 64)):
#         """Setup data generators for training, validation, and testing"""
#         try:
#             # Data augmentation for training data
#             train_datagen = ImageDataGenerator(
#                 rescale=1./255,
#                 rotation_range=20,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 horizontal_flip=True,
#                 zoom_range=0.2,
#                 shear_range=0.2,
#                 fill_mode='nearest'
#             )
            
#             # Only rescaling for validation and test data
#             validation_datagen = ImageDataGenerator(rescale=1./255)
#             test_datagen = ImageDataGenerator(rescale=1./255)
            
#             # Paths to your dataset folders
#             train_path = os.path.join(self.dataset_path, 'train')
#             validation_path = os.path.join(self.dataset_path, 'validation')
#             test_path = os.path.join(self.dataset_path, 'test')
            
#             # Check if paths exist
#             if not os.path.exists(train_path):
#                 raise FileNotFoundError(f"Training data path not found: {train_path}")
#             if not os.path.exists(validation_path):
#                 raise FileNotFoundError(f"Validation data path not found: {validation_path}")
#             if not os.path.exists(test_path):
#                 raise FileNotFoundError(f"Test data path not found: {test_path}")
            
#             # Create generators
#             self.train_generator = train_datagen.flow_from_directory(
#                 train_path,
#                 target_size=target_size,
#                 batch_size=batch_size,
#                 class_mode='binary',  # Binary classification (cat=0, dog=1)
#                 # classes=['cats', 'dogs'],  # Explicit class order
#                 shuffle=True
#             )
            
#             self.validation_generator = validation_datagen.flow_from_directory(
#                 validation_path,
#                 target_size=target_size,
#                 batch_size=batch_size,
#                 class_mode='binary',
#                 # classes=['cats', 'dogs'],
#                 shuffle=False
#             )
            
#             self.test_generator = test_datagen.flow_from_directory(
#                 test_path,
#                 target_size=target_size,
#                 batch_size=batch_size,
#                 class_mode='binary',
#                 # classes=['cats', 'dogs'],
#                 shuffle=False
#             )
            
#             # Update input shape based on target size
#             self.input_shape = (target_size[0], target_size[1], 3)
            
#             logger.info(f"Data generators setup successfully")
#             logger.info(f"Found {self.train_generator.samples} training images")
#             logger.info(f"Found {self.validation_generator.samples} validation images")
#             logger.info(f"Found {self.test_generator.samples} test images")
            
#             return {
#                 "status": "success",
#                 "message": "Data generators setup successfully",
#                 "train_samples": self.train_generator.samples,
#                 "validation_samples": self.validation_generator.samples,
#                 "test_samples": self.test_generator.samples,
#                 "input_shape": self.input_shape
#             }
            
#         except Exception as e:
#             logger.error(f"Error setting up data generators: {str(e)}")
#             return {
#                 "status": "error",
#                 "message": f"Failed to setup data generators: {str(e)}"
#             }
    
#     def create_simple_cnn(self) -> keras.Model:
#         """Create a simple LeNet-inspired CNN"""
#         model = keras.Sequential([
#             # First Convolutional Block
#             layers.Conv2D(32, kernel_size=3, strides=1, padding='same', 
#                          activation='relu', input_shape=self.input_shape, name='conv1'),
#             layers.MaxPooling2D(pool_size=2, strides=2, name='pool1'),
            
#             # Second Convolutional Block
#             layers.Conv2D(64, kernel_size=3, strides=1, padding='same', 
#                          activation='relu', name='conv2'),
#             layers.MaxPooling2D(pool_size=2, strides=2, name='pool2'),
            
#             # Third Convolutional Block
#             layers.Conv2D(128, kernel_size=3, strides=1, padding='same', 
#                          activation='relu', name='conv3'),
#             layers.MaxPooling2D(pool_size=2, strides=2, name='pool3'),
            
#             # Flatten and Dense layers
#             layers.Flatten(),
#             layers.Dense(512, activation='relu', name='fc1'),
#             layers.Dropout(0.5),
#             layers.Dense(256, activation='relu', name='fc2'),
#             layers.Dropout(0.5),
#             layers.Dense(1, activation='sigmoid', name='output')  # Binary classification
#         ])
        
#         return model
    
#     def create_vgg_style_cnn(self) -> keras.Model:
#         """Create a VGG-inspired CNN optimized for cats vs dogs"""
#         model = keras.Sequential([
#             # Block 1
#             layers.Conv2D(32, 3, padding='same', activation='relu', 
#                          input_shape=self.input_shape),
#             layers.Conv2D(32, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(2, strides=2),
            
#             # Block 2
#             layers.Conv2D(64, 3, padding='same', activation='relu'),
#             layers.Conv2D(64, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(2, strides=2),
            
#             # Block 3
#             layers.Conv2D(128, 3, padding='same', activation='relu'),
#             layers.Conv2D(128, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(2, strides=2),
            
#             # Block 4
#             layers.Conv2D(256, 3, padding='same', activation='relu'),
#             layers.Conv2D(256, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(2, strides=2),
            
#             # Classifier
#             layers.Flatten(),
#             layers.Dense(512, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(256, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(1, activation='sigmoid')  # Binary classification
#         ])
        
#         return model
    
#     def create_resnet_style_cnn(self) -> keras.Model:
#         """Create a simplified ResNet-inspired CNN with skip connections"""
#         inputs = keras.Input(shape=self.input_shape)
        
#         # Initial convolution
#         x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
#         # Residual block 1
#         shortcut = x
#         x = layers.Conv2D(64, 3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2D(64, 3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
        
#         # Skip connection (ensuring same dimensions)
#         shortcut = layers.Conv2D(64, 1, padding='same')(shortcut)
#         x = layers.Add()([x, shortcut])
#         x = layers.Activation('relu')(x)
        
#         # Residual block 2
#         shortcut = x
#         x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2D(128, 3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
        
#         shortcut = layers.Conv2D(128, 1, strides=2, padding='same')(shortcut)
#         x = layers.Add()([x, shortcut])
#         x = layers.Activation('relu')(x)
        
#         # Global average pooling and output
#         x = layers.GlobalAveragePooling2D()(x)
#         x = layers.Dense(128, activation='relu')(x)
#         x = layers.Dropout(0.5)(x)
#         outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        
#         return keras.Model(inputs=inputs, outputs=outputs)
    
#     def create_model(self, model_type: str = "simple", target_size: Tuple[int, int] = (64, 64)) -> Dict[str, Any]:
#         """Create and compile a CNN model"""
#         try:
#             # Setup data generators first
#             data_setup = self.setup_data_generators(target_size=target_size)
#             if data_setup["status"] == "error":
#                 return data_setup
            
#             # Create model based on type
#             if model_type == "simple":
#                 self.model = self.create_simple_cnn()
#             elif model_type == "vgg":
#                 self.model = self.create_vgg_style_cnn()
#             elif model_type == "resnet":
#                 self.model = self.create_resnet_style_cnn()
#             else:
#                 raise ValueError(f"Unknown model type: {model_type}")
            
#             # Compile the model for binary classification
#             self.model.compile(
#                 optimizer=keras.optimizers.Adam(learning_rate=0.001),
#                 loss='binary_crossentropy',  # Changed to binary crossentropy
#                 metrics=['accuracy']
#             )
            
#             self.model_type = model_type
#             self.is_loaded = True
#             self.is_trained = False
            
#             logger.info(f"Created {model_type} model with {self.model.count_params()} parameters")
            
#             return {
#                 "status": "success",
#                 "message": f"{model_type.upper()} model created successfully",
#                 "model_type": model_type,
#                 "parameters": int(self.model.count_params()),
#                 "input_shape": self.input_shape,
#                 "output_classes": self.num_classes,
#                 "train_samples": self.train_generator.samples,
#                 "validation_samples": self.validation_generator.samples,
#                 "test_samples": self.test_generator.samples
#             }
            
#         except Exception as e:
#             logger.error(f"Error creating model: {str(e)}")
#             self.is_loaded = False
#             return {
#                 "status": "error",
#                 "message": f"Failed to create model: {str(e)}"
#             }
    
#     def train_model(self, epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
#         """Train the CNN model with real dataset"""
#         if self.model is None:
#             return {
#                 "status": "error",
#                 "message": "No model found. Please create a model first."
#             }
        
#         if self.train_generator is None:
#             return {
#                 "status": "error",
#                 "message": "No training data generator found. Please create a model first to setup data generators."
#             }
        
#         try:
#             logger.info("Starting training with real dataset...")
            
#             # Calculate steps per epoch
#             steps_per_epoch = self.train_generator.samples // batch_size
#             validation_steps = self.validation_generator.samples // batch_size
            
#             # Add callbacks for better training
#             callbacks = [
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_loss',
#                     patience=5,
#                     restore_best_weights=True
#                 ),
#                 keras.callbacks.ReduceLROnPlateau(
#                     monitor='val_loss',
#                     factor=0.2,
#                     patience=3,
#                     min_lr=0.0001
#                 )
#             ]
            
#             # Train the model
#             history = self.model.fit(
#                 self.train_generator,
#                 steps_per_epoch=steps_per_epoch,
#                 epochs=epochs,
#                 validation_data=self.validation_generator,
#                 validation_steps=validation_steps,
#                 callbacks=callbacks,
#                 verbose=1
#             )
            
#             # Save the model
#             os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#             self.model.save(self.model_path)
#             self.is_trained = True
            
#             # Get final metrics
#             final_loss = history.history['loss'][-1]
#             final_acc = history.history['accuracy'][-1]
#             final_val_loss = history.history['val_loss'][-1]
#             final_val_acc = history.history['val_accuracy'][-1]
            
#             logger.info("Training completed successfully!")
            
#             return {
#                 "status": "success",
#                 "message": "Model trained successfully with real dataset",
#                 "epochs": len(history.history['loss']),
#                 "final_metrics": {
#                     "loss": float(final_loss),
#                     "accuracy": float(final_acc),
#                     "val_loss": float(final_val_loss),
#                     "val_accuracy": float(final_val_acc)
#                 },
#                 "training_history": {
#                     "loss": [float(x) for x in history.history['loss']],
#                     "accuracy": [float(x) for x in history.history['accuracy']],
#                     "val_loss": [float(x) for x in history.history['val_loss']],
#                     "val_accuracy": [float(x) for x in history.history['val_accuracy']]
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"Error during training: {str(e)}")
#             return {
#                 "status": "error",
#                 "message": f"Training failed: {str(e)}"
#             }
    
#     def evaluate_model(self) -> Dict[str, Any]:
#         """Evaluate the model on test dataset"""
#         if self.model is None or not self.is_trained:
#             return {
#                 "status": "error",
#                 "message": "No trained model found. Please train a model first."
#             }
        
#         if self.test_generator is None:
#             return {
#                 "status": "error",
#                 "message": "No test data generator found."
#             }
        
#         try:
#             logger.info("Evaluating model on test dataset...")
            
#             # Calculate test steps
#             test_steps = self.test_generator.samples // self.test_generator.batch_size
            
#             # Evaluate the model
#             test_loss, test_accuracy = self.model.evaluate(
#                 self.test_generator,
#                 steps=test_steps,
#                 verbose=1
#             )
            
#             # Get predictions for detailed analysis
#             self.test_generator.reset()
#             predictions = self.model.predict(self.test_generator, steps=test_steps, verbose=1)
            
#             # Convert predictions to class predictions
#             pred_classes = (predictions > 0.5).astype(int).flatten()
#             true_classes = self.test_generator.classes[:len(pred_classes)]
            
#             # Calculate additional metrics
#             from sklearn.metrics import classification_report, confusion_matrix
            
#             # Classification report
#             class_report = classification_report(
#                 true_classes, pred_classes,
#                 target_names=['Cat', 'Dog'],
#                 output_dict=True
#             )
            
#             # Confusion matrix
#             conf_matrix = confusion_matrix(true_classes, pred_classes).tolist()
            
#             logger.info(f"Test accuracy: {test_accuracy:.4f}")
            
#             return {
#                 "status": "success",
#                 "test_accuracy": float(test_accuracy),
#                 "test_loss": float(test_loss),
#                 "classification_report": class_report,
#                 "confusion_matrix": conf_matrix,
#                 "total_test_samples": len(pred_classes)
#             }
            
#         except Exception as e:
#             logger.error(f"Error during evaluation: {str(e)}")
#             return {
#                 "status": "error",
#                 "message": f"Evaluation failed: {str(e)}"
#             }
    
#     def preprocess_image(self, image_data: bytes) -> np.ndarray:
#         """Preprocess image for prediction"""
#         try:
#             # Open image from bytes
#             image = Image.open(io.BytesIO(image_data))
            
#             # Convert to RGB if necessary
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
            
#             # Resize to model input size
#             target_size = (self.input_shape[0], self.input_shape[1])
#             image = image.resize(target_size)
            
#             # Convert to numpy array and normalize
#             image_array = np.array(image) / 255.0
            
#             # Add batch dimension
#             image_array = np.expand_dims(image_array, axis=0)
            
#             return image_array
            
#         except Exception as e:
#             logger.error(f"Error preprocessing image: {str(e)}")
#             raise
    
#     def predict_image(self, image_data: bytes) -> Dict[str, Any]:
#         """Make prediction on an image"""
#         if self.model is None:
#             return {
#                 "status": "error",
#                 "message": "No model found. Please create and train a model first."
#             }
        
#         try:
#             # Preprocess the image
#             processed_image = self.preprocess_image(image_data)
            
#             # Make prediction
#             prediction = self.model.predict(processed_image, verbose=0)[0][0]
            
#             # Convert to class prediction
#             predicted_class_idx = 1 if prediction > 0.5 else 0
#             predicted_class = 'Dog' if predicted_class_idx == 1 else 'Cat'
#             confidence = float(prediction if predicted_class_idx == 1 else 1 - prediction)
            
#             return {
#                 "status": "success",
#                 "predicted_class": predicted_class,
#                 "confidence": confidence,
#                 "probabilities": {
#                     "Cat": float(1 - prediction),
#                     "Dog": float(prediction)
#                 },
#                 "raw_prediction": float(prediction)
#             }
            
#         except Exception as e:
#             logger.error(f"Error during prediction: {str(e)}")
#             return {
#                 "status": "error",
#                 "message": f"Prediction failed: {str(e)}"
#             }
    
#     def predict_from_path(self, image_path: str) -> Dict[str, Any]:
#         """Make prediction on an image from file path"""
#         try:
#             with open(image_path, 'rb') as f:
#                 image_data = f.read()
#             return self.predict_image(image_data)
#         except Exception as e:
#             return {
#                 "status": "error",
#                 "message": f"Failed to load image from {image_path}: {str(e)}"
#             }
    
#     def get_model_info(self) -> Dict[str, Any]:
#         """Get information about the current model"""
#         if self.model is None:
#             return {
#                 "status": "error",
#                 "message": "No model found"
#             }
        
#         try:
#             info = {
#                 "status": "success",
#                 "model_type": self.model_type,
#                 "parameters": int(self.model.count_params()),
#                 "input_shape": self.input_shape,
#                 "output_classes": self.num_classes,
#                 "class_names": ['Cat', 'Dog'],
#                 "is_loaded": self.is_loaded,
#                 "is_trained": self.is_trained,
#                 "dataset_info": {}
#             }
            
#             # Add dataset information if generators exist
#             if self.train_generator:
#                 info["dataset_info"]["train_samples"] = self.train_generator.samples
#             if self.validation_generator:
#                 info["dataset_info"]["validation_samples"] = self.validation_generator.samples
#             if self.test_generator:
#                 info["dataset_info"]["test_samples"] = self.test_generator.samples
                
#             return info
            
#         except Exception as e:
#             return {
#                 "status": "error",
#                 "message": f"Error getting model info: {str(e)}"
#             }
    
#     def load_model(self, model_path: str = None) -> Dict[str, Any]:
#         """Load a saved model"""
#         try:
#             path = model_path or self.model_path
#             if os.path.exists(path):
#                 self.model = keras.models.load_model(path)
#                 self.is_loaded = True
#                 self.is_trained = True
                
#                 # Try to detect model type from the loaded model
#                 if self.model_type is None:
#                     num_layers = len(self.model.layers)
#                     if num_layers <= 12:
#                         self.model_type = "simple"
#                     elif num_layers <= 20:
#                         self.model_type = "vgg"
#                     else:
#                         self.model_type = "resnet"
                
#                 logger.info(f"Model loaded from {path}")
#                 return {
#                     "status": "success",
#                     "message": f"Model loaded successfully from {path}",
#                     "model_type": self.model_type,
#                     "parameters": int(self.model.count_params()) if self.model else 0
#                 }
#             else:
#                 return {
#                     "status": "error",
#                     "message": f"Model file not found at {path}"
#                 }
#         except Exception as e:
#             logger.error(f"Error loading model: {str(e)}")
#             self.is_loaded = False
#             return {
#                 "status": "error",
#                 "message": f"Failed to load model: {str(e)}"
#             }


# # Example usage:
# if __name__ == "__main__":
#     # Initialize the model
#     cnn_model = CNNModel()
    
#     # Create and train a model
#     print("Creating model...")
#     result = cnn_model.create_model(model_type="simple", target_size=(150, 150))
#     print(result)
    
#     if result["status"] == "success":
#         print("\nTraining model...")
#         train_result = cnn_model.train_model(epochs=20, batch_size=32)
#         print(train_result)
        
#         if train_result["status"] == "success":
#             print("\nEvaluating model...")
#             eval_result = cnn_model.evaluate_model()
#             print(eval_result)
            
#             # Test prediction on a single image
#             # Replace 'path/to/test/image.jpg' with actual path
#             # pred_result = cnn_model.predict_from_path('path/to/test/image.jpg')
#             # print(pred_result)




import os
import io
import logging
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------------------------------------------------------
# TensorFlow / Logging setup
# -----------------------------------------------------------------------------
# Run functions eagerly to make debugging easier (optional)
tf.config.run_functions_eagerly(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNModel:
    def __init__(self):
        self.model: keras.Model | None = None
        self.model_type: str | None = None
        # Default to a more reasonable size for cats vs dogs
        self.input_shape = (150, 150, 3)
        self.num_classes = 2
        # Will be detected from folders; fallback kept for safety
        self.class_names: list[str] = ["cats", "dogs"]

        self.model_path = "app/ml_models/cnn_cats_dogs.h5"
        self.dataset_path = "app/dataset"  # Expected structure: train/ | validation/ | test/
        self.is_loaded = False
        self.is_trained = False

        # Data generators for real dataset
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def setup_data_generators(
        self,
        batch_size: int = 32,
        target_size: Tuple[int, int] = (150, 150),
    ) -> Dict[str, Any]:
        """Setup data generators for training, validation, and testing."""
        try:
            # Data augmentation for training data
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode="nearest",
            )

            # Only rescaling for validation and test data
            validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
            test_datagen = ImageDataGenerator(rescale=1.0 / 255)

            # Paths to your dataset folders
            train_path = os.path.join(self.dataset_path, "train")
            validation_path = os.path.join(self.dataset_path, "validation")
            test_path = os.path.join(self.dataset_path, "test")

            # Check if paths exist
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training data path not found: {train_path}")
            if not os.path.exists(validation_path):
                raise FileNotFoundError(f"Validation data path not found: {validation_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test data path not found: {test_path}")

            # Create generators (let Keras infer class names from folder names)
            self.train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=target_size,
                batch_size=batch_size,
                class_mode="binary",
                shuffle=True,
            )

            self.validation_generator = validation_datagen.flow_from_directory(
                validation_path,
                target_size=target_size,
                batch_size=batch_size,
                class_mode="binary",
                shuffle=False,
            )

            self.test_generator = test_datagen.flow_from_directory(
                test_path,
                target_size=target_size,
                batch_size=batch_size,
                class_mode="binary",
                shuffle=False,
            )

            # Save the detected class order (e.g., {"cats":0, "dogs":1})
            # Ensure list is ordered by index value
            idx_to_class = {v: k for k, v in self.train_generator.class_indices.items()}
            self.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

            # Update input shape based on target size
            self.input_shape = (target_size[0], target_size[1], 3)

            logger.info("Data generators setup successfully")
            logger.info(f"Class indices: {self.train_generator.class_indices}")
            logger.info(f"Found {self.train_generator.samples} training images")
            logger.info(f"Found {self.validation_generator.samples} validation images")
            logger.info(f"Found {self.test_generator.samples} test images")

            return {
                "status": "success",
                "message": "Data generators setup successfully",
                "train_samples": self.train_generator.samples,
                "validation_samples": self.validation_generator.samples,
                "test_samples": self.test_generator.samples,
                "input_shape": self.input_shape,
                "class_names": self.class_names,
            }

        except Exception as e:
            logger.exception("Error setting up data generators")
            return {
                "status": "error",
                "message": f"Failed to setup data generators: {str(e)}",
            }

    # ------------------------------------------------------------------
    # Architectures
    # ------------------------------------------------------------------
    def create_simple_cnn(self) -> keras.Model:
        """Create a simple LeNet-inspired CNN."""
        model = keras.Sequential(
            [
                layers.Conv2D(
                    32,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="relu",
                    input_shape=self.input_shape,
                    name="conv1",
                ),
                layers.MaxPooling2D(pool_size=2, strides=2, name="pool1"),
                layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
                layers.MaxPooling2D(2, name="pool2"),
                layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3"),
                layers.MaxPooling2D(2, name="pool3"),
                layers.Flatten(),
                layers.Dense(512, activation="relu", name="fc1"),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu", name="fc2"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid", name="output"),  # Binary
            ]
        )
        return model

    def create_vgg_style_cnn(self) -> keras.Model:
        """Create a VGG-inspired CNN optimized for cats vs dogs."""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=self.input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2, strides=2),

            # Block 2
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2, strides=2),

            # Block 3
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2, strides=2),

            # Block 4
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2, strides=2),

            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation="relu", name="fc1"),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu", name="fc2"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")  # Binary classification (cats vs dogs)
        ])
        return model


    def create_resnet_style_cnn(self) -> keras.Model:
        """Create a simplified ResNet-inspired CNN with skip connections."""
        inputs = keras.Input(shape=self.input_shape)

        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Residual block 1
        shortcut = x
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        shortcut = layers.Conv2D(64, 1, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)

        # Residual block 2
        shortcut = x
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        shortcut = layers.Conv2D(128, 1, strides=2, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)

        # Global average pooling and output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)  # Binary

        return keras.Model(inputs=inputs, outputs=outputs)

    # ------------------------------------------------------------------
    # Create / Compile
    # ------------------------------------------------------------------
    def create_model(
        self,
        model_type: str = "simple",
        target_size: Tuple[int, int] = (150, 150),
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Create and compile a CNN model, after setting up data generators."""
        try:
            # Setup data generators first
            data_setup = self.setup_data_generators(batch_size=batch_size, target_size=target_size)
            if data_setup["status"] == "error":
                return data_setup

            # Create model based on type
            if model_type == "simple":
                self.model = self.create_simple_cnn()
            elif model_type == "vgg":
                self.model = self.create_vgg_style_cnn()
            elif model_type == "resnet":
                self.model = self.create_resnet_style_cnn()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Compile the model for binary classification
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            self.model_type = model_type
            self.is_loaded = True
            self.is_trained = False

            logger.info(f"Created {model_type} model with {self.model.count_params()} parameters")

            return {
                "status": "success",
                "message": f"{model_type.upper()} model created successfully",
                "model_type": model_type,
                "parameters": int(self.model.count_params()),
                "input_shape": self.input_shape,
                "output_classes": self.num_classes,
                "train_samples": self.train_generator.samples,
                "validation_samples": self.validation_generator.samples,
                "test_samples": self.test_generator.samples,
                "class_names": self.class_names,
            }

        except Exception as e:
            logger.exception("Error creating model")
            self.is_loaded = False
            return {"status": "error", "message": f"Failed to create model: {str(e)}"}

    # ------------------------------------------------------------------
    # Train / Evaluate
    # ------------------------------------------------------------------
    def train_model(self, epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        """Train the CNN model with real dataset."""
        if self.model is None:
            return {"status": "error", "message": "No model found. Please create a model first."}
        if self.train_generator is None:
            return {
                "status": "error",
                "message": "No training data generator found. Please create a model first to setup data generators.",
            }

        try:
            logger.info("Starting training with real dataset...")

            # Calculate steps per epoch (at least 1)
            steps_per_epoch = max(1, self.train_generator.samples // batch_size)
            validation_steps = max(1, self.validation_generator.samples // batch_size)

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-4),
            ]

            # Train the model
            history = self.model.fit(
                self.train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=self.validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
            )

            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            self.is_trained = True

            # Final metrics
            final_loss = float(history.history["loss"][-1])
            final_acc = float(history.history["accuracy"][-1])
            final_val_loss = float(history.history["val_loss"][-1])
            final_val_acc = float(history.history["val_accuracy"][-1])

            logger.info("Training completed successfully!")

            return {
                "status": "success",
                "message": "Model trained successfully with real dataset",
                "epochs": len(history.history["loss"]),
                "final_metrics": {
                    "loss": final_loss,
                    "accuracy": final_acc,
                    "val_loss": final_val_loss,
                    "val_accuracy": final_val_acc,
                },
                "training_history": {
                    "loss": [float(x) for x in history.history["loss"]],
                    "accuracy": [float(x) for x in history.history["accuracy"]],
                    "val_loss": [float(x) for x in history.history["val_loss"]],
                    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
                },
            }

        except Exception as e:
            logger.exception("Error during training")
            return {"status": "error", "message": f"Training failed: {str(e)}"}

    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the model on test dataset."""
        if self.model is None or not self.is_trained:
            return {"status": "error", "message": "No trained model found. Please train a model first."}
        if self.test_generator is None:
            return {"status": "error", "message": "No test data generator found."}

        try:
            logger.info("Evaluating model on test dataset...")

            test_steps = max(1, self.test_generator.samples // self.test_generator.batch_size)

            # Evaluate the model
            test_loss, test_accuracy = self.model.evaluate(self.test_generator, steps=test_steps, verbose=1)

            # Predictions for detailed analysis
            self.test_generator.reset()
            predictions = self.model.predict(self.test_generator, steps=test_steps, verbose=1)

            # Convert predictions to class predictions
            pred_classes = (predictions > 0.5).astype(int).flatten()
            true_classes = self.test_generator.classes[: len(pred_classes)]

            # Additional metrics
            from sklearn.metrics import classification_report, confusion_matrix

            class_report = classification_report(
                true_classes,
                pred_classes,
                target_names=[name.capitalize() for name in self.class_names],
                output_dict=True,
            )
            conf_matrix = confusion_matrix(true_classes, pred_classes).tolist()

            logger.info(f"Test accuracy: {test_accuracy:.4f}")

            return {
                "status": "success",
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "classification_report": class_report,
                "confusion_matrix": conf_matrix,
                "total_test_samples": int(len(pred_classes)),
            }

        except Exception as e:
            logger.exception("Error during evaluation")
            return {"status": "error", "message": f"Evaluation failed: {str(e)}"}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess a raw image bytes for prediction."""
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            target_size = (self.input_shape[0], self.input_shape[1])
            image = image.resize(target_size)

            image_array = np.array(image).astype(np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # add batch dim
            return image_array
        except Exception as e:
            logger.exception("Error preprocessing image")
            raise

    def predict_image(self, image_data: bytes) -> Dict[str, Any]:
        """Make prediction on an image (bytes)."""
        if self.model is None:
            return {
                "status": "error",
                "message": "No model found. Please create and train a model first.",
            }
        try:
            processed_image = self.preprocess_image(image_data)
            prediction = float(self.model.predict(processed_image, verbose=0)[0][0])

            predicted_class_idx = 1 if prediction > 0.5 else 0
            # Use detected class names if available
            label_candidates = [name.capitalize() for name in self.class_names]
            if len(label_candidates) < 2:
                label_candidates = ["Cat", "Dog"]
            predicted_class = label_candidates[predicted_class_idx]
            confidence = prediction if predicted_class_idx == 1 else 1.0 - prediction

            return {
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "probabilities": {
                    label_candidates[0]: float(1.0 - prediction),
                    label_candidates[1]: float(prediction),
                },
                "raw_prediction": prediction,
            }
        except Exception as e:
            logger.exception("Error during prediction")
            return {"status": "error", "message": f"Prediction failed: {str(e)}"}

    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """Make prediction on an image from file path."""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return self.predict_image(image_data)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load image from {image_path}: {str(e)}"}

    # ------------------------------------------------------------------
    # Introspection / Persistence
    # ------------------------------------------------------------------
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "error", "message": "No model found"}
        try:
            info = {
                "status": "success",
                "model_type": self.model_type,
                "parameters": int(self.model.count_params()),
                "input_shape": self.input_shape,
                "output_classes": self.num_classes,
                "class_names": [name.capitalize() for name in self.class_names],
                "is_loaded": self.is_loaded,
                "is_trained": self.is_trained,
                "dataset_info": {},
            }
            if self.train_generator:
                info["dataset_info"]["train_samples"] = self.train_generator.samples
            if self.validation_generator:
                info["dataset_info"]["validation_samples"] = self.validation_generator.samples
            if self.test_generator:
                info["dataset_info"]["test_samples"] = self.test_generator.samples
            return info
        except Exception as e:
            return {"status": "error", "message": f"Error getting model info: {str(e)}"}

    def load_model(self, model_path: str | None = None) -> Dict[str, Any]:
        """Load a saved model and ensure proper architecture setup."""
        try:
            path = model_path or self.model_path
            if os.path.exists(path):
                # Load the model
                self.model = keras.models.load_model(path)
                self.is_loaded = True
                self.is_trained = True
                
                # Extract input shape from the loaded model
                self.input_shape = self.model.layers[0].input_shape[0][1:]  # Get (150, 150, 3)
                
                # Try to infer model type from layer names and structure
                layer_names = [layer.name for layer in self.model.layers]
                
                if 'resnet' in str(layer_names).lower() or any('add' in name for name in layer_names):
                    self.model_type = "resnet"
                elif 'vgg' in str(layer_names).lower() or len(self.model.layers) > 15:
                    self.model_type = "vgg"
                else:
                    self.model_type = "simple"
                
                # Update class names if they were saved with the model
                if hasattr(self.model, 'class_names_'):
                    self.class_names = self.model.class_names_
                
                logger.info(f"Model loaded from {path}")
                logger.info(f"Input shape: {self.input_shape}")
                logger.info(f"Model type: {self.model_type}")
                
                return {
                    "status": "success",
                    "message": f"Model loaded successfully from {path}",
                    "model_type": self.model_type,
                    "input_shape": self.input_shape,
                    "parameters": int(self.model.count_params()),
                }
            else:
                return {"status": "error", "message": f"Model file not found at {path}"}
        except Exception as e:
            logger.exception("Error loading model")
            self.is_loaded = False
            return {"status": "error", "message": f"Failed to load model: {str(e)}"}

# -----------------------------------------------------------------------------
# Example usage (run as a script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the model
    cnn_model = CNNModel()

    # Create and train a model
    print("Creating model...")
    result = cnn_model.create_model(model_type="simple", target_size=(150, 150), batch_size=32)
    print(result)

    if result.get("status") == "success":
        print("\nTraining model...")
        train_result = cnn_model.train_model(epochs=20, batch_size=32)
        print(train_result)

        if train_result.get("status") == "success":
            print("\nEvaluating model...")
            eval_result = cnn_model.evaluate_model()
            print(eval_result)
            # Example single-image prediction
            # pred_result = cnn_model.predict_from_path('path/to/test/image.jpg')
            # print(pred_result)
