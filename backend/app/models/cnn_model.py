import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from typing import Tuple, Dict, Any
import logging
from PIL import Image
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNModel:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.input_shape = (64, 64, 3)
        self.num_classes = 2
        self.class_names = ['Cat', 'Dog']
        self.model_path = "models/cnn_cats_dogs.h5"
        
    def create_simple_cnn(self) -> keras.Model:
        """Create a simple LeNet-inspired CNN"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(6, kernel_size=5, strides=1, padding='same', 
                         activation='relu', input_shape=self.input_shape, name='conv1'),
            layers.MaxPooling2D(pool_size=2, strides=2, name='pool1'),
            
            # Second Convolutional Block
            layers.Conv2D(16, kernel_size=5, strides=1, padding='same', 
                         activation='relu', name='conv2'),
            layers.MaxPooling2D(pool_size=2, strides=2, name='pool2'),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(120, activation='relu', name='fc1'),
            layers.Dropout(0.5),
            layers.Dense(84, activation='relu', name='fc2'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        return model
    
    def create_vgg_style_cnn(self) -> keras.Model:
        """Create a VGG-inspired CNN"""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(64, 3, padding='same', activation='relu', 
                         input_shape=self.input_shape),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2, strides=2),
            
            # Block 2
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2, strides=2),
            
            # Block 3
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2, strides=2),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_resnet_style_cnn(self) -> keras.Model:
        """Create a simplified ResNet-inspired CNN with skip connections"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual block
        shortcut = x
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection (ensuring same dimensions)
        shortcut = layers.Conv2D(64, 1, padding='same')(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # Another residual block
        shortcut = x
        x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        shortcut = layers.Conv2D(128, 1, strides=2, padding='same')(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # Global average pooling and output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs)
    
    def create_model(self, model_type: str = "simple") -> Dict[str, Any]:
        """Create and compile a CNN model"""
        try:
            if model_type == "simple":
                self.model = self.create_simple_cnn()
            elif model_type == "vgg":
                self.model = self.create_vgg_style_cnn()
            elif model_type == "resnet":
                self.model = self.create_resnet_style_cnn()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Compile the model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model_type = model_type
            
            logger.info(f"Created {model_type} model with {self.model.count_params()} parameters")
            
            return {
                "status": "success",
                "message": f"{model_type.upper()} model created successfully",
                "model_type": model_type,
                "parameters": int(self.model.count_params()),
                "input_shape": self.input_shape,
                "output_classes": self.num_classes
            }
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to create model: {str(e)}"
            }
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data"""
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create 64x64x3 synthetic images
            image = np.random.normal(0.5, 0.2, (64, 64, 3))
            
            # Create distinct patterns for cats (0) vs dogs (1)
            is_cat = i < num_samples // 2
            
            if is_cat:
                # Cats: add circular patterns (simulating cat faces)
                center_x, center_y = 32, 32
                y, x = np.ogrid[:64, :64]
                mask = (x - center_x)**2 + (y - center_y)**2 <= 400  # radius 20
                image[mask] += 0.3
                labels.append(0)  # Cat
            else:
                # Dogs: add rectangular patterns (simulating dog snouts)
                image[25:40, 20:45] += 0.3
                labels.append(1)  # Dog
            
            # Clip values to valid range
            image = np.clip(image, 0, 1)
            images.append(image)
        
        images = np.array(images)
        labels = keras.utils.to_categorical(labels, self.num_classes)
        
        return images, labels
    
    def train_model(self, epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        """Train the CNN model"""
        if self.model is None:
            return {
                "status": "error",
                "message": "No model found. Please create a model first."
            }
        
        try:
            logger.info("Generating synthetic training data...")
            
            # Generate training data
            X_train, y_train = self.generate_synthetic_data(800)
            X_val, y_val = self.generate_synthetic_data(200)
            
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Training labels shape: {y_train.shape}")
            
            # Train the model
            logger.info("Starting training...")
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            
            # Get final metrics
            final_loss = history.history['loss'][-1]
            final_acc = history.history['accuracy'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            
            logger.info("Training completed successfully!")
            
            return {
                "status": "success",
                "message": "Model trained successfully",
                "epochs": epochs,
                "final_metrics": {
                    "loss": float(final_loss),
                    "accuracy": float(final_acc),
                    "val_loss": float(final_val_loss),
                    "val_accuracy": float(final_val_acc)
                },
                "training_history": {
                    "loss": [float(x) for x in history.history['loss']],
                    "accuracy": [float(x) for x in history.history['accuracy']],
                    "val_loss": [float(x) for x in history.history['val_loss']],
                    "val_accuracy": [float(x) for x in history.history['val_accuracy']]
                }
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return {
                "status": "error",
                "message": f"Training failed: {str(e)}"
            }
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for prediction"""
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((64, 64))
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_image(self, image_data: bytes) -> Dict[str, Any]:
        """Make prediction on an image"""
        if self.model is None:
            return {
                "status": "error",
                "message": "No model found. Please create and train a model first."
            }
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            return {
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": {
                    "Cat": float(probabilities[0]),
                    "Dog": float(probabilities[1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }
    
    def generate_random_prediction(self) -> Dict[str, Any]:
        """Generate a random test image and make prediction"""
        if self.model is None:
            return {
                "status": "error",
                "message": "No model found. Please create and train a model first."
            }
        
        try:
            # Generate random test image
            test_image = np.random.normal(0.5, 0.2, (1, 64, 64, 3))
            test_image = np.clip(test_image, 0, 1)
            
            # Add some pattern (randomly choose cat or dog pattern)
            if np.random.random() > 0.5:
                # Cat pattern (circular)
                center_x, center_y = 32, 32
                y, x = np.ogrid[:64, :64]
                mask = (x - center_x)**2 + (y - center_y)**2 <= 400
                test_image[0][mask] += 0.3
                actual_class = "Cat"
            else:
                # Dog pattern (rectangular)
                test_image[0, 25:40, 20:45] += 0.3
                actual_class = "Dog"
            
            test_image = np.clip(test_image, 0, 1)
            
            # Make prediction
            predictions = self.model.predict(test_image, verbose=0)
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Convert image to base64 for visualization
            image_pil = Image.fromarray((test_image[0] * 255).astype(np.uint8))
            buffer = io.BytesIO()
            image_pil.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "status": "success",
                "predicted_class": predicted_class,
                "actual_class": actual_class,
                "confidence": confidence,
                "probabilities": {
                    "Cat": float(probabilities[0]),
                    "Dog": float(probabilities[1])
                },
                "test_image_base64": image_base64
            }
            
        except Exception as e:
            logger.error(f"Error generating random prediction: {str(e)}")
            return {
                "status": "error",
                "message": f"Random prediction failed: {str(e)}"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.model is None:
            return {
                "status": "error",
                "message": "No model found"
            }
        
        try:
            return {
                "status": "success",
                "model_type": self.model_type,
                "parameters": int(self.model.count_params()),
                "input_shape": self.input_shape,
                "output_classes": self.num_classes,
                "class_names": self.class_names,
                "model_summary": str(self.model.summary())
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting model info: {str(e)}"
            }
    
    def load_model(self, model_path: str = None) -> Dict[str, Any]:
        """Load a saved model"""
        try:
            path = model_path or self.model_path
            if os.path.exists(path):
                self.model = keras.models.load_model(path)
                logger.info(f"Model loaded from {path}")
                return {
                    "status": "success",
                    "message": f"Model loaded successfully from {path}"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Model file not found at {path}"
                }
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to load model: {str(e)}"
            }