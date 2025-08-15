"""
MNIST Model Implementation
=========================
Contains model architecture, training, and prediction logic
Save as: app/models/mnist_model.py
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class MNISTModel:
    def __init__(self, model_path: str = "mnist_model.h5"):
        self.model_path = model_path
        self.model = None
        self.x_test = None
        self.y_test = None
        self.is_loaded = False
        
    def build_model(self) -> tf.keras.Model:
        """Build the neural network architecture"""
        model = tf.keras.models.Sequential([
            # Flatten 28x28 images to 784 features
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            
            # Hidden layers with ReLU activation
            tf.keras.layers.Dense(128, activation='relu', name='hidden_layer_1'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            
            tf.keras.layers.Dense(64, activation='relu', name='hidden_layer_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            
            # Output layer: 10 neurons for digits 0-9
            tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess MNIST dataset"""
        logger.info("Loading MNIST dataset...")
        
        # Load dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Store test data for later use
        self.x_test = x_test
        self.y_test = y_test
        
        logger.info(f"Data loaded - Train: {x_train.shape}, Test: {x_test.shape}")
        return x_train, y_train, x_test, y_test
    
    def train_model(self, epochs: int = 10, batch_size: int = 128, validation_split: float = 0.1) -> float:
        """Train the model and save it"""
        logger.info("Starting model training...")
        
        # Load data
        x_train, y_train, x_test, y_test = self.load_data()
        
        # Build model
        self.model = self.build_model()
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        logger.info(f"Training completed - Test Accuracy: {test_accuracy:.4f}")
        
        # Save the model
        self.save_model()
        self.is_loaded = True
        
        return test_accuracy
    
    def load_model(self) -> bool:
        """Load pre-trained model from file"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading model from {self.model_path}")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                
                # Load test data for evaluation
                (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.x_test = x_test.astype('float32') / 255.0
                self.y_test = y_test
                
                self.is_loaded = True
                logger.info("Model loaded successfully!")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                return False
        else:
            logger.info("No existing model found")
            return False
    
    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)
            self.model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    def predict_single(self, image: np.ndarray) -> dict:
        """Make prediction on a single image"""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
        
        # Ensure image is the right shape
        if image.shape != (28, 28):
            raise ValueError(f"Image must be 28x28, got {image.shape}")
        
        # Add batch dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        probabilities = prediction[0].tolist()
        
        return {
            "predicted_digit": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }
    
    def predict_batch(self, images: np.ndarray) -> List[dict]:
        """Make predictions on batch of images"""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
        
        predictions = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        results = []
        for i in range(len(images)):
            results.append({
                "predicted_digit": int(predicted_classes[i]),
                "confidence": float(confidences[i]),
                "probabilities": predictions[i].tolist()
            })
        
        return results
    
    def preprocess_image_array(self, image_data: List[List[float]]) -> np.ndarray:
        """Preprocess 28x28 array for prediction"""
        # Convert to numpy array
        img = np.array(image_data, dtype=np.float32)
        
        # Validate shape
        if img.shape != (28, 28):
            raise ValueError(f"Image must be 28x28, got {img.shape}")
        
        # Normalize if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        return img
    
    def preprocess_base64_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 image to preprocessed array"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale and resize
            image = image.convert('L')
            image = image.resize((28, 28))
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            return img_array
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")
    
    def preprocess_uploaded_file(self, file_data: bytes) -> np.ndarray:
        """Convert uploaded file to preprocessed array"""
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(file_data))
            
            # Convert to grayscale and resize
            image = image.convert('L')
            image = image.resize((28, 28))
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            return img_array
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
    
    def get_test_sample(self, index: Optional[int] = None) -> dict:
        """Get a test sample for evaluation"""
        if self.x_test is None or self.y_test is None:
            raise ValueError("Test data not loaded")
        
        if index is None:
            index = np.random.randint(0, len(self.x_test))
        elif index < 0 or index >= len(self.x_test):
            raise ValueError(f"Index must be between 0 and {len(self.x_test) - 1}")
        
        # Get sample
        image = self.x_test[index]
        true_label = int(self.y_test[index])
        
        # Make prediction
        prediction_result = self.predict_single(image)
        
        return {
            **prediction_result,
            "test_info": {
                "true_label": true_label,
                "sample_index": index,
                "correct_prediction": prediction_result["predicted_digit"] == true_label
            }
        }
    
    def get_model_info(self) -> dict:
        """Get model information and performance"""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        return {
            "model_summary": {
                "total_params": int(self.model.count_params()),
                "trainable_params": int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
                "layers": len(self.model.layers)
            },
            "performance": {
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss)
            },
            "input_shape": list(self.model.input_shape[1:]),
            "output_shape": list(self.model.output_shape[1:]),
            "model_path": self.model_path
        }
    
    def evaluate_batch(self, count: int) -> dict:
        """Evaluate model on a batch of test samples"""
        if self.x_test is None or self.y_test is None:
            raise ValueError("Test data not loaded")
        
        if count > len(self.x_test):
            count = len(self.x_test)
        
        # Get random samples
        indices = np.random.choice(len(self.x_test), count, replace=False)
        batch_images = self.x_test[indices]
        true_labels = self.y_test[indices]
        
        # Make predictions
        predictions = self.predict_batch(batch_images)
        
        # Prepare results
        results = []
        correct_count = 0
        
        for i, prediction in enumerate(predictions):
            is_correct = prediction["predicted_digit"] == true_labels[i]
            if is_correct:
                correct_count += 1
            
            results.append({
                "index": int(indices[i]),
                "true_label": int(true_labels[i]),
                "predicted_digit": prediction["predicted_digit"],
                "confidence": prediction["confidence"],
                "correct": is_correct
            })
        
        accuracy = correct_count / count
        
        return {
            "batch_size": count,
            "accuracy": accuracy,
            "correct_predictions": correct_count,
            "total_predictions": count,
            "results": results
        }

# Global model instance
mnist_model = MNISTModel()

def initialize_model():
    """Initialize the model (load existing or train new one)"""
    if not mnist_model.load_model():
        logger.info("No existing model found. Training new model...")
        accuracy = mnist_model.train_model(epochs=5)  # Quick training for demo
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
    return mnist_model