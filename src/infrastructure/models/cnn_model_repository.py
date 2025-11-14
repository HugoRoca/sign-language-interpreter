"""CNN model repository implementation."""
from typing import Optional, Dict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from src.domain.repositories.model_repository import ModelRepository
from src.domain.entities.prediction import Prediction
from src.infrastructure.models.training_progress_callback import TrainingProgressCallback
from config.settings import (
    INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE,
    MODEL_CHECKPOINT_DIR
)


class CNNModelRepository(ModelRepository):
    """Concrete implementation of ModelRepository using CNN."""
    
    def __init__(self, num_classes: int = NUM_CLASSES, input_shape: tuple = INPUT_SHAPE):
        """Initialize the model repository.
        
        Args:
            num_classes: Number of output classes.
            input_shape: Input shape for the model (height, width, channels).
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model: Optional[keras.Model] = None
        self.class_names: list = []
    
    def _build_model(self) -> keras.Model:
        """Build the CNN model architecture.
        
        Returns:
            Compiled Keras model.
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Fourth convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model(self) -> keras.Model:
        """Get the underlying model object."""
        if self.model is None:
            self.model = self._build_model()
        return self.model
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to the model file.
        """
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def save_model(self, model_path: str) -> None:
        """Save the current model to disk.
        
        Args:
            model_path: Path where to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def set_class_names(self, class_names: list) -> None:
        """Set the class names for predictions.
        
        Args:
            class_names: List of class names in order.
        """
        self.class_names = class_names
    
    def predict(self, image: np.ndarray) -> Prediction:
        """Make a prediction on an image.
        
        Args:
            image: Preprocessed image array.
            
        Returns:
            Prediction entity with results.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load or train a model first.")
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        probabilities = predictions[0]
        
        predicted_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_idx])
        
        # Map probabilities to class names
        all_probabilities = {}
        if self.class_names:
            for idx, class_name in enumerate(self.class_names):
                all_probabilities[class_name] = float(probabilities[idx])
        else:
            all_probabilities[str(predicted_idx)] = confidence
        
        predicted_class = (
            self.class_names[predicted_idx]
            if self.class_names and predicted_idx < len(self.class_names)
            else str(predicted_idx)
        )
        
        return Prediction(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probabilities
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Dict:
        """Train the model.
        
        Args:
            X_train: Training images.
            y_train: Training labels.
            X_val: Validation images (optional).
            y_val: Validation labels (optional).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            
        Returns:
            Training history dictionary.
        """
        if self.model is None:
            self.model = self._build_model()
        
        # Create callbacks
        callback_list = [
            # Custom progress callback
            TrainingProgressCallback(total_epochs=epochs),
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add checkpoint callback if checkpoint directory exists
        checkpoint_dir = MODEL_CHECKPOINT_DIR
        if checkpoint_dir.exists():
            callback_list.append(
                callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_dir / "checkpoint-{epoch:02d}-{val_loss:.2f}.keras"),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        return history.history

