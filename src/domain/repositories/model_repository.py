"""Model repository interface."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from src.domain.entities.prediction import Prediction


class ModelRepository(ABC):
    """Abstract interface for model operations."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to the model file.
        """
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """Save the current model to disk.
        
        Args:
            model_path: Path where to save the model.
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Prediction:
        """Make a prediction on an image.
        
        Args:
            image: Preprocessed image array.
            
        Returns:
            Prediction entity with results.
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32
    ) -> dict:
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
        pass
    
    @abstractmethod
    def get_model(self):
        """Get the underlying model object.
        
        Returns:
            The model object (e.g., Keras model).
        """
        pass

