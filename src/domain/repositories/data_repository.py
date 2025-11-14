"""Data repository interface."""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from src.domain.entities.sign_image import SignImage


class DataRepository(ABC):
    """Abstract interface for data loading operations."""
    
    @abstractmethod
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data.
        
        Returns:
            Tuple of (images, labels) as numpy arrays.
        """
        pass
    
    @abstractmethod
    def load_image(self, image_path: str) -> SignImage:
        """Load a single image for prediction.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            SignImage entity.
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get list of class names.
        
        Returns:
            List of class names in order.
        """
        pass

