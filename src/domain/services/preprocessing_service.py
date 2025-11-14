"""Preprocessing service interface."""
from abc import ABC, abstractmethod
import numpy as np

from src.domain.entities.sign_image import SignImage


class PreprocessingService(ABC):
    """Abstract interface for image preprocessing."""
    
    @abstractmethod
    def preprocess_image(self, image: SignImage) -> np.ndarray:
        """Preprocess an image for model input.
        
        Args:
            image: SignImage entity to preprocess.
            
        Returns:
            Preprocessed image as numpy array.
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess a batch of images.
        
        Args:
            images: Batch of images as numpy array.
            
        Returns:
            Preprocessed batch as numpy array.
        """
        pass

