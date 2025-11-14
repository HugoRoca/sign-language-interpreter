"""Image preprocessing service implementation."""
import numpy as np
from typing import Tuple

from src.domain.services.preprocessing_service import PreprocessingService
from src.domain.entities.sign_image import SignImage
from config.settings import IMAGE_SIZE


class ImagePreprocessingService(PreprocessingService):
    """Concrete implementation of PreprocessingService."""
    
    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE):
        """Initialize the preprocessing service.
        
        Args:
            target_size: Target size for images (width, height).
        """
        self.target_size = target_size
    
    def preprocess_image(self, image: SignImage) -> np.ndarray:
        """Preprocess an image for model input.
        
        Args:
            image: SignImage entity to preprocess.
            
        Returns:
            Preprocessed image as numpy array.
        """
        img = image.image_data.copy()
        
        # Resize if needed
        if img.shape[:2] != self.target_size:
            from PIL import Image
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize(self.target_size)
            img = np.array(img_pil, dtype=np.float32) / 255.0
        
        # Ensure correct shape: (height, width, channels)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess a batch of images.
        
        Args:
            images: Batch of images as numpy array.
            
        Returns:
            Preprocessed batch as numpy array.
        """
        processed = []
        
        for img in images:
            # Normalize if not already normalized
            if img.max() > 1.0:
                img = img / 255.0
            
            # Resize if needed
            if img.shape[:2] != self.target_size:
                from PIL import Image
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil = img_pil.resize(self.target_size)
                img = np.array(img_pil, dtype=np.float32) / 255.0
            
            # Ensure correct shape
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            
            processed.append(img)
        
        return np.array(processed, dtype=np.float32)

