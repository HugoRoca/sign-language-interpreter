"""Prediction service for making predictions."""
from pathlib import Path
import numpy as np

from src.domain.repositories.data_repository import DataRepository
from src.domain.repositories.model_repository import ModelRepository
from src.domain.services.preprocessing_service import PreprocessingService
from src.domain.entities.prediction import Prediction


class PredictionService:
    """Service for making predictions on sign language images."""
    
    def __init__(
        self,
        data_repository: DataRepository,
        model_repository: ModelRepository,
        preprocessing_service: PreprocessingService
    ):
        """Initialize the prediction service.
        
        Args:
            data_repository: Repository for loading images.
            model_repository: Repository for model operations.
            preprocessing_service: Service for preprocessing images.
        """
        self.data_repository = data_repository
        self.model_repository = model_repository
        self.preprocessing_service = preprocessing_service
    
    def predict_from_path(self, image_path: str) -> Prediction:
        """Make a prediction from an image file path.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Prediction entity with results.
        """
        # Load image
        sign_image = self.data_repository.load_image(image_path)
        
        # Preprocess image
        preprocessed = self.preprocessing_service.preprocess_image(sign_image)
        
        # Make prediction
        prediction = self.model_repository.predict(preprocessed[0])
        
        return prediction
    
    def predict_from_array(self, image_array) -> Prediction:
        """Make a prediction from a numpy array.
        
        Args:
            image_array: Image as numpy array.
            
        Returns:
            Prediction entity with results.
        """
        # Preprocess image
        preprocessed = self.preprocessing_service.preprocess_batch(
            np.expand_dims(image_array, axis=0)
        )
        
        # Make prediction
        prediction = self.model_repository.predict(preprocessed[0])
        
        return prediction

