"""Prediction entity."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class Prediction:
    """Represents a prediction result."""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    
    def __post_init__(self):
        """Validate prediction data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if not self.predicted_class:
            raise ValueError("Predicted class cannot be empty")

