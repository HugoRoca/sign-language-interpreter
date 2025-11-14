"""Sign image entity."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class SignImage:
    """Represents a sign language image."""
    image_data: np.ndarray
    label: str
    file_path: Optional[Path] = None
    
    def __post_init__(self):
        """Validate image data."""
        if self.image_data is None or len(self.image_data.shape) != 3:
            raise ValueError("Image data must be a 3D numpy array (height, width, channels)")
        
        if not self.label:
            raise ValueError("Label cannot be empty")

