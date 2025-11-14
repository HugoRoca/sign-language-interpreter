"""Camera repository interface."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class CameraRepository(ABC):
    """Abstract interface for camera operations."""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the camera.
        
        Returns:
            True if camera started successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera."""
        pass
    
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera.
        
        Returns:
            Frame as numpy array (BGR format), or None if no frame available.
        """
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is opened.
        
        Returns:
            True if camera is opened, False otherwise.
        """
        pass

