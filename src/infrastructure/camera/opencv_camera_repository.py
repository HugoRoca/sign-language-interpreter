"""OpenCV camera repository implementation."""
from typing import Optional
import cv2
import numpy as np

from src.domain.repositories.camera_repository import CameraRepository


class OpenCVCameraRepository(CameraRepository):
    """Concrete implementation of CameraRepository using OpenCV."""
    
    def __init__(self, camera_index: int = 0):
        """Initialize the camera repository.
        
        Args:
            camera_index: Index of the camera to use (default: 0).
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
    
    def start(self) -> bool:
        """Start the camera.
        
        Returns:
            True if camera started successfully, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return True
            return False
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera.
        
        Returns:
            Frame as numpy array (BGR format), or None if no frame available.
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def is_opened(self) -> bool:
        """Check if camera is opened.
        
        Returns:
            True if camera is opened, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

