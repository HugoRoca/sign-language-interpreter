"""Camera service for real-time sign language interpretation."""
import cv2
import numpy as np
from typing import Optional, Tuple

from src.domain.repositories.camera_repository import CameraRepository
from src.domain.repositories.model_repository import ModelRepository
from src.domain.services.preprocessing_service import PreprocessingService
from src.domain.services.word_formation_service import WordFormationService
from src.domain.entities.prediction import Prediction
from config.settings import IMAGE_SIZE


class CameraService:
    """Service for real-time camera-based sign language interpretation."""
    
    def __init__(
        self,
        camera_repository: CameraRepository,
        model_repository: ModelRepository,
        preprocessing_service: PreprocessingService,
        word_formation_service: WordFormationService
    ):
        """Initialize the camera service.
        
        Args:
            camera_repository: Repository for camera operations.
            model_repository: Repository for model operations.
            preprocessing_service: Service for preprocessing images.
            word_formation_service: Service for forming words from letters.
        """
        self.camera_repository = camera_repository
        self.model_repository = model_repository
        self.preprocessing_service = preprocessing_service
        self.word_formation_service = word_formation_service
        
        # ROI (Region of Interest) for hand detection
        self.roi_x = 100
        self.roi_y = 100
        self.roi_width = 300
        self.roi_height = 300
    
    def extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract region of interest from frame.
        
        Args:
            frame: Full camera frame.
            
        Returns:
            Tuple of (ROI image, (x, y, width, height)).
        """
        h, w = frame.shape[:2]
        
        # Ensure ROI is within frame bounds
        x = max(0, min(self.roi_x, w - self.roi_width))
        y = max(0, min(self.roi_y, h - self.roi_height))
        width = min(self.roi_width, w - x)
        height = min(self.roi_height, h - y)
        
        roi = frame[y:y+height, x:x+width]
        return roi, (x, y, width, height)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame for model input.
        
        Args:
            frame: Frame in BGR format.
            
        Returns:
            Preprocessed image as numpy array.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, IMAGE_SIZE)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def predict_frame(self, frame: np.ndarray) -> Optional[Prediction]:
        """Make a prediction on a frame.
        
        Args:
            frame: Frame in BGR format.
            
        Returns:
            Prediction entity or None if prediction fails.
        """
        try:
            # Preprocess frame
            preprocessed = self.preprocess_frame(frame)
            
            # Make prediction
            prediction = self.model_repository.predict(preprocessed)
            
            return prediction
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def draw_roi(self, frame: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Draw ROI rectangle on frame.
        
        Args:
            frame: Frame to draw on.
            roi_coords: (x, y, width, height) of ROI.
            
        Returns:
            Frame with ROI drawn.
        """
        x, y, w, h = roi_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
    
    def draw_prediction(
        self,
        frame: np.ndarray,
        prediction: Optional[Prediction],
        roi_coords: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Draw prediction results on frame.
        
        Args:
            frame: Frame to draw on.
            prediction: Prediction result.
            roi_coords: (x, y, width, height) of ROI.
            
        Returns:
            Frame with prediction drawn.
        """
        x, y, w, h = roi_coords
        
        if prediction:
            # Draw prediction text above ROI
            text = f"{prediction.predicted_class} ({prediction.confidence:.1%})"
            cv2.putText(
                frame,
                text,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return frame
    
    def draw_word_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw current word and word history on frame.
        
        Args:
            frame: Frame to draw on.
            
        Returns:
            Frame with word information drawn.
        """
        # Draw current word
        current_word = self.word_formation_service.get_current_word()
        if current_word:
            cv2.putText(
                frame,
                f"Current: {current_word}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
        
        # Draw word history
        history = self.word_formation_service.get_word_history()
        if history:
            history_text = " ".join(history[-5:])  # Show last 5 words
            cv2.putText(
                frame,
                f"Words: {history_text}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2
            )
        
        return frame
    
    def run(self) -> None:
        """Run the camera service in real-time."""
        print("Starting camera service...")
        print("Press 'q' to quit, 'c' to clear current word, 'r' to reset ROI position")
        
        if not self.camera_repository.start():
            print("Error: Could not start camera")
            return
        
        try:
            while True:
                frame = self.camera_repository.read_frame()
                if frame is None:
                    continue
                
                # Extract ROI
                roi, roi_coords = self.extract_roi(frame)
                
                # Make prediction on ROI
                prediction = self.predict_frame(roi)
                
                # Add letter to word formation service
                if prediction:
                    self.word_formation_service.add_letter(
                        prediction.predicted_class,
                        prediction.confidence
                    )
                
                # Draw ROI
                frame = self.draw_roi(frame, roi_coords)
                
                # Draw prediction
                frame = self.draw_prediction(frame, prediction, roi_coords)
                
                # Draw word information
                frame = self.draw_word_info(frame)
                
                # Draw instructions
                cv2.putText(
                    frame,
                    "Press 'q' to quit, 'c' to clear, 'r' to reset ROI",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (150, 150, 150),
                    1
                )
                
                # Display frame
                cv2.imshow('Sign Language Interpreter', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.word_formation_service.clear_word()
                    print("Current word cleared")
                elif key == ord('r'):
                    # Reset ROI to center
                    h, w = frame.shape[:2]
                    self.roi_x = (w - self.roi_width) // 2
                    self.roi_y = (h - self.roi_height) // 2
                    print("ROI reset to center")
        
        finally:
            self.camera_repository.stop()
            cv2.destroyAllWindows()
            print("Camera service stopped")

