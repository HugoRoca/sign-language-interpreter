"""Main camera interface for real-time sign language interpretation."""
import argparse
from pathlib import Path

from src.infrastructure.camera.opencv_camera_repository import OpenCVCameraRepository
from src.infrastructure.models.cnn_model_repository import CNNModelRepository
from src.infrastructure.preprocessing.image_preprocessing_service import ImagePreprocessingService
from src.infrastructure.services.word_formation_service_impl import WordFormationServiceImpl
from src.application.services.camera_service import CameraService
from config.settings import MODEL_SAVE_PATH


def camera_command(args):
    """Handle camera command."""
    print("=" * 60)
    print("Sign Language Interpreter - Camera Mode")
    print("=" * 60)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python -m src.interfaces.cli.main train")
        return
    
    # Initialize repositories
    camera_repo = OpenCVCameraRepository(camera_index=args.camera_index)
    model_repo = CNNModelRepository()
    model_repo.load_model(str(model_path))
    
    # Load class names (we need a data repo for this)
    from src.infrastructure.data_loaders.file_data_repository import FileDataRepository
    data_repo = FileDataRepository()
    class_names = data_repo.get_class_names()
    model_repo.set_class_names(class_names)
    
    # Initialize services
    preprocessing_service = ImagePreprocessingService()
    word_formation_service = WordFormationServiceImpl(
        min_confidence=args.min_confidence,
        stability_threshold=args.stability_threshold,
        space_delay=args.space_delay
    )
    
    # Initialize camera service
    camera_service = CameraService(
        camera_repository=camera_repo,
        model_repository=model_repo,
        preprocessing_service=preprocessing_service,
        word_formation_service=word_formation_service
    )
    
    # Run camera service
    camera_service.run()


def main():
    """Main entry point for camera interface."""
    parser = argparse.ArgumentParser(
        description="Sign Language Interpreter - Real-time camera mode"
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=str(MODEL_SAVE_PATH),
        help=f'Path to the trained model (default: {MODEL_SAVE_PATH})'
    )
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera index to use (default: 0)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.7,
        help='Minimum confidence to accept a letter (default: 0.7)'
    )
    parser.add_argument(
        '--stability-threshold',
        type=int,
        default=10,
        help='Number of consecutive detections needed to add a letter (default: 10)'
    )
    parser.add_argument(
        '--space-delay',
        type=float,
        default=2.0,
        help='Seconds of space detection to add a space (default: 2.0)'
    )
    
    args = parser.parse_args()
    camera_command(args)


if __name__ == '__main__':
    main()

