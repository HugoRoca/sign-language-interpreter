"""Main CLI interface for the sign language interpreter."""
import argparse
from pathlib import Path

from src.infrastructure.data_loaders.file_data_repository import FileDataRepository
from src.infrastructure.models.cnn_model_repository import CNNModelRepository
from src.infrastructure.preprocessing.image_preprocessing_service import ImagePreprocessingService
from src.application.services.training_service import TrainingService
from src.application.services.prediction_service import PredictionService
from config.settings import (
    TRAIN_DATA_DIR, MODEL_SAVE_PATH, MODEL_CHECKPOINT_DIR,
    EPOCHS, BATCH_SIZE
)


def train_command(args):
    """Handle train command."""
    print("=" * 60)
    print("Sign Language Interpreter - Training")
    print("=" * 60)
    
    # Initialize repositories
    data_repo = FileDataRepository(data_dir=Path(args.data_dir))
    model_repo = CNNModelRepository()
    
    # Initialize training service
    training_service = TrainingService(
        data_repository=data_repo,
        model_repository=model_repo
    )
    
    # Train model
    history = training_service.train_model(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_repo.save_model(str(model_path))
    
    print(f"\n💾 Modelo guardado en: {model_path}")
    print("✅ ¡Entrenamiento completado exitosamente!")
    print("\n💡 Ahora puedes usar la cámara con: python3 -m src.interfaces.cli.main camera")


def predict_command(args):
    """Handle predict command."""
    print("=" * 60)
    print("Sign Language Interpreter - Prediction")
    print("=" * 60)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python -m src.interfaces.cli.main train")
        return
    
    # Initialize repositories
    data_repo = FileDataRepository()
    model_repo = CNNModelRepository()
    model_repo.load_model(str(model_path))
    
    # Set class names
    class_names = data_repo.get_class_names()
    model_repo.set_class_names(class_names)
    
    # Initialize preprocessing service
    preprocessing_service = ImagePreprocessingService()
    
    # Initialize prediction service
    prediction_service = PredictionService(
        data_repository=data_repo,
        model_repository=model_repo,
        preprocessing_service=preprocessing_service
    )
    
    # Make prediction
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Predicting on image: {image_path}")
    prediction = prediction_service.predict_from_path(str(image_path))
    
    # Display results
    print("\n" + "=" * 60)
    print("Prediction Results:")
    print("=" * 60)
    print(f"Predicted Class: {prediction.predicted_class}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print("\nTop 5 Predictions:")
    
    sorted_probs = sorted(
        prediction.all_probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for i, (class_name, prob) in enumerate(sorted_probs, 1):
        print(f"  {i}. {class_name}: {prob:.2%}")


def camera_command(args):
    """Handle camera command."""
    # Import here to avoid circular dependencies
    from src.infrastructure.camera.opencv_camera_repository import OpenCVCameraRepository
    from src.infrastructure.services.word_formation_service_impl import WordFormationServiceImpl
    from src.application.services.camera_service import CameraService
    
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
    
    # Load class names
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
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Sign Language Interpreter - Train and predict ASL signs"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--data-dir',
        type=str,
        default=str(TRAIN_DATA_DIR),
        help=f'Path to training data directory (default: {TRAIN_DATA_DIR})'
    )
    train_parser.add_argument(
        '--model-path',
        type=str,
        default=str(MODEL_SAVE_PATH),
        help=f'Path to save the trained model (default: {MODEL_SAVE_PATH})'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for training (default: {BATCH_SIZE})'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict from an image')
    predict_parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file to predict'
    )
    predict_parser.add_argument(
        '--model-path',
        type=str,
        default=str(MODEL_SAVE_PATH),
        help=f'Path to the trained model (default: {MODEL_SAVE_PATH})'
    )
    
    # Camera command
    camera_parser = subparsers.add_parser('camera', help='Start real-time camera interpretation')
    camera_parser.add_argument(
        '--model-path',
        type=str,
        default=str(MODEL_SAVE_PATH),
        help=f'Path to the trained model (default: {MODEL_SAVE_PATH})'
    )
    camera_parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Camera index to use (default: 0)'
    )
    camera_parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.7,
        help='Minimum confidence to accept a letter (default: 0.7)'
    )
    camera_parser.add_argument(
        '--stability-threshold',
        type=int,
        default=10,
        help='Number of consecutive detections needed to add a letter (default: 10)'
    )
    camera_parser.add_argument(
        '--space-delay',
        type=float,
        default=2.0,
        help='Seconds of space detection to add a space (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'camera':
        camera_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

