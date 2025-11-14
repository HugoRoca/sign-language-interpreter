"""Example usage of the Sign Language Interpreter."""
from pathlib import Path

from src.infrastructure.data_loaders.file_data_repository import FileDataRepository
from src.infrastructure.models.cnn_model_repository import CNNModelRepository
from src.infrastructure.preprocessing.image_preprocessing_service import ImagePreprocessingService
from src.application.services.training_service import TrainingService
from src.application.services.prediction_service import PredictionService
from config.settings import TRAIN_DATA_DIR, MODEL_SAVE_PATH, TEST_DATA_DIR


def example_train():
    """Example: Train the model."""
    print("Example: Training the model")
    print("=" * 60)
    
    # Initialize repositories
    data_repo = FileDataRepository(data_dir=TRAIN_DATA_DIR)
    model_repo = CNNModelRepository()
    
    # Initialize training service
    training_service = TrainingService(
        data_repository=data_repo,
        model_repository=model_repo
    )
    
    # Train model
    history = training_service.train_model(epochs=5, batch_size=32)
    
    # Save model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_repo.save_model(str(MODEL_SAVE_PATH))
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")


def example_predict():
    """Example: Make a prediction."""
    print("Example: Making a prediction")
    print("=" * 60)
    
    # Check if model exists
    if not MODEL_SAVE_PATH.exists():
        print(f"Error: Model not found at {MODEL_SAVE_PATH}")
        print("Please train the model first.")
        return
    
    # Initialize repositories
    data_repo = FileDataRepository()
    model_repo = CNNModelRepository()
    model_repo.load_model(str(MODEL_SAVE_PATH))
    
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
    
    # Try to predict on a test image (if available)
    test_dir = TEST_DATA_DIR
    if test_dir.exists():
        # Get first test image
        test_images = list(test_dir.glob("**/*.jpg"))
        if test_images:
            test_image = test_images[0]
            print(f"Predicting on: {test_image}")
            
            prediction = prediction_service.predict_from_path(str(test_image))
            
            print(f"\nPredicted Class: {prediction.predicted_class}")
            print(f"Confidence: {prediction.confidence:.2%}")
            print("\nTop 3 Predictions:")
            sorted_probs = sorted(
                prediction.all_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for i, (class_name, prob) in enumerate(sorted_probs, 1):
                print(f"  {i}. {class_name}: {prob:.2%}")
        else:
            print("No test images found.")
    else:
        print(f"Test directory not found: {test_dir}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        example_train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'predict':
        example_predict()
    else:
        print("Usage:")
        print("  python example_usage.py train   - Train the model")
        print("  python example_usage.py predict - Make a prediction")

