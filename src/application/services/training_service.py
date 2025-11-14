"""Training service for model training."""
from typing import Dict, Optional
import numpy as np
from sklearn.model_selection import train_test_split

from src.domain.repositories.data_repository import DataRepository
from src.domain.repositories.model_repository import ModelRepository
from config.settings import VALIDATION_SPLIT, EPOCHS, BATCH_SIZE


class TrainingService:
    """Service for training the sign language model."""
    
    def __init__(
        self,
        data_repository: DataRepository,
        model_repository: ModelRepository
    ):
        """Initialize the training service.
        
        Args:
            data_repository: Repository for loading data.
            model_repository: Repository for model operations.
        """
        self.data_repository = data_repository
        self.model_repository = model_repository
    
    def train_model(
        self,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        validation_split: float = VALIDATION_SPLIT
    ) -> Dict:
        """Train the model using data from the repository.
        
        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.
            
        Returns:
            Training history dictionary.
        """
        print("📂 Cargando datos de entrenamiento...")
        print("   Esto puede tardar unos minutos dependiendo del tamaño del dataset...")
        X, y = self.data_repository.load_training_data()
        
        print(f"\n✅ Datos cargados exitosamente:")
        print(f"   📷 Total de imágenes: {len(X):,}")
        print(f"   🏷️  Número de clases: {len(np.unique(y))}")
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42,
            stratify=y
        )
        
        print(f"\n📊 División de datos:")
        print(f"   🎓 Muestras de entrenamiento: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   ✅ Muestras de validación: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
        
        # Set class names in model repository
        class_names = self.data_repository.get_class_names()
        self.model_repository.set_class_names(class_names)
        
        print(f"\n🔧 Configuración de entrenamiento:")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Épocas: {epochs}")
        print(f"   📐 Tamaño de imagen: {X_train[0].shape}")
        history = self.model_repository.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        return history

