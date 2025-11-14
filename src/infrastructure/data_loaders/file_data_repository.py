"""File-based data repository implementation."""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

from src.domain.repositories.data_repository import DataRepository
from src.domain.entities.sign_image import SignImage
from config.settings import TRAIN_DATA_DIR, IMAGE_SIZE


class FileDataRepository(DataRepository):
    """Concrete implementation of DataRepository using file system."""
    
    def __init__(self, data_dir: Path = TRAIN_DATA_DIR, image_size: Tuple[int, int] = IMAGE_SIZE):
        """Initialize the repository.
        
        Args:
            data_dir: Directory containing training data.
            image_size: Target size for images (width, height).
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self._class_names: List[str] = []
        self._load_class_names()
    
    def _load_class_names(self) -> None:
        """Load class names from directory structure."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        # Get all subdirectories (classes) and sort them
        self._class_names = sorted([
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self._class_names.copy()
    
    def _load_image_file(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image file.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Image as numpy array.
        """
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(self.image_size)
        return np.array(img, dtype=np.float32) / 255.0
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all training data from directory structure.
        
        Returns:
            Tuple of (images, labels) as numpy arrays.
        """
        images = []
        labels = []
        
        # First pass: count total images for progress
        total_images = 0
        for class_name in self._class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                image_files = [
                    f for f in class_dir.iterdir()
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and not f.name.startswith('.')
                ]
                total_images += len(image_files)
        
        loaded_count = 0
        
        # Second pass: load images with progress
        for class_idx, class_name in enumerate(self._class_names):
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                continue
            
            # Load all images from this class directory
            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and not f.name.startswith('.')
            ]
            
            for image_file in image_files:
                try:
                    img_array = self._load_image_file(image_file)
                    images.append(img_array)
                    labels.append(class_idx)
                    loaded_count += 1
                    
                    # Show progress every 100 images or at milestones
                    if loaded_count % 100 == 0 or loaded_count == total_images:
                        progress = (loaded_count / total_images) * 100
                        print(f"   📥 Cargando imágenes: {loaded_count:,}/{total_images:,} ({progress:.1f}%) - Clase: {class_name}", end='\r')
                except Exception as e:
                    print(f"\nWarning: Could not load {image_file}: {e}")
                    continue
        
        print()  # New line after progress
        if not images:
            raise ValueError("No images found in data directory")
        
        return np.array(images), np.array(labels)
    
    def load_image(self, image_path: str) -> SignImage:
        """Load a single image for prediction.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            SignImage entity.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img_array = self._load_image_file(path)
        # Extract label from filename or directory if possible
        label = path.stem
        
        return SignImage(
            image_data=img_array,
            label=label,
            file_path=path
        )

