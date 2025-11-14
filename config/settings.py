"""Configuration settings for the sign language interpreter."""
from pathlib import Path
from typing import List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "asl_alphabet_train"
TEST_DATA_DIR = DATA_DIR / "asl_alphabet_test"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "asl_model.keras"
MODEL_CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# Training parameters
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Model architecture
NUM_CLASSES = 29  # A-Z (26) + space + del + nothing
INPUT_SHAPE = (*IMAGE_SIZE, 3)  # RGB images

# ASL classes
ASL_CLASSES: List[str] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'space', 'del', 'nothing'
]

