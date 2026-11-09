from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
