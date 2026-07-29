from pathlib import Path
import torch

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "processed_dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

OUTPUT_DIR = BASE_DIR / "outputs"

LOG_DIR = BASE_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# DATASET
# ==========================================================

NUM_CLASSES = 4

CLASS_NAMES = [
    "Adenomyosis",
    "Endometrial_Cancer",
    "Fibroid",
    "Normal"
]

IMAGE_SIZE = 224

# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 16

NUM_WORKERS = 0

EPOCHS = 40

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "tf_efficientnetv2_s"

PRETRAINED = True

DROPOUT = 0.3

# ==========================================================
# EARLY STOPPING
# ==========================================================

PATIENCE = 8

MIN_DELTA = 0.001

# ==========================================================
# LABEL SMOOTHING
# ==========================================================

LABEL_SMOOTHING = 0.1

# ==========================================================
# MIXED PRECISION
# ==========================================================

USE_AMP = True

# ==========================================================
# SAVE PATHS
# ==========================================================

BEST_MODEL = MODEL_DIR / "best_model.pth"

LAST_MODEL = MODEL_DIR / "last_model.pth"

TRAIN_HISTORY = OUTPUT_DIR / "training_history.csv"