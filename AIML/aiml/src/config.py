import torch

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ==========================================
# DATASET
# ==========================================

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_CLASSES = 4

CLASS_NAMES = [
    "adenomyosis",
    "cancer",
    "fibroid",
    "normal"
]

# ==========================================
# TRAINING
# ==========================================

EPOCHS = 40

LEARNING_RATE = 1e-4

FINE_TUNE_LR = 1e-5

WEIGHT_DECAY = 5e-4

LABEL_SMOOTHING = 0.1

PATIENCE = 10

FREEZE_EPOCHS = 5

# ==========================================
# PATHS
# ==========================================

MODEL_SAVE_PATH = "../models_v2/best_model.pth"

OUTPUT_DIR = "../outputs"

# ==========================================
# RANDOM SEED
# ==========================================

SEED = 42