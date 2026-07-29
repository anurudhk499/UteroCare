
import torch
import timm
import torch.nn as nn
from pathlib import Path

# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================
# CLASSES
# =========================

classes = [

    "adenomyosis",

    "cancer",

    "fibroid",

    "normal"

]
# =========================
# MODEL
# =========================

model = timm.create_model(

    "tf_efficientnetv2_s",

    pretrained=False

)

# =========================
# CUSTOM CLASSIFIER
# =========================

in_features = model.classifier.in_features

model.classifier = nn.Sequential(

    nn.Linear(in_features, 384),

    nn.BatchNorm1d(384),

    nn.SiLU(),

    nn.Dropout(0.35),

    nn.Linear(384, 128),

    nn.BatchNorm1d(128),

    nn.SiLU(),

    nn.Dropout(0.25),

    nn.Linear(128, 4)

)

# =========================
# MODEL PATH
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR/
    "aiml" /
    "models_v2" /
    "best_model.pth"
)

# =========================
# LOAD WEIGHTS
# =========================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)



# =========================
# FINALIZE
# =========================

model.to(device)

model.eval()


def extract_features(img):

    with torch.no_grad():

        features = model.forward_features(img)

        features = torch.mean(
            features,
            dim=[2,3]
        )

    return features

