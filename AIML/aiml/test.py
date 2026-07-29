import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model import build_model
from config import *

# TEST TRANSFORM

test_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# TEST DATASET

test_dataset = datasets.ImageFolder(
    "../preprocessed_dataset/test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# LOAD MODEL

model = build_model(NUM_CLASSES)

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
         map_location=device
    )
)

model.to(DEVICE)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# RESULTS

print("\nTEST SET EVALUATION\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES
    )
)

print("\nCONFUSION MATRIX\n")

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)