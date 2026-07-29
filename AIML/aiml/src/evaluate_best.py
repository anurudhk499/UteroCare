import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

from torch.cuda.amp import autocast

from dataset import get_dataloaders
from model import build_model
from config import *

# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD TEST DATA
# ==========================================

_, val_loader, _ = get_dataloaders()

test_loader = val_loader

# ==========================================
# BUILD MODEL
# ==========================================

model = build_model(NUM_CLASSES)

model = model.to(DEVICE)

# ==========================================
# LOAD CHECKPOINT
# ==========================================

print("\nLoading Best Model...\n")

checkpoint = torch.load(
    MODEL_SAVE_PATH,
    map_location=DEVICE
)
print("Checkpoint Accuracy:", checkpoint["accuracy"])
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Best model loaded successfully.")

# ==========================================
# STORAGE
# ==========================================

all_labels = []

all_preds = []

all_probs = []

image_names = []
# ==========================================
# EVALUATE BEST MODEL
# ==========================================

print("\n========================================")
print("Evaluating on Test Set...")
print("========================================\n")

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast():

            outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(probabilities, 1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            predicted.cpu().numpy()
        )

        all_probs.extend(
            probabilities.cpu().numpy()
        )

# ==========================================
# NUMPY
# ==========================================

all_labels = np.array(all_labels)

all_preds = np.array(all_preds)

all_probs = np.array(all_probs)

# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_preds,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="weighted"
)

print("\n========================================")
print("TEST METRICS")
print("========================================")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES
)

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================\n")

print(report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

# ==========================================
# SAVE METRICS
# ==========================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "metrics.txt"
    ),
    "w"
) as f:

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n")

# ==========================================
# SAVE PREDICTIONS CSV
# ==========================================

df = pd.DataFrame({

    "True Label":
        [CLASS_NAMES[i] for i in all_labels],

    "Predicted":
        [CLASS_NAMES[i] for i in all_preds],

    "Confidence":
        np.max(all_probs, axis=1)

})

df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),

    index=False

)
# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8,7))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap=plt.cm.Blues
)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

# ==========================================
# PER CLASS ACCURACY
# ==========================================

class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\n========================================")
print("PER CLASS ACCURACY")
print("========================================")

for i, cls in enumerate(CLASS_NAMES):

    print(
        f"{cls:15s}: {class_accuracy[i]*100:.2f}%"
    )

plt.figure(figsize=(8,5))

plt.bar(
    CLASS_NAMES,
    class_accuracy * 100
)

plt.ylabel("Accuracy (%)")

plt.title("Per-Class Accuracy")

plt.ylim(0,100)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "per_class_accuracy.png"
    ),
    dpi=300
)

plt.close()

# ==========================================
# ROC CURVE
# ==========================================

labels_bin = label_binarize(
    all_labels,
    classes=list(range(NUM_CLASSES))
)

plt.figure(figsize=(8,6))

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        labels_bin[:, i],
        all_probs[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.3f})"
    )

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()

# ==========================================
# SAVE CONFIDENCE SCORES
# ==========================================

confidence_df = pd.DataFrame(
    all_probs,
    columns=CLASS_NAMES
)

confidence_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_probabilities.csv"
    ),
    index=False
)

# ==========================================
# SUMMARY
# ==========================================

print("\n========================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("========================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nGenerated Files:")

print("✔ classification_report.txt")
print("✔ metrics.txt")
print("✔ predictions.csv")
print("✔ class_probabilities.csv")
print("✔ confusion_matrix.png")
print("✔ per_class_accuracy.png")
print("✔ roc_curve.png")