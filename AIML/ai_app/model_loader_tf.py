import tensorflow as tf
import numpy as np
from pathlib import Path

# ==========================================
# MODEL PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "aiml" /
    "models" /
    "keras_model.h5"
)

LABELS_PATH = (
    BASE_DIR /
    "aiml" /
    "models" /
    "labels.txt"
)

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading Keras Model...")
print("\nLoading model from:")
print(MODEL_PATH)
print("\nModel exists:", MODEL_PATH.exists())
model = tf.keras.models.load_model(
    MODEL_PATH
)
print(model.input_shape)
print("\n======================")
print("MODEL SUMMARY")
print("======================")
model.summary()
print("✓ Keras model loaded.")

# ==========================================
# LOAD LABELS
# ==========================================

classes = []

with open(LABELS_PATH, "r") as f:

    for line in f:

        line = line.strip()

        if line == "":
            continue

        if " " in line:

            classes.append(
                line.split(" ",1)[1]
            )

        else:

            classes.append(line)

print(classes)

IMG_SIZE = (224,224)