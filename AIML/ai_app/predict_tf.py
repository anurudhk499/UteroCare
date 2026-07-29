import numpy as np

from PIL import Image

from .model_loader_tf import (
    model,
    classes,
    IMG_SIZE
)

# ==========================================
# PREDICT IMAGE
# ==========================================

def predict_image(image_path):

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.asarray(image).astype(np.float32)

    # ---------------------------------------
    # NORMALIZATION (Teachable Machine)
    # ---------------------------------------

    image = (image / 127.5) - 1

    image = np.expand_dims(
        image,
        axis=0
    )

    # ---------------------------------------
    # PREDICTION
    # ---------------------------------------

    predictions = model.predict(image, verbose=0)

    print("\nRAW OUTPUT")
    print(predictions)

    probabilities = predictions[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    # ---------------------------------------
    # ALL PROBABILITIES
    # ---------------------------------------

    probability_dict = {}

    for i, cls in enumerate(classes):

        probability_dict[cls] = round(
            float(probabilities[i]) * 100,
            2
        )

    return {

        "prediction": predicted_class,

        "confidence": round(
            confidence * 100,
            2
        ),

        "probabilities": probability_dict

    }