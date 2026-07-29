import torch
import joblib
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms

from model import build_model
from config import *

# -----------------------------
# LOAD MRI MODEL
# -----------------------------

mri_model = build_model(NUM_CLASSES)

mri_model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        weights_only=True
    )
)

mri_model.to(DEVICE)

mri_model.eval()

# -----------------------------
# LOAD SYMPTOM MODEL
# -----------------------------

symptom_model = joblib.load(
    "../models/symptom_model.pkl"
)

label_encoder = joblib.load(
    "../models/symptom_label_encoder.pkl"
)

# -----------------------------
# MRI PREDICTION
# -----------------------------

def predict_mri(image_path):

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = mri_model(tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )[0]

    return probs.cpu().numpy()

# -----------------------------
# SYMPTOM PREDICTION
# -----------------------------

def predict_symptoms(symptom_input):

    df = pd.DataFrame([symptom_input])

    probs = symptom_model.predict_proba(df)[0]

    return probs

# -----------------------------
# FUSION
# -----------------------------

def fusion_prediction(
    image_path,
    symptom_input
):

    mri_probs = predict_mri(image_path)

    symptom_probs = predict_symptoms(
        symptom_input
    )

    # weighted fusion

    final_probs = (
        0.85 * mri_probs
        +
        0.15 * symptom_probs
    )

    predicted_index = np.argmax(
        final_probs
    )

    predicted_class = (
        label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    )

    confidence = final_probs[
        predicted_index
    ]

    return {

        "prediction": predicted_class,

        "confidence": float(confidence),

        "mri_probs": mri_probs.tolist(),

        "symptom_probs": symptom_probs.tolist(),

        "final_probs": final_probs.tolist()
    }

# -----------------------------
# TEST
# -----------------------------

sample_symptoms = {

    "age": 41,
    "height_cm": 158,
    "weight_kg": 74,
    "bmi": 29,

    "parity": 2,

    "menopausal_status": 0,

    "heavy_bleeding": 1,
    "pelvic_pain": 1,
    "painful_periods": 1,
    "chronic_cramping": 1,
    "frequent_urination": 1,
    "pelvic_pressure": 1,

    "postmenopausal_bleeding": 0,

    "abnormal_discharge": 0,

    "unexplained_weight_loss": 0,

    "fatigue": 1,
    "irregular_cycle": 1,
    "infertility": 0,
    "lower_back_pain": 1,

    "bloating": 1,
    "nausea": 0,
    "dyspareunia": 0,

    "enlarged_uterus": 1,
    "bowel_changes": 0,
    "clots_in_bleeding": 1,

    "pain_between_periods": 1,

    "cycle_length_days": 34,
    "bleeding_duration_days": 8,

    "pain_score_0_10": 6,

    "anemia_diagnosed": 1,

    "uterine_tenderness": 1
}
result = fusion_prediction(
    "../fibroid.jpg",
    sample_symptoms
)

print("\nFINAL FUSION RESULT\n")

print(result)