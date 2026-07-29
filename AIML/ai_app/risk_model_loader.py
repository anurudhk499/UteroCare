import os
import joblib
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "aiml",
    "src",
    "risk_model",
    "models"
)

MODEL_PATH = os.path.join(MODEL_DIR, "best_risk_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
# ==========================================================
# LOAD
# ==========================================================

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

label_encoder = joblib.load(LABEL_ENCODER_PATH)
# ==========================================================
# MAPPINGS
# ==========================================================

YES_NO = {

    "Yes": 1,
    "No": 0,
    "yes": 1,
    "no": 0,

    1: 1,
    0: 0

}

SEVERITY = {

    "None": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,

    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3

}

# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess_input(data):

    data = data.copy()

    severity_columns = [

        "Heavy_Menstrual_Bleeding",
        "Menstrual_Cramps",
        "Pelvic_Pain"

    ]

    binary_columns = [

        "Menopause",
        "Bleeding_Between_Periods",
        "Bleeding_After_Menopause",
        "Periods_Longer_Than_7_Days",
        "Pain_During_Intercourse",
        "Frequent_Urination",
        "Constipation",
        "Pelvic_Pressure_or_Fullness",
        "Abdominal_Swelling",
        "Lower_Back_Pain",
        "Abnormal_Vaginal_Discharge",
        "Fatigue",
        "Diagnosed_Anemia",
        "Difficulty_Conceiving"

    ]

# ==========================================================
# Convert Severity Features
# ==========================================================

    for col in severity_columns:

        if col in data:
            data[col] = SEVERITY.get(data[col], 0)

    # ==========================================================
    # Convert Binary Features
    # ==========================================================

    for col in binary_columns:

        if col in data:
            data[col] = YES_NO.get(data[col], 0)

    data["Final_Diagnosis"] = label_encoder.transform(
        [data["Final_Diagnosis"]]
    )[0]

    df = pd.DataFrame([data])

    df = df[feature_columns]

    return df


# ==========================================================
# PREDICT
# ==========================================================

def predict_risk(patient_data):

    processed = preprocess_input(patient_data)

    risk = model.predict(processed)[0]

    risk = float(round(risk, 1))

    risk = max(0, min(100, risk))

    return round(float(risk), 2)