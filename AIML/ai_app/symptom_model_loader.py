import os
import joblib
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "symptom_model_results"
)

symptom_model = joblib.load(
    os.path.join(MODEL_DIR, "best_xgboost.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

print("✓ Symptom Model Loaded")


# ==========================================================
# ENCODERS
# ==========================================================

SEVERITY = {
    "None": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3
}

YES_NO = {
    "No": 0,
    "Yes": 1
}

SPECIAL = {
    "NA": -1,
    "No": 0,
    "Yes": 1
}


# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess_input(patient):

    data = patient.copy()

    severity_cols = [
        "Heavy_Menstrual_Bleeding",
        "Menstrual_Cramps",
        "Pelvic_Pain"
    ]

    for col in severity_cols:
        data[col] = SEVERITY.get(data[col], 0)

    yes_no_cols = [

        "Menopause",

        "Bleeding_Between_Periods",

        "Periods_Longer_Than_7_Days",

        "Pain_During_Intercourse",

        "Frequent_Urination",

        "Constipation",

        "Pelvic_Pressure_or_Fullness",

        "Abdominal_Swelling",

        "Lower_Back_Pain",

        "Abnormal_Vaginal_Discharge",

        "Fatigue",

        "Diagnosed_Anemia"

    ]

    for col in yes_no_cols:
        data[col] = YES_NO.get(data[col], 0)

    data["Bleeding_After_Menopause"] = SPECIAL.get(
        data["Bleeding_After_Menopause"], -1
    )

    data["Difficulty_Conceiving"] = SPECIAL.get(
        data["Difficulty_Conceiving"], -1
    )

    df = pd.DataFrame([data])

    df = df[feature_columns]

    return df


# ==========================================================
# PREDICT
# ==========================================================

def predict_symptom(patient):

    processed = preprocess_input(patient)

    prediction = symptom_model.predict(processed)[0]

    probabilities = symptom_model.predict_proba(processed)[0]

    confidence = round(float(max(probabilities) * 100), 2)

    disease = label_encoder.inverse_transform([prediction])[0]

    return {
        "prediction": disease,
        "confidence": confidence,
        "probabilities": probabilities.tolist()
    }