import pandas as pd

import joblib

# =========================
# LOAD
# =========================

model = joblib.load(
    "fusion_risk_model.pkl"
)

scaler = joblib.load(
    "fusion_scaler.pkl"
)

# =========================
# SAMPLE
# =========================
sample = {

    "age": 58,

    "height_cm": 160,

    "weight_kg": 68,

    "bmi": 26.5,

    "parity": 1,

    "menopausal_status": 1,

    "heavy_bleeding": 1,

    "pelvic_pain": 1,

    "painful_periods": 1,

    "chronic_cramping": 1,

    "frequent_urination": 1,

    "pelvic_pressure": 1,

    "postmenopausal_bleeding": 1,

    "abnormal_discharge": 1,

    "unexplained_weight_loss": 1,

    "fatigue": 1,

    "irregular_cycle": 1,

    "infertility": 0,

    "lower_back_pain": 1,

    "bloating": 1,

    "nausea": 1,

    "dyspareunia": 0,

    "enlarged_uterus": 1,

    "bowel_changes": 1,

    "clots_in_bleeding": 1,

    "pain_between_periods": 1,

    "cycle_length_days": 22,

    "bleeding_duration_days": 10,

    "pain_score_0_10": 9,

    "anemia_diagnosed": 1,

    "uterine_tenderness": 1,

    "mri_adeno_prob": 0.02,

    "mri_fibroid_prob": 0.04,

    "mri_cancer_prob": 0.71,

    "mri_normal_prob": 0.23,

    "affected_area_percent": 24.5
}
# =========================
# DATAFRAME
# =========================

df = pd.DataFrame([sample])

# =========================
# SCALE
# =========================

scaled = scaler.transform(df)

# =========================
# PREDICT
# =========================

risk = model.predict(scaled)[0]

print(
    f"\nFusion Risk Score: {risk:.1f}%"
)