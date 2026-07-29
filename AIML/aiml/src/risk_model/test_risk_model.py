from risk_model_loader import predict_risk

# ==========================================================
# SAMPLE PATIENT
# ==========================================================
patient = {

    "Age": 56,
    "Menopause": "Yes",
    "Heavy_Menstrual_Bleeding": "Moderate",
    "Menstrual_Cramps": "Mild",
    "Pelvic_Pain": "Moderate",
    "Bleeding_Between_Periods": "No",
    "Bleeding_After_Menopause": "Yes",
    "Periods_Longer_Than_7_Days": "No",
    "Pain_During_Intercourse": "No",
    "Frequent_Urination": "No",
    "Constipation": "No",
    "Pelvic_Pressure_or_Fullness": "No",
    "Abdominal_Swelling": "No",
    "Lower_Back_Pain": "No",
    "Abnormal_Vaginal_Discharge": "Yes",
    "Fatigue": "Yes",
    "Diagnosed_Anemia": "No",
    "Difficulty_Conceiving": "No",
    "BMI": 27,
    "Pain_Level": 4,
    "MRI_Confidence": 70,
    "Affected_Area_Pct": 9,
    "Final_Diagnosis": "Endometrial_Cancer"

}

# ==========================================================
# PREDICT
# ==========================================================

risk = predict_risk(patient)

print("=" * 60)
print("UTEROCARE RISK MODEL")
print("=" * 60)

print("\nInput Patient\n")

for key, value in patient.items():
    print(f"{key:30}: {value}")

print("\n" + "=" * 60)

print(f"Predicted Risk Score : {risk:.1f}%")

print("=" * 60)

# ==========================================================
# RISK CATEGORY
# ==========================================================

if risk < 25:

    category = "LOW"

elif risk < 50:

    category = "MODERATE"

elif risk < 75:

    category = "HIGH"

else:

    category = "VERY HIGH"

print(f"Risk Category       : {category}")

print("=" * 60)