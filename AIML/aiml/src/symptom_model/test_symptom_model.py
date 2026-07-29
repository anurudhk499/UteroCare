import os
import joblib
import pandas as pd

from patient_cases import patient_cases

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(BASE_DIR, "symptom_model_results")

MODEL_PATH = os.path.join(RESULT_DIR, "best_xgboost.pkl")
LABEL_ENCODER_PATH = os.path.join(RESULT_DIR, "label_encoder.pkl")
FEATURE_COLUMNS_PATH = os.path.join(RESULT_DIR, "feature_columns.pkl")

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("Loading Symptom Model...")
print("=" * 60)

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(LABEL_ENCODER_PATH)

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("✓ Model Loaded Successfully!")

# ==========================================================
# TEST ALL CASES
# ==========================================================

correct = 0

total = len(patient_cases)

wrong_cases = []

print("\n")
print("=" * 70)
print("STARTING CLINICAL VALIDATION")
print("=" * 70)

for idx, case in enumerate(patient_cases, start=1):

    sample = pd.DataFrame([case["data"]])

    sample = sample[feature_columns]

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

    predicted = label_encoder.inverse_transform([prediction])[0]

    confidence = max(probabilities) * 100

    print("\n" + "=" * 70)

    print(f"CASE {idx}")

    print("=" * 70)

    print(f"Case Name          : {case['name']}")
    print(f"Expected Disease   : {case['expected']}")
    print(f"Predicted Disease  : {predicted}")
    print(f"Confidence         : {confidence:.2f}%")

    print("\nProbabilities")

    sorted_probs = sorted(
        zip(label_encoder.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    for disease, prob in sorted_probs:

        print(f"{disease:<22} : {prob*100:6.2f}%")

    if predicted == case["expected"]:

        correct += 1

        print("\n✅ CORRECT")

    else:

        print("\n❌ WRONG")

        wrong_cases.append({
            "case": idx,
            "name": case["name"],
            "expected": case["expected"],
            "predicted": predicted,
            "confidence": confidence
        })

# ==========================================================
# SUMMARY
# ==========================================================

accuracy = (correct / total) * 100

print("\n\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Total Cases     : {total}")
print(f"Correct         : {correct}")
print(f"Wrong           : {total-correct}")
print(f"Accuracy        : {accuracy:.2f}%")

print("\n")

if len(wrong_cases) == 0:

    print("🎉 ALL TEST CASES PASSED!")

else:

    print("=" * 70)
    print("MISCLASSIFIED CASES")
    print("=" * 70)

    for w in wrong_cases:

        print(f"\nCase {w['case']} : {w['name']}")
        print(f"Expected   : {w['expected']}")
        print(f"Predicted  : {w['predicted']}")
        print(f"Confidence : {w['confidence']:.2f}%")

print("\n")
print("=" * 70)
print("TESTING COMPLETED")
print("=" * 70)