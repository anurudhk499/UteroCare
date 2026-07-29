from fusion_engine import fuse_predictions
from fusion_cases import fusion_cases

print("=" * 80)
print("UTEROCARE FUSION ENGINE TEST")
print("=" * 80)

agreement = 0
fusion = 0
warning = 0

for i, case in enumerate(fusion_cases, start=1):

    print("\n" + "=" * 80)

    print(f"CASE {i}")
    print("=" * 80)

    print("Scenario :", case["name"])

    result = fuse_predictions(
        case["mri"],
        case["symptom"]
    )

    print("\nMRI Prediction")

    print(
        case["mri"]["prediction"],
        f"({case['mri']['confidence']}%)"
    )

    print("\nSymptom Prediction")

    print(
        case["symptom"]["prediction"],
        f"({case['symptom']['confidence']}%)"
    )

    print("\n--------------------------------")

    print("Decision :", result["decision"])

    print("Final Prediction :", result["final_prediction"])

    print("Confidence :", result["confidence"])

    print("\nMessage")

    print(result["message"])

    if "probabilities" in result:

        print("\nFinal Probabilities")

        probs = sorted(

            result["probabilities"].items(),

            key=lambda x: x[1],

            reverse=True

        )

        for disease, score in probs:

            print(f"{disease:<25} {score:.2f}")

    if result["decision"] == "AGREEMENT":
        agreement += 1

    elif result["decision"] == "FUSION":
        fusion += 1

    else:
        warning += 1


print("\n\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print("Agreement Cases :", agreement)

print("Fusion Cases    :", fusion)

print("Warning Cases   :", warning)

print("\nFusion Engine Working Successfully.")