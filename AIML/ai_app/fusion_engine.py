"""
==========================================================
UTEROCARE FUSION ENGINE
==========================================================
"""

OVERLAP_DISEASES = {
    ("Fibroid", "Adenomyosis"),
    ("Adenomyosis", "Fibroid"),
}


def normalize_name(name):
    mapping = {
        "Endometrial_Cancer": "Endometrial Cancer",
        "Cancer": "Endometrial Cancer",
        "Normal": "Normal Uterus",
        "normal": "Normal Uterus",
        "fibroid": "Fibroid",
        "adenomyosis": "Adenomyosis",
    }
    return mapping.get(name, name)


def fuse_predictions(mri_prediction, symptom_prediction, mri_probs):
    """
    Parameters
    ----------
    mri_prediction : str
    symptom_prediction : str
    mri_probs : dict
        Example:
        {
            "Adenomyosis":0.15,
            "Endometrial Cancer":0.70,
            "Fibroid":0.10,
            "Normal Uterus":0.05
        }
    """

    mri_prediction = normalize_name(mri_prediction)
    symptom_prediction = normalize_name(symptom_prediction)

    # Highest MRI probability
    confidence = round(max(mri_probs.values()) * 100, 2)

    # ======================================================
    # CASE 1 : BOTH AGREE
    # ======================================================

    if mri_prediction == symptom_prediction:

        return {
            "prediction": mri_prediction,
            "confidence": confidence,
            "mode": "AGREEMENT"
        }

    # ======================================================
    # CASE 2 : FIBROID ↔ ADENOMYOSIS
    # ======================================================

    if (mri_prediction, symptom_prediction) in OVERLAP_DISEASES:

        fibroid_score = mri_probs.get("Fibroid", 0)
        adeno_score = mri_probs.get("Adenomyosis", 0)

        if fibroid_score >= adeno_score:
            prediction = "Fibroid"
            confidence = round(fibroid_score * 100, 2)
        else:
            prediction = "Adenomyosis"
            confidence = round(adeno_score * 100, 2)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "mode": "WEIGHTED_FUSION"
        }

    # ======================================================
    # CASE 3 : CANCER CONFLICT
    # ======================================================

    if (
        mri_prediction == "Endometrial Cancer"
        or
        symptom_prediction == "Endometrial Cancer"
    ):

        return {
            "prediction": "Endometrial Cancer",
            "confidence": max(
                confidence,
                90.0
            ),
            "mode": "WARNING"
        }

    # ======================================================
    # DEFAULT
    # ======================================================

    return {
        "prediction": mri_prediction,
        "confidence": confidence,
        "mode": "MRI_PRIORITY"
    }