fusion_cases = [

# ==========================================================
# AGREEMENT CASES
# ==========================================================

{
    "name": "Agreement - Fibroid",
    "mri": {
        "prediction": "Fibroid",
        "confidence": 92,
        "probabilities": {
            "Adenomyosis": 5,
            "Endometrial_Cancer": 1,
            "Fibroid": 92,
            "Normal": 2
        }
    },
    "symptom": {
        "prediction": "Fibroid",
        "confidence": 95,
        "probabilities": {
            "Adenomyosis": 3,
            "Endometrial_Cancer": 1,
            "Fibroid": 95,
            "Normal": 1
        }
    }
},

{
    "name": "Agreement - Adenomyosis",
    "mri": {
        "prediction": "Adenomyosis",
        "confidence": 88,
        "probabilities": {
            "Adenomyosis": 88,
            "Endometrial_Cancer": 2,
            "Fibroid": 8,
            "Normal": 2
        }
    },
    "symptom": {
        "prediction": "Adenomyosis",
        "confidence": 91,
        "probabilities": {
            "Adenomyosis": 91,
            "Endometrial_Cancer": 1,
            "Fibroid": 6,
            "Normal": 2
        }
    }
},

{
    "name": "Agreement - Cancer",
    "mri": {
        "prediction": "Endometrial_Cancer",
        "confidence": 96,
        "probabilities": {
            "Adenomyosis": 1,
            "Endometrial_Cancer": 96,
            "Fibroid": 2,
            "Normal": 1
        }
    },
    "symptom": {
        "prediction": "Endometrial_Cancer",
        "confidence": 94,
        "probabilities": {
            "Adenomyosis": 2,
            "Endometrial_Cancer": 94,
            "Fibroid": 2,
            "Normal": 2
        }
    }
},

{
    "name": "Agreement - Normal",
    "mri": {
        "prediction": "Normal",
        "confidence": 97,
        "probabilities": {
            "Adenomyosis": 1,
            "Endometrial_Cancer": 1,
            "Fibroid": 1,
            "Normal": 97
        }
    },
    "symptom": {
        "prediction": "Normal",
        "confidence": 95,
        "probabilities": {
            "Adenomyosis": 2,
            "Endometrial_Cancer": 1,
            "Fibroid": 2,
            "Normal": 95
        }
    }
},

# ==========================================================
# FIBROID <-> ADENOMYOSIS
# ==========================================================

{
    "name": "Fibroid MRI | Adenomyosis Symptoms",
    "mri": {
        "prediction": "Fibroid",
        "confidence": 74,
        "probabilities": {
            "Adenomyosis": 22,
            "Endometrial_Cancer": 1,
            "Fibroid": 74,
            "Normal": 3
        }
    },
    "symptom": {
        "prediction": "Adenomyosis",
        "confidence": 90,
        "probabilities": {
            "Adenomyosis": 90,
            "Endometrial_Cancer": 1,
            "Fibroid": 7,
            "Normal": 2
        }
    }
},

{
    "name": "Adenomyosis MRI | Fibroid Symptoms",
    "mri": {
        "prediction": "Adenomyosis",
        "confidence": 79,
        "probabilities": {
            "Adenomyosis": 79,
            "Endometrial_Cancer": 2,
            "Fibroid": 17,
            "Normal": 2
        }
    },
    "symptom": {
        "prediction": "Fibroid",
        "confidence": 88,
        "probabilities": {
            "Adenomyosis": 8,
            "Endometrial_Cancer": 1,
            "Fibroid": 88,
            "Normal": 3
        }
    }
},

# ==========================================================
# CANCER CONFLICTS
# ==========================================================

{
    "name": "Cancer MRI | Fibroid Symptoms",
    "mri": {
        "prediction": "Endometrial_Cancer",
        "confidence": 91,
        "probabilities": {
            "Adenomyosis": 2,
            "Endometrial_Cancer": 91,
            "Fibroid": 5,
            "Normal": 2
        }
    },
    "symptom": {
        "prediction": "Fibroid",
        "confidence": 94,
        "probabilities": {
            "Adenomyosis": 3,
            "Endometrial_Cancer": 1,
            "Fibroid": 94,
            "Normal": 2
        }
    }
},

{
    "name": "Cancer MRI | Normal Symptoms",
    "mri": {
        "prediction": "Endometrial_Cancer",
        "confidence": 95,
        "probabilities": {
            "Adenomyosis": 1,
            "Endometrial_Cancer": 95,
            "Fibroid": 2,
            "Normal": 2
        }
    },
    "symptom": {
        "prediction": "Normal",
        "confidence": 96,
        "probabilities": {
            "Adenomyosis": 1,
            "Endometrial_Cancer": 1,
            "Fibroid": 2,
            "Normal": 96
        }
    }
},

{
    "name": "Normal MRI | Cancer Symptoms",
    "mri": {
        "prediction": "Normal",
        "confidence": 90,
        "probabilities": {
            "Adenomyosis": 2,
            "Endometrial_Cancer": 3,
            "Fibroid": 5,
            "Normal": 90
        }
    },
    "symptom": {
        "prediction": "Endometrial_Cancer",
        "confidence": 92,
        "probabilities": {
            "Adenomyosis": 2,
            "Endometrial_Cancer": 92,
            "Fibroid": 4,
            "Normal": 2
        }
    }
},

{
    "name": "Cancer MRI | Adenomyosis Symptoms",
    "mri": {
        "prediction": "Endometrial_Cancer",
        "confidence": 93,
        "probabilities": {
            "Adenomyosis": 3,
            "Endometrial_Cancer": 93,
            "Fibroid": 2,
            "Normal": 2
        }
    },
    "symptom": {
        "prediction": "Adenomyosis",
        "confidence": 89,
        "probabilities": {
            "Adenomyosis": 89,
            "Endometrial_Cancer": 2,
            "Fibroid": 6,
            "Normal": 3
        }
    }
},

# ==========================================================
# LOW CONFIDENCE MRI
# ==========================================================

{
    "name": "Low Confidence MRI",
    "mri": {
        "prediction": "Fibroid",
        "confidence": 51,
        "probabilities": {
            "Adenomyosis": 40,
            "Endometrial_Cancer": 3,
            "Fibroid": 51,
            "Normal": 6
        }
    },
    "symptom": {
        "prediction": "Fibroid",
        "confidence": 93,
        "probabilities": {
            "Adenomyosis": 4,
            "Endometrial_Cancer": 1,
            "Fibroid": 93,
            "Normal": 2
        }
    }
},

{
    "name": "Both Low Confidence",
    "mri": {
        "prediction": "Fibroid",
        "confidence": 45,
        "probabilities": {
            "Adenomyosis": 43,
            "Endometrial_Cancer": 5,
            "Fibroid": 45,
            "Normal": 7
        }
    },
    "symptom": {
        "prediction": "Adenomyosis",
        "confidence": 50,
        "probabilities": {
            "Adenomyosis": 50,
            "Endometrial_Cancer": 3,
            "Fibroid": 42,
            "Normal": 5
        }
    }
}
]