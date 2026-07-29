import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

fusion_model = joblib.load(
    BASE_DIR / "aiml/fusion_risk_model.pkl"
)

fusion_scaler = joblib.load(
    BASE_DIR / "aiml/fusion_scaler.pkl"
)