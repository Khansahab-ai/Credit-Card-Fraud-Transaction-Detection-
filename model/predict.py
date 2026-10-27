import joblib
import os

MODEL_VERSION = "1.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'fraud_model_with_smotenc.pkl')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'fraud_model_comparison_artifacts.pkl')

if not os.path.exists(MODEL_DIR):
        raise FileExistsError(f'Model not fond {MODEL_DIR}')
elif not os.path.exists(ARTIFACTS_DIR):
    raise FileExistsError(f'Artifacts not fond {ARTIFACTS_DIR}')

model = joblib.load(MODEL_DIR)
artifacts = joblib.load(ARTIFACTS_DIR)

