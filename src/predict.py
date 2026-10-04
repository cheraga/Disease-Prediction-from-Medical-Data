import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH


def load_model():

    if not BEST_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Best model not found. "
            "Run training and evaluation first."
        )

    return joblib.load(
        BEST_MODEL_PATH
    )


def predict_disease(patient_data):

    model = load_model()

    patient_df = pd.DataFrame(
        [patient_data]
    )

    prediction = model.predict(
        patient_df
    )[0]

    probability = model.predict_proba(
        patient_df
    )[0][1]

    return prediction, probability


if __name__ == "__main__":

    patient = {

        "age": 55,

        "sex": 1,

        "cp": 1,

        "trestbps": 140,

        "chol": 250,

        "fbs": 0,

        "restecg": 1,

        "thalach": 150,

        "exang": 0,

        "oldpeak": 1.0,

        "slope": 1,

        "ca": 0,

        "thal": 2
    }

    prediction, probability = (
        predict_disease(patient)
    )

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Estimated probability: "
        f"{probability:.2%}"
    )
