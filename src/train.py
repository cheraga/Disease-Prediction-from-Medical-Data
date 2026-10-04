import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.data_loader import load_data, validate_data
from src.preprocessing import (
    split_features_target,
    create_preprocessor,
    split_data
)

from src.config import (
    BEST_MODEL_PATH,
    MODEL_DIR
)


def build_models(preprocessor):

    models = {

        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42
                    )
                )
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ]
        ),

        "gradient_boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        random_state=42
                    )
                )
            ]
        )
    }

    return models


def train_models():

    df = load_data()

    validate_data(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    preprocessor = create_preprocessor(X)

    models = build_models(preprocessor)

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        print(
            f"{name} training completed."
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save all models
    for name, model in trained_models.items():

        path = MODEL_DIR / f"{name}.joblib"

        joblib.dump(
            model,
            path
        )

        print(
            f"Saved: {path}"
        )

    # Save train/test datasets
    return (
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":

    train_models()
