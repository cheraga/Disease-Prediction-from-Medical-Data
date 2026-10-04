import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    MODEL_DIR,
    REPORTS_DIR,
    FIGURES_DIR
)

from src.data_loader import load_data
from src.preprocessing import (
    split_features_target,
    split_data
)


def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\n===== {model_name} ====="
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        FIGURES_DIR /
        f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def main():

    df = load_data()

    X, y = split_features_target(df)

    _, X_test, _, y_test = split_data(
        X,
        y
    )

    results = []

    model_files = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "gradient_boosting.joblib"
    ]

    for filename in model_files:

        model_path = MODEL_DIR / filename

        model = joblib.load(
            model_path
        )

        model_name = filename.replace(
            ".joblib",
            ""
        )

        result = evaluate_model(
            model,
            X_test,
            y_test,
            model_name
        )

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        REPORTS_DIR / "model_results.csv",
        index=False
    )

    print("\nModel comparison:")

    print(results_df)

    best_model_name = results_df.iloc[0]["model"]

    best_model_path = (
        MODEL_DIR /
        f"{best_model_name}.joblib"
    )

    best_model = joblib.load(
        best_model_path
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.joblib"
    )

    print(
        f"\nBest model: {best_model_name}"
    )


if __name__ == "__main__":
    main()
