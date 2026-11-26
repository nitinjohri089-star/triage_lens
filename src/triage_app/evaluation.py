"""Metrics and plots for the binary triage regression models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .model_training import LINEAR_REGRESSION_THRESHOLD
from .preprocessing import TRIAGE_NAMES, TRIAGE_ORDER


def evaluate_models(models, x_test, y_test, output_dir: Path):
    """Save binary classification metrics and confusion matrices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_predictions = {
        name: model.predict(x_test) for name, model in models.items()
    }
    predictions = {
        name: (
            predicted >= LINEAR_REGRESSION_THRESHOLD
        ).astype(int)
        if name == "Linear Regression"
        else predicted.astype(int)
        for name, predicted in raw_predictions.items()
    }

    results = []
    for name, predicted in predictions.items():
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision (Urgent)": precision_score(
                    y_test, predicted, pos_label=1, zero_division=0
                ),
                "Recall (Urgent)": recall_score(
                    y_test, predicted, pos_label=1, zero_division=0
                ),
                "F1 Score (Urgent)": f1_score(
                    y_test, predicted, pos_label=1, zero_division=0
                ),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_metrics.csv", index=False)
    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))

    results_df.set_index("Model").plot(kind="bar", figsize=(12, 6))
    plt.title("Linear vs Logistic Regression")
    plt.xlabel("Model")
    plt.ylabel("Performance Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png")
    plt.close()

    target_names = [TRIAGE_NAMES[label] for label in TRIAGE_ORDER]
    for name, predicted in predictions.items():
        confusion = confusion_matrix(y_test, predicted, labels=TRIAGE_ORDER)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Category")
        plt.ylabel("Actual Category")
        plt.tight_layout()
        filename = name.lower().replace(" ", "_")
        plt.savefig(output_dir / f"{filename}_confusion_matrix.png")
        plt.close()

        print(f"\n{name} classification report:")
        print(
            classification_report(
                y_test,
                predicted,
                labels=TRIAGE_ORDER,
                target_names=target_names,
                zero_division=0,
            )
        )

    return results_df, predictions
