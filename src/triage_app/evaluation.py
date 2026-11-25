"""Metrics and plots for trained triage classifiers."""

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

from .preprocessing import TRIAGE_ORDER


def evaluate_models(models, x_test, y_test, output_dir: Path):
    """Save metrics, diagnostic plots, and random-forest feature importance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {name: model.predict(x_test) for name, model in models.items()}

    results = []
    for name, predicted in predictions.items():
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision": precision_score(
                    y_test, predicted, average="macro", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predicted, average="macro", zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predicted, average="macro", zero_division=0
                ),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_metrics.csv", index=False)
    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))

    results_df.set_index("Model").plot(kind="bar", figsize=(12, 6))
    plt.title("Comparison of Machine Learning Algorithms")
    plt.xlabel("Algorithms")
    plt.ylabel("Performance Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png")
    plt.close()

    rf_model = models["Random Forest"]
    rf_predictions = predictions["Random Forest"]
    confusion = confusion_matrix(y_test, rf_predictions, labels=TRIAGE_ORDER)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TRIAGE_ORDER,
        yticklabels=TRIAGE_ORDER,
    )
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig(output_dir / "random_forest_confusion_matrix.png")
    plt.close()

    print("\nRandom Forest classification report:")
    print(
        classification_report(
            y_test,
            rf_predictions,
            labels=TRIAGE_ORDER,
            zero_division=0,
        )
    )

    preprocessor = rf_model.named_steps["preprocessor"]
    classifier = rf_model.named_steps["classifier"]
    importance_df = pd.DataFrame(
        {
            "Feature": preprocessor.get_feature_names_out(),
            "Importance": classifier.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(11, 7))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(output_dir / "random_forest_feature_importance.png")
    plt.close()

    return results_df, predictions
