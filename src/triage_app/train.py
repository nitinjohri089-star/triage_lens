"""Run the emergency triage machine-learning project."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .evaluation import evaluate_models
from .model_training import train_models
from .preprocessing import TRIAGE_ORDER, load_and_clean_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "Emergency_Triage_Real_Dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def save_data_overview_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Save the dataset distribution plots produced in the notebook."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x="IMMEDR", order=TRIAGE_ORDER)
    plt.title("Distribution of Emergency Triage Categories")
    plt.xlabel("Triage Category")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_dir / "triage_category_distribution.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(data=data, x="AGE", bins=30, kde=True)
    plt.title("Patient Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "patient_age_distribution.png")
    plt.close()

    vital_columns = ["PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT"]
    plt.figure(figsize=(12, 6))
    data[vital_columns].boxplot()
    plt.title("Distribution of Initial Vital Signs")
    plt.ylabel("Recorded Values")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "vital_signs_distribution.png")
    plt.close()

    numeric_columns = [
        "AGE",
        "TEMPF",
        "PULSE",
        "RESPR",
        "BPSYS",
        "BPDIAS",
        "POPCT",
        "PAINSCALE",
    ]
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        data[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Correlation Between Numerical Features")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation.png")
    plt.close()


def main() -> None:
    data = load_and_clean_data(DATASET_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cleaned dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    print("\nTriage category counts:")
    print(data["IMMEDR"].value_counts().reindex(TRIAGE_ORDER, fill_value=0))
    print("\nMissing values:")
    print(data.isnull().sum().to_string())

    data.to_csv(OUTPUT_DIR / "Emergency_Triage_Cleaned_Dataset.csv", index=False)
    save_data_overview_plots(data, OUTPUT_DIR)

    features = data.drop(columns=["IMMEDR"])
    target = data["IMMEDR"]
    models, _, x_test, _, y_test = train_models(features, target)
    _, predictions = evaluate_models(models, x_test, y_test, OUTPUT_DIR)

    random_forest = models["Random Forest"]
    model_path = OUTPUT_DIR / "Emergency_Triage_Model.pkl"
    joblib.dump(random_forest, model_path)

    sample_patient = x_test.iloc[[0]]
    prediction = random_forest.predict(sample_patient)[0]
    probabilities = random_forest.predict_proba(sample_patient)[0]
    probability_df = pd.DataFrame(
        [probabilities],
        columns=random_forest.named_steps["classifier"].classes_,
    )

    print(f"\nExample predicted triage category: {prediction}")
    print("Example prediction probabilities:")
    print(probability_df.round(4).to_string(index=False))
    print(f"\nSaved trained model: {model_path}")
    print(f"Saved evaluation results and plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
