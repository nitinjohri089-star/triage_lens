"""Run the emergency triage machine-learning project."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .evaluation import evaluate_models
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .model_training import LINEAR_REGRESSION_THRESHOLD, train_models
from .preprocessing import TRIAGE_NAMES, TRIAGE_ORDER, load_and_clean_data


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "Emergency_Triage_Real_Dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def save_data_overview_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Save the dataset distribution plots produced in the notebook."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    axis = sns.countplot(data=data, x="IMMEDR", order=TRIAGE_ORDER)
    axis.set_xticks(range(len(TRIAGE_ORDER)))
    axis.set_xticklabels([TRIAGE_NAMES[label] for label in TRIAGE_ORDER])
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
    print("\nBinary triage category counts (0=Non-Urgent, 1=Urgent):")
    print(data["IMMEDR"].value_counts().reindex(TRIAGE_ORDER, fill_value=0))
    print("\nMissing values:")
    print(data.isnull().sum().to_string())

    data.to_csv(OUTPUT_DIR / "Emergency_Triage_Cleaned_Dataset.csv", index=False)
    save_data_overview_plots(data, OUTPUT_DIR)

    features = data.drop(columns=["IMMEDR"])
    target = data["IMMEDR"]
    models, x_train, x_test, _, y_test = train_models(features, target)
    evaluate_models(models, x_test, y_test, OUTPUT_DIR)

    preprocessor = models["Logistic Regression"].named_steps["preprocessor"]
    imputed_training_features = x_train.copy()
    for transformer_name, columns in (
        ("num", NUMERIC_FEATURES),
        ("cat", CATEGORICAL_FEATURES),
    ):
        imputer = preprocessor.named_transformers_[transformer_name].named_steps[
            "imputer"
        ]
        imputed_values = imputer.transform(x_train[columns])
        imputed_training_features[columns] = imputed_values[:, : len(columns)]

    missing_rows = x_train.isna().any(axis=1)
    print("\nExamples of training rows after filling missing values:")
    print(
        imputed_training_features.loc[missing_rows].head(10).to_string(index=False)
    )
    print("\nMissing values remaining in training features:")
    print(imputed_training_features.isnull().sum().to_string())

    model_path = OUTPUT_DIR / "Emergency_Triage_Models.pkl"
    joblib.dump(models, model_path)

    sample_patient = x_test.iloc[[0]]
    print("\nExample predictions:")
    for name, model in models.items():
        raw_prediction = float(model.predict(sample_patient)[0])
        prediction = (
            int(raw_prediction >= LINEAR_REGRESSION_THRESHOLD)
            if name == "Linear Regression"
            else int(raw_prediction)
        )
        print(f"{name}: {TRIAGE_NAMES[prediction]}")
        if name == "Linear Regression":
            print(f"  Raw score (not a probability): {raw_prediction:.4f}")
        else:
            probabilities = model.predict_proba(sample_patient)[0]
            probability_df = pd.DataFrame(
                [probabilities],
                columns=[
                    TRIAGE_NAMES[int(label)]
                    for label in model.named_steps["classifier"].classes_
                ],
            )
            print("  Class probabilities:")
            print(probability_df.round(4).to_string(index=False))

    print(f"\nSaved trained models: {model_path}")
    print(f"Saved evaluation results and plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
