"""Build and train the triage classification models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .feature_engineering import create_preprocessor


def train_models(features, target, test_size=0.20, random_state=42):
    """Split the data and fit all classifiers used by the notebook."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    }

    fitted_models = {}
    for name, classifier in models.items():
        model = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", classifier),
            ]
        )
        model.fit(x_train, y_train)
        fitted_models[name] = model

    return fitted_models, x_train, x_test, y_train, y_test
