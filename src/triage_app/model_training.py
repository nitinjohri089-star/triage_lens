"""Build and train binary triage regression models."""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .feature_engineering import create_preprocessor


LINEAR_REGRESSION_THRESHOLD = 0.5


def train_models(features, target, test_size=0.20, random_state=42):
    """Train linear and logistic regression for the binary triage target."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    models = {
        "Linear Regression": LinearRegression(),
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
