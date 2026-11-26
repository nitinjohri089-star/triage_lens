"""FastAPI service for emergency triage predictions."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from triage_app.model_training import LINEAR_REGRESSION_THRESHOLD
from triage_app.preprocessing import TRIAGE_NAMES

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "outputs" / "Emergency_Triage_Models.pkl"
MODEL_NAMES = ("Linear Regression", "Logistic Regression")


class PatientFeatures(BaseModel):
    """Patient measurements accepted by the prediction endpoint."""

    age: float = Field(ge=0, le=120)
    sex: Literal["Female", "Male"]
    temperature_f: float = Field(ge=70, le=115)
    pulse: float = Field(ge=0, le=300)
    respiratory_rate: float = Field(ge=0, le=100)
    systolic_bp: float = Field(ge=0, le=350)
    diastolic_bp: float = Field(ge=0, le=250)
    oxygen_saturation: float = Field(ge=0, le=100)
    pain_scale: float = Field(ge=0, le=10)
    ambulance_arrival: Literal["Yes", "No"]
    model: Literal["Linear Regression", "Logistic Regression"] = (
        "Logistic Regression"
    )


class PredictionResponse(BaseModel):
    """Binary triage prediction and model-specific score details."""

    prediction: str
    model: str
    score: float | None = None
    probabilities: dict[str, float] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `py main.py` from the project root first."
        )
    models = joblib.load(MODEL_PATH)
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError(
            f"Expected both binary regression models in {MODEL_PATH}. "
            "Run `py main.py` to retrain them."
        )
    app.state.models = models
    yield


app = FastAPI(
    title="Emergency Triage Prediction API",
    description="Predicts a triage category from patient measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "models"):
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request) -> PredictionResponse:
    models = getattr(request.app.state, "models", None)
    if models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    model_input = pd.DataFrame(
        [
            {
                "AGE": patient.age,
                "SEX": patient.sex,
                "TEMPF": patient.temperature_f,
                "PULSE": patient.pulse,
                "RESPR": patient.respiratory_rate,
                "BPSYS": patient.systolic_bp,
                "BPDIAS": patient.diastolic_bp,
                "POPCT": patient.oxygen_saturation,
                "PAINSCALE": patient.pain_scale,
                "ARREMS": patient.ambulance_arrival,
            }
        ]
    )
    model = models[patient.model]
    raw_prediction = float(model.predict(model_input)[0])

    if patient.model == "Linear Regression":
        predicted_label = int(raw_prediction >= LINEAR_REGRESSION_THRESHOLD)
        return PredictionResponse(
            prediction=TRIAGE_NAMES[predicted_label],
            model=patient.model,
            score=raw_prediction,
        )

    predicted_label = int(raw_prediction)
    probabilities = model.predict_proba(model_input)[0]
    classes = model.named_steps["classifier"].classes_
    return PredictionResponse(
        prediction=TRIAGE_NAMES[predicted_label],
        model=patient.model,
        probabilities={
            TRIAGE_NAMES[int(label)]: float(probability)
            for label, probability in zip(classes, probabilities)
        },
    )
