"""FastAPI service for emergency triage predictions."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "outputs" / "Emergency_Triage_Model.pkl"


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


class PredictionResponse(BaseModel):
    """Predicted triage category and model class probabilities."""

    prediction: str
    probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `py main.py` from the project root first."
        )
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Emergency Triage Prediction API",
    description="Predicts a triage category from patient measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

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
    prediction = str(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)[0]
    classes = model.named_steps["classifier"].classes_
    return PredictionResponse(
        prediction=prediction,
        probabilities={
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        },
    )
