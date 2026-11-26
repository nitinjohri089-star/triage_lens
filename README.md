# Emergency Triage Classifier

The project is organized as a `src` package. It includes a FastAPI prediction
service served by Uvicorn and a Streamlit interface that sends patient inputs
to that service.

## Structure

- `src/triage_app/preprocessing.py` — dataset cleaning and binary target mapping.
- `src/triage_app/feature_engineering.py` — imputation, scaling, and encoding.
- `src/triage_app/model_training.py` — Linear Regression and Logistic
  Regression training.
- `src/triage_app/evaluation.py` — binary classification metrics and confusion
  matrices.
- `src/triage_app/train.py` — trains models and writes outputs.
- `src/api.py` — FastAPI health and prediction endpoints.
- `main.py` — convenient model-training entry point.
- `streamlit_app.py` — Streamlit UI; calls the API at `http://127.0.0.1:8000`.

The `src` directory contains backend code only. Streamlit runs from the project
root as a separate frontend and connects to the API over HTTP.

## Install

From the project root, install the dependencies:

```powershell
py -m pip install -r requirements.txt
```

The trained models are saved under `outputs/`. After changing the target or
models, regenerate them from the dataset by running:

```powershell
py main.py
```

## Start the application

Open two terminals in the project root.

Terminal 1 — start the API:

```powershell
py -m uvicorn api:app --app-dir src --reload
```

The API health check is at `http://127.0.0.1:8000/health`, and interactive API
documentation is at `http://127.0.0.1:8000/docs`.

Terminal 2 — start the Streamlit UI:

```powershell
py -m streamlit run streamlit_app.py
```

The UI connects to `http://127.0.0.1:8000` by default. Change the API URL in
the sidebar if the service is running elsewhere.

## Important

The target is binary: Immediate, Emergent, and Urgent map to `1` (Urgent);
Semi-Urgent and Non-Urgent map to `0` (Non-Urgent). Linear Regression outputs a
raw numeric score that the project classifies using a 0.5 threshold; that score
is not a probability. Logistic Regression outputs class probabilities.

This is a machine-learning demonstration, not a medical device. Its output is
not medical advice and must not be used to make patient-care decisions.
