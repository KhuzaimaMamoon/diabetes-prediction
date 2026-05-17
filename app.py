"""
Diabetes Prediction FastAPI Application
MLOps SP26 Assignment 1 - Part 5
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
import numpy as np

# Load model and training columns
model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Class label mapping
CLASS_LABELS = {0: "N - Non-Diabetic", 1: "P - Pre-Diabetic", 2: "Y - Diabetic"}

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes class (N/P/Y) from patient lab values.",
    version="1.0.0",
)


class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: str

    @validator("gender")
    def validate_gender(cls, v):
        v = v.strip().upper()
        if v not in ("M", "F"):
            raise ValueError("Gender must be 'M' or 'F'")
        return v


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(patient: PatientData):
    try:
        # Build input dict
        input_data = {
            "AGE": patient.age,
            "Urea": patient.urea,
            "Cr": patient.cr,
            "HbA1c": patient.hba1c,
            "Chol": patient.chol,
            "TG": patient.tg,
            "HDL": patient.hdl,
            "LDL": patient.ldl,
            "VLDL": patient.vldl,
            "BMI": patient.bmi,
            "Gender": patient.gender,
        }

        # Create DataFrame and one-hot encode gender
        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df, columns=["Gender"])

        # Align columns with training columns
        for col in training_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[training_columns]

        # Predict
        prediction_encoded = model.predict(df)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        prediction_desc = CLASS_LABELS.get(prediction_encoded, prediction_label)

        # Get probability if available
        proba = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(df)[0]
            proba = {
                CLASS_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(probas)
            }

        return {
            "prediction": prediction_label,
            "description": prediction_desc,
            "probabilities": proba,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
