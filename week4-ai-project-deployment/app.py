"""
Week 4 capstone — Employee Attrition Risk API + UI.

Run from this folder:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "attrition_pipeline.joblib"
META_PATH = ROOT / "models" / "metadata.json"

app = FastAPI(
    title="Employee Attrition Risk API",
    description="Week 4 capstone: score an employee for leaving-risk from a serialised sklearn Pipeline.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model file missing. Run train_model.py first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class EmployeeFeatures(BaseModel):
    age: int = Field(..., ge=18, le=70, examples=[29])
    gender: Literal["Male", "Female"]
    department: Literal["Sales", "IT", "HR", "Finance", "Marketing"]
    city: Literal["New York", "Chicago", "Houston", "Phoenix", "Los Angeles"]
    years_experience: int = Field(..., ge=0, le=45, examples=[3])
    monthly_salary: float = Field(..., ge=10000, le=250000, examples=[38000])
    performance_score: int = Field(..., ge=1, le=5, examples=[2])


class Prediction(BaseModel):
    attrition: Literal["Yes", "No"]
    probability_leave: float
    risk_level: Literal["Low", "Medium", "High"]
    message: str


def risk_band(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.55:
        return "Medium"
    return "High"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/metadata")
def metadata():
    if not META_PATH.exists():
        raise HTTPException(status_code=404, detail="metadata.json not found")
    import json

    return JSONResponse(json.loads(META_PATH.read_text()))


@app.post("/predict", response_model=Prediction)
def predict(payload: EmployeeFeatures):
    model = get_model()
    row = pd.DataFrame([payload.model_dump()])
    try:
        proba = float(model.predict_proba(row)[0, 1])
        label = int(model.predict(row)[0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    attrition = "Yes" if label == 1 else "No"
    level = risk_band(proba)
    if level == "High":
        msg = "High leaving-risk. Consider a retention conversation and a workload / compensation review."
    elif level == "Medium":
        msg = "Moderate leaving-risk. Monitor engagement and performance trend over the next cycle."
    else:
        msg = "Low leaving-risk at the default threshold. Continue routine check-ins."
    return Prediction(
        attrition=attrition,
        probability_leave=round(proba, 4),
        risk_level=level,  # type: ignore[arg-type]
        message=msg,
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((ROOT / "templates" / "index.html").read_text(encoding="utf-8"))


@app.get("/presentation", response_class=HTMLResponse)
def presentation():
    return HTMLResponse((ROOT / "templates" / "presentation.html").read_text(encoding="utf-8"))
