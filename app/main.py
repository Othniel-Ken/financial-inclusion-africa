import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modele_inclusion_financiere.joblib"
COLUMNS_PATH = BASE_DIR / "models" / "colonnes_entree.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Ordre exact des colonnes attendues par le modèle (fourni par la Personne 2)
COLUMNS = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))

_model = None
_model_mtime = 0.0


def get_model():
    """Recharge le modèle automatiquement s'il a été remplacé (réentraînement)."""
    global _model, _model_mtime
    mtime = MODEL_PATH.stat().st_mtime
    if _model is None or mtime != _model_mtime:
        _model = joblib.load(MODEL_PATH)
        _model_mtime = mtime
    return _model


def age_to_group(age: int) -> str:
    """Tranches d'âge identiques à celles du jeu de données nettoyé."""
    if age <= 24:
        return "16-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


app = FastAPI(title="Prédiction d'inclusion financière")


class Person(BaseModel):
    country: str
    location_type: str
    cellphone_access: str
    household_size: int = Field(ge=1, le=30)
    age_of_respondent: int = Field(ge=16, le=120)
    gender_of_respondent: str
    relationship_with_head: str
    marital_status: str
    education_level: str
    job_type: str
    # Indicateurs construits par la Personne 1 lors du nettoyage. 0 = cas normal.
    flag_age_role: int = Field(default=0, ge=0, le=1)
    flag_age_outlier: int = Field(default=0, ge=0, le=1)
    flag_hh_outlier: int = Field(default=0, ge=0, le=1)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(person: Person):
    row = person.model_dump()
    row["age_group"] = age_to_group(row["age_of_respondent"])
    data = pd.DataFrame([row])[COLUMNS]
    proba = float(get_model().predict_proba(data)[0][1])
    return {
        "has_bank_account": proba >= 0.5,
        "probability": round(proba, 3),
    }
