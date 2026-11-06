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

# Ordre exact des variables attendues par le pipeline d'entraînement.
COLUMNS = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))

_model = None
_model_mtime = 0.0


def get_model():
    """Charge le modèle au premier appel et le recharge après un réentraînement."""
    global _model, _model_mtime

    mtime = MODEL_PATH.stat().st_mtime
    if _model is None or mtime != _model_mtime:
        _model = joblib.load(MODEL_PATH)
        _model_mtime = mtime

    return _model


def age_to_group(age: int) -> str:
    """Reproduit les intervalles pd.cut définis dans le notebook EDA."""
    if 16 <= age <= 24:
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


app = FastAPI(
    title="Financial Inclusion in Africa",
    description=(
        "API de prédiction de la possession d'un compte bancaire "
        "à partir de caractéristiques socio-démographiques."
    ),
    version="1.0.0",
)


class Person(BaseModel):
    country: str
    location_type: str
    cellphone_access: str
    household_size: int = Field(ge=1, le=30)
    age_of_respondent: int = Field(ge=16, le=100)
    gender_of_respondent: str
    relationship_with_head: str
    marital_status: str
    education_level: str
    job_type: str


@app.get("/")
def home():
    """Affiche le formulaire web."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Vérifie que l'API répond."""
    return {"status": "ok"}


@app.post("/predict")
def predict(person: Person):
    """Prépare les variables dérivées puis retourne la probabilité prédite."""
    row = person.model_dump()

    age = row["age_of_respondent"]
    household_size = row["household_size"]
    relationship = row["relationship_with_head"]

    # Variables construites dans le notebook EDA : mêmes règles que le nettoyage.
    row["age_group"] = age_to_group(age)
    row["flag_age_role"] = int(
        relationship in ["Head of Household", "Spouse"] and age < 18
    )
    row["flag_age_outlier"] = int(age > 83)
    row["flag_hh_outlier"] = int(household_size > 9)

    # Le modèle reçoit les colonnes dans le même ordre que pendant l'entraînement.
    data = pd.DataFrame([row])[COLUMNS]

    probability = float(get_model().predict_proba(data)[0][1])

    return {
        "has_bank_account": probability >= 0.5,
        "probability": round(probability, 3),
    }
