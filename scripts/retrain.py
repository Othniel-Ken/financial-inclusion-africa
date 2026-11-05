import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train_clean.csv"
MODEL_PATH = ROOT / "models" / "modele_inclusion_financiere.joblib"
COLUMNS_PATH = ROOT / "models" / "colonnes_entree.json"
BACKUP_DIR = ROOT / "models" / "backups"

TARGET = "bank_account"          # 0 / 1 dans train_clean.csv
DROP = ["id_unique", "year"]     # colonnes retirées, comme dans la Partie 2
NUMERIC = ["household_size", "age_of_respondent",
           "flag_age_role", "flag_age_outlier", "flag_hh_outlier"]
FEATURES = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


def build_model(scale_pos_weight):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, n_jobs=1,
    )
    return ImbPipeline([("pretraitement", pre), ("modele", clf)])


def scores(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return average_precision_score(y, p), roc_auc_score(y, p)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + [c for c in DROP if c in df.columns])[FEATURES]
    y = df[TARGET]

    # Même découpage que la Partie 2 (stratifié sur pays + cible, seed 42)
    strat = df["country"].astype(str) + "_" + y.astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=strat)

    ratio = (y_tr == 0).sum() / (y_tr == 1).sum()
    new_model = build_model(ratio).fit(X_tr, y_tr)
    new_ap, new_auc = scores(new_model, X_te, y_te)

    old_ap = -1.0
    old_auc = float("nan")
    if MODEL_PATH.exists():
        old_ap, old_auc = scores(joblib.load(MODEL_PATH), X_te, y_te)

    print(f"Ancien modèle  : AP={old_ap:.4f}  ROC-AUC={old_auc:.4f}")
    print(f"Nouveau modèle : AP={new_ap:.4f}  ROC-AUC={new_auc:.4f}")

    # Critère de choix : Average Precision (celui de la Partie 2)
    if new_ap > old_ap:
        if MODEL_PATH.exists():
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(MODEL_PATH, BACKUP_DIR / f"modele_{stamp}.joblib")
        joblib.dump(new_model, MODEL_PATH)
        print("Nouveau modèle adopté.")
    else:
        print("Ancien modèle conservé.")


if __name__ == "__main__":
    main()
