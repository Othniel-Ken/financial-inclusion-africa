import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(42)
n = 1000

choices = {
    "country": ["Kenya", "Rwanda", "Tanzania", "Uganda"],
    "location_type": ["Rural", "Urban"],
    "cellphone_access": ["Yes", "No"],
    "gender_of_respondent": ["Male", "Female"],
    "relationship_with_head": ["Head of Household", "Spouse", "Child",
                               "Parent", "Other relative", "Other non-relatives"],
    "marital_status": ["Married/Living together", "Single/Never Married",
                       "Widowed", "Divorced/Seperated", "Dont know"],
    "education_level": ["No formal education", "Primary education",
                        "Secondary education", "Vocational/Specialised training",
                        "Tertiary education", "Other/Dont know/RTA"],
    "job_type": ["Self employed", "Government Dependent",
                 "Formally employed Private", "Informally employed",
                 "Formally employed Government", "Farming and Fishing",
                 "Remittance Dependent", "Other Income",
                 "Dont Know/Refuse to answer", "No Income"],
}

df = pd.DataFrame({col: rng.choice(vals, n) for col, vals in choices.items()})
df["household_size"] = rng.integers(1, 10, n)
df["age_of_respondent"] = rng.integers(16, 90, n)
y = rng.integers(0, 2, n)

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), list(choices.keys()))],
    remainder="passthrough",
)
model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
])
model.fit(df, y)

joblib.dump(model, "models/model.joblib")
print("Modèle provisoire sauvegardé : models/model.joblib")
df["bank_account"] = np.where(y == 1, "Yes", "No")
df.to_csv("data/train.csv", index=False)