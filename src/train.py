from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils import find_dataset_path, load_csv

ROOT = Path(__file__).resolve().parents[1]

def main():
    df = load_csv(find_dataset_path(ROOT))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn"])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    print("ROC-AUC:", roc_auc_score(yte, p))

    out = ROOT/"reports"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out/"churn_model_logreg.joblib")
    print("Saved:", out/"churn_model_logreg.joblib)

if __name__ == "__main__":
    main()
