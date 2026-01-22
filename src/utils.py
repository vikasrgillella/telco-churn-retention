from __future__ import annotations
from pathlib import Path
import pandas as pd

DEFAULT_CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def find_dataset_path(project_root: Path) -> Path:
    csv_path = project_root / "data" / "raw" / DEFAULT_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return csv_path

def load_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)
