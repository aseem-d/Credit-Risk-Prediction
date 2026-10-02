"""Load and use the final unweighted XGBoost pipeline."""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_xgb_model.joblib"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
    "MonthlyIncome_missing", "SuspiciousDelinqValue",
]
DELINQUENCY_FEATURES = [
    "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def build_model_input(values: dict) -> pd.DataFrame:
    """Apply the notebook's deterministic feature construction to raw inputs."""
    row = {name: values.get(name) for name in FEATURES[:10]}
    row["MonthlyIncome_missing"] = int(pd.isna(row["MonthlyIncome"]))
    row["SuspiciousDelinqValue"] = int(any(row[name] in (96, 98) for name in DELINQUENCY_FEATURES))
    for name in DELINQUENCY_FEATURES:
        if row[name] in (96, 98):
            row[name] = float("nan")
    return pd.DataFrame([row], columns=FEATURES)


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_probability(values: dict, model=None) -> float:
    """Return estimated probability for a dict of raw user-entered features."""
    model = model or load_model()
    return float(model.predict_proba(build_model_input(values))[0, 1])
