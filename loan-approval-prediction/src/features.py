"""Carga de datos y features para el challenge de Loan Approval Prediction.

Lo saqué a un módulo aparte para no tener el mismo código copiado en el notebook y en
train.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "loan_status"
ID_COL = "id"

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

# loan_grade es categórica pero tiene un orden natural de riesgo creciente (A = mejor, G = peor).
GRADE_ORDER = {g: i for i, g in enumerate("ABCDEFG")}

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]


def load_data(data_dir: str | Path):
    """Lee train/test/sample_submission desde `data_dir`."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige valores imposibles observados en la EDA (p. ej. edades de 100+ años, historial
    crediticio mayor a la edad de la persona), acotándolos a percentiles razonables en vez de
    eliminar filas, para no perder registros de test."""
    df = df.copy()
    df["person_age"] = df["person_age"].clip(upper=80)
    df["person_emp_length"] = df["person_emp_length"].clip(upper=60)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade variables derivadas con valor predictivo sobre el riesgo de impago."""
    df = df.copy()

    df["loan_grade_ordinal"] = df["loan_grade"].map(GRADE_ORDER)

    # Ratios de carga financiera / capacidad de pago.
    df["income_to_loan_ratio"] = df["person_income"] / (df["loan_amnt"] + 1)
    df["credit_history_ratio"] = df["cb_person_cred_hist_length"] / (df["person_age"] + 1)
    df["age_emp_ratio"] = df["person_emp_length"] / (df["person_age"] + 1)
    df["income_per_year_employed"] = df["person_income"] / (df["person_emp_length"] + 1)

    # Transformaciones log para variables con cola larga (ingreso y monto del préstamo).
    df["log_person_income"] = np.log1p(df["person_income"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])

    return df


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo: limpieza + features, listo para entrenar/predecir."""
    df = clean_outliers(df)
    df = engineer_features(df)
    return df


def feature_columns() -> list[str]:
    """Columnas finales usadas por los modelos (excluye id y target)."""
    engineered = [
        "loan_grade_ordinal",
        "income_to_loan_ratio",
        "credit_history_ratio",
        "age_emp_ratio",
        "income_per_year_employed",
        "log_person_income",
        "log_loan_amnt",
    ]
    return NUMERIC_COLS + CATEGORICAL_COLS + engineered
