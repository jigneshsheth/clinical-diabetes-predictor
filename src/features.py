"""
features.py
-----------
Selects the final feature columns and target label.
Encodes categoricals, returns X (features) and y (label).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import PREDICTION_TARGET


# Columns used as model features
NUMERIC_FEATURES = [
    "AGE_YEARS",
    "ENCOUNTER_COUNT",
    "UNIQUE_ENCOUNTER_TYPES",
    "CONDITION_COUNT",
    "MED_COUNT",
    "PROCEDURE_COUNT",
    "HBA1C",
    "SYSTOLIC_BP",
    "BMI",
    "GLUCOSE",
    "IS_DECEASED",
]

CATEGORICAL_FEATURES = ["GENDER", "RACE"]

# Map prediction target name -> column in the merged DataFrame
TARGET_COLUMN_MAP = {
    "diabetes_complication": "DIABETES_COMPLICATION",
    "heart_disease":         "DIABETES_COMPLICATION",   # reuse same column in starter
    "readmission":           "DIABETES_COMPLICATION",   # reuse same column in starter
}


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (feature matrix) and y (label) from the merged DataFrame.

    Parameters
    ----------
    df : patient-level DataFrame from preprocess.merge_all()

    Returns
    -------
    X : pd.DataFrame of features (all numeric)
    y : pd.Series of 0/1 labels
    """
    df = df.copy()

    # ── Target label ─────────────────────────────────────────────────────────
    target_col = TARGET_COLUMN_MAP.get(PREDICTION_TARGET, "DIABETES_COMPLICATION")
    if target_col not in df.columns:
        print(f"[features] WARNING: target column '{target_col}' not found - creating zeros.")
        df[target_col] = 0
    y = df[target_col].clip(0, 1).astype(int)  # ensure binary 0/1

    # ── Numeric features ─────────────────────────────────────────────────────
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    X_num = df[num_cols].copy().fillna(0)

    # ── Categorical features (label-encode) ──────────────────────────────────
    cat_frames = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].fillna("unknown").astype(str))
            cat_frames.append(pd.Series(encoded, name=col + "_ENC", index=df.index))

    # Combine
    if cat_frames:
        X = pd.concat([X_num] + cat_frames, axis=1)
    else:
        X = X_num

    # Keep PATIENT id alongside X for later use (not used by the model)
    if "PATIENT" in df.columns:
        X.index = df["PATIENT"].values

    print(f"[features] Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"[features] Label distribution:\n{y.value_counts().to_string()}")
    return X, y


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature column names."""
    X, _ = build_feature_matrix(df)
    return X.columns.tolist()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from load_data import load_all_csv
    from preprocess import merge_all
    data = load_all_csv()
    merged = merge_all(data)
    X, y = build_feature_matrix(merged)
    print(X.head(3))
    print("y head:", y.head(3).values)
