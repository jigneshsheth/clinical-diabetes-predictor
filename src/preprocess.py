"""
preprocess.py
-------------
Cleans individual tables and joins them around the PATIENT id.
Returns one merged DataFrame ready for feature engineering.
"""

import pandas as pd
from config import DIABETES_CODES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    """Return True if `col` exists in df (case-insensitive already handled upstream)."""
    return col in df.columns


def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep useful demographic columns, compute age from BIRTHDATE.
    BIRTHDATE -> age_years (integer).
    """
    df = df.copy()
    if _safe_col(df, "BIRTHDATE"):
        df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"], errors="coerce")
        today = pd.Timestamp.today()
        df["AGE_YEARS"] = ((today - df["BIRTHDATE"]).dt.days / 365.25).astype(int)
    else:
        df["AGE_YEARS"] = 0

    if _safe_col(df, "DEATHDATE"):
        df["IS_DECEASED"] = df["DEATHDATE"].notna().astype(int)
    else:
        df["IS_DECEASED"] = 0

    keep = ["ID", "GENDER", "RACE", "ETHNICITY", "AGE_YEARS", "IS_DECEASED"]
    keep = [c for c in keep if _safe_col(df, c)]
    return df[keep].rename(columns={"ID": "PATIENT"})


def clean_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """Count encounters per patient and extract encounter types."""
    df = df.copy()
    if not _safe_col(df, "PATIENT"):
        return pd.DataFrame(columns=["PATIENT", "ENCOUNTER_COUNT", "UNIQUE_ENCOUNTER_TYPES"])

    agg = df.groupby("PATIENT").agg(
        ENCOUNTER_COUNT=("ID", "count"),
    ).reset_index()

    if _safe_col(df, "ENCOUNTERCLASS"):
        types = df.groupby("PATIENT")["ENCOUNTERCLASS"].nunique().reset_index()
        types.columns = ["PATIENT", "UNIQUE_ENCOUNTER_TYPES"]
        agg = agg.merge(types, on="PATIENT", how="left")
    else:
        agg["UNIQUE_ENCOUNTER_TYPES"] = 0

    return agg


def clean_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Count conditions per patient; build diabetes-complication label."""
    df = df.copy()
    if not _safe_col(df, "PATIENT"):
        return pd.DataFrame(columns=["PATIENT", "CONDITION_COUNT", "DIABETES_COMPLICATION"])

    agg = df.groupby("PATIENT").agg(
        CONDITION_COUNT=("CODE", "count"),
    ).reset_index()

    # Build label: 1 if patient has any diabetes complication code
    if _safe_col(df, "CODE"):
        df["CODE"] = df["CODE"].astype(str)
        comp = df[df["CODE"].isin([str(c) for c in DIABETES_CODES])]
        comp_patients = set(comp["PATIENT"].unique())
        agg["DIABETES_COMPLICATION"] = agg["PATIENT"].isin(comp_patients).astype(int)
    else:
        agg["DIABETES_COMPLICATION"] = 0

    return agg


def clean_medications(df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct medications per patient."""
    df = df.copy()
    if not _safe_col(df, "PATIENT"):
        return pd.DataFrame(columns=["PATIENT", "MED_COUNT"])

    agg = df.groupby("PATIENT").agg(
        MED_COUNT=("CODE", "nunique"),
    ).reset_index()
    return agg


def clean_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key numeric lab values (most recent per patient):
      - HbA1c   (LOINC 4548-4)
      - Systolic BP (LOINC 8480-6)
      - BMI      (LOINC 39156-5)
      - Glucose  (LOINC 2339-0)
    """
    df = df.copy()
    if not _safe_col(df, "PATIENT") or not _safe_col(df, "CODE"):
        return pd.DataFrame(columns=["PATIENT"])

    # Sort by date so we can take the most recent value
    if _safe_col(df, "DATE"):
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.sort_values("DATE")

    # Convert VALUE to numeric
    if _safe_col(df, "VALUE"):
        df["NUMERIC_VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    else:
        df["NUMERIC_VALUE"] = float("nan")

    lab_map = {
        "4548-4":  "HBA1C",
        "8480-6":  "SYSTOLIC_BP",
        "39156-5": "BMI",
        "2339-0":  "GLUCOSE",
    }

    result = None
    for code, col_name in lab_map.items():
        sub = df[df["CODE"].astype(str) == code][["PATIENT", "NUMERIC_VALUE"]]
        sub = sub.dropna(subset=["NUMERIC_VALUE"])
        sub = sub.groupby("PATIENT")["NUMERIC_VALUE"].last().reset_index()
        sub.columns = ["PATIENT", col_name]
        if result is None:
            result = sub
        else:
            result = result.merge(sub, on="PATIENT", how="outer")

    if result is None:
        return pd.DataFrame(columns=["PATIENT"])
    return result


def clean_procedures(df: pd.DataFrame) -> pd.DataFrame:
    """Count procedures per patient."""
    df = df.copy()
    if not _safe_col(df, "PATIENT"):
        return pd.DataFrame(columns=["PATIENT", "PROCEDURE_COUNT"])

    agg = df.groupby("PATIENT").agg(
        PROCEDURE_COUNT=("CODE", "count"),
    ).reset_index()
    return agg


def merge_all(data: dict) -> pd.DataFrame:
    """
    Join all cleaned tables into one patient-level DataFrame.

    Parameters
    ----------
    data : dict of table_name -> raw DataFrame (from load_data.load_all_csv)

    Returns
    -------
    pd.DataFrame with one row per patient and all engineered columns.
    """
    base = clean_patients(data.get("patients", pd.DataFrame()))
    if base.empty:
        raise ValueError("patients.csv is required but was not loaded.")

    joins = []
    if "encounters"   in data: joins.append(clean_encounters(data["encounters"]))
    if "conditions"   in data: joins.append(clean_conditions(data["conditions"]))
    if "medications"  in data: joins.append(clean_medications(data["medications"]))
    if "observations" in data: joins.append(clean_observations(data["observations"]))
    if "procedures"   in data: joins.append(clean_procedures(data["procedures"]))

    merged = base.copy()
    for df in joins:
        if "PATIENT" in df.columns:
            merged = merged.merge(df, on="PATIENT", how="left")

    # Drop any duplicate patient rows created by the merge
    merged = merged.drop_duplicates(subset=["PATIENT"], keep="first")

    # Fill numeric nulls with 0 (a patient who appears in patients.csv
    # but has no records in another table genuinely has count = 0)
    numeric_cols = merged.select_dtypes(include="number").columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    # Clamp DIABETES_COMPLICATION to binary 0/1 (safety net)
    if "DIABETES_COMPLICATION" in merged.columns:
        merged["DIABETES_COMPLICATION"] = merged["DIABETES_COMPLICATION"].clip(0, 1).astype(int)

    print(f"[preprocess] Merged DataFrame: {merged.shape[0]:,} patients × {merged.shape[1]} columns")
    return merged


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from load_data import load_all_csv
    data = load_all_csv()
    df = merge_all(data)
    print(df.head(3))
    print("Columns:", df.columns.tolist())
