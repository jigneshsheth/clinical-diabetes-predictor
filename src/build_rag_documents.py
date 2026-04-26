"""
build_rag_documents.py
----------------------
Generates a natural-language text summary for every patient.
These summaries are later embedded and stored in ChromaDB.

Run this after preprocessing:
    cd src && python build_rag_documents.py
"""

import pandas as pd
from load_data  import load_all_csv
from preprocess import merge_all


def build_patient_summary(row: pd.Series) -> str:
    """
    Convert one patient row into a plain-English paragraph.
    This text will be stored in ChromaDB as the RAG document.
    """
    parts = []

    # Demographics
    age    = int(row.get("AGE_YEARS", 0))
    gender = str(row.get("GENDER", "Unknown"))
    race   = str(row.get("RACE", "Unknown"))
    parts.append(f"Patient demographics: {age}-year-old {gender} ({race}).")

    # Comorbidity burden
    n_cond = int(row.get("CONDITION_COUNT", 0))
    n_med  = int(row.get("MED_COUNT", 0))
    n_proc = int(row.get("PROCEDURE_COUNT", 0))
    parts.append(
        f"Clinical history: {n_cond} recorded conditions, "
        f"{n_med} distinct medications, {n_proc} procedures."
    )

    # Encounter utilisation
    enc    = int(row.get("ENCOUNTER_COUNT", 0))
    e_type = int(row.get("UNIQUE_ENCOUNTER_TYPES", 0))
    parts.append(
        f"Healthcare utilisation: {enc} total encounters "
        f"across {e_type} different encounter types."
    )

    # Lab values (only if present)
    hba1c = row.get("HBA1C", None)
    bp    = row.get("SYSTOLIC_BP", None)
    bmi   = row.get("BMI", None)
    glu   = row.get("GLUCOSE", None)

    lab_parts = []
    if pd.notna(hba1c) and hba1c > 0:
        lab_parts.append(f"HbA1c {hba1c:.1f}%")
    if pd.notna(bp) and bp > 0:
        lab_parts.append(f"systolic BP {bp:.0f} mmHg")
    if pd.notna(bmi) and bmi > 0:
        lab_parts.append(f"BMI {bmi:.1f}")
    if pd.notna(glu) and glu > 0:
        lab_parts.append(f"glucose {glu:.1f} mg/dL")
    if lab_parts:
        parts.append("Lab values: " + ", ".join(lab_parts) + ".")

    # Outcome label
    label = int(row.get("DIABETES_COMPLICATION", 0))
    outcome_text = "has a recorded diabetes complication" if label == 1 \
                   else "has no recorded diabetes complication"
    parts.append(f"Outcome: Patient {outcome_text}.")

    return " ".join(parts)


def build_all_summaries(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summaries for all patients.

    Returns
    -------
    pd.DataFrame with columns: PATIENT, SUMMARY
    """
    records = []
    for _, row in merged_df.iterrows():
        patient_id = str(row.get("PATIENT", "unknown"))
        summary    = build_patient_summary(row)
        records.append({"PATIENT": patient_id, "SUMMARY": summary})

    summaries_df = pd.DataFrame(records)
    print(f"[build_rag_documents] Built {len(summaries_df):,} patient summaries.")
    return summaries_df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data      = load_all_csv()
    merged    = merge_all(data)
    summaries = build_all_summaries(merged)
    print("\nExample summary:")
    print(summaries.iloc[0]["SUMMARY"])
