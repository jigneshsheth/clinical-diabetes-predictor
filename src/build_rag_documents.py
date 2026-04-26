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


def build_patient_summary_paragraph(row: pd.Series) -> str:
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

def build_patient_summary(row: pd.Series) -> str:
    """
    Build a structured, readable patient summary for display in the Streamlit UI.
    Returns a markdown-formatted multi-section string.
    """
    age    = int(row.get("AGE_YEARS", 0))
    gender = str(row.get("GENDER", "Unknown"))
    race   = str(row.get("RACE",   "Unknown"))

    n_cond = int(row.get("CONDITION_COUNT", 0))
    n_med  = int(row.get("MED_COUNT",       0))
    n_proc = int(row.get("PROCEDURE_COUNT", 0))

    enc    = int(row.get("ENCOUNTER_COUNT",        0))
    e_type = int(row.get("UNIQUE_ENCOUNTER_TYPES", 0))

    hba1c  = row.get("HBA1C",       None)
    bp     = row.get("SYSTOLIC_BP", None)
    bmi    = row.get("BMI",         None)
    glu    = row.get("GLUCOSE",     None)

    label  = int(row.get("DIABETES_COMPLICATION", 0))

    # ── Gender display ────────────────────────────────────────────────────────
    gender_symbol = {"M": "♂ Male", "F": "♀ Female"}.get(gender.upper(), gender)

    # ── Outcome badge ─────────────────────────────────────────────────────────
    outcome_text = (
        "Diabetes complication recorded" if label == 1
        else "✓ No diabetes complication recorded"
    )

    # ── Lab values (only include if present and non-zero) ─────────────────────
    lab_lines = []
    if pd.notna(hba1c) and hba1c > 0:
        flag = " elevated" if hba1c >= 7.0 else ""
        lab_lines.append(f"HbA1c:       {hba1c:.1f}%{flag}")
    if pd.notna(bp) and bp > 0:
        flag = " elevated" if bp >= 130 else ""
        lab_lines.append(f"Systolic BP: {bp:.0f} mmHg{flag}")
    if pd.notna(bmi) and bmi > 0:
        flag = " overweight" if bmi >= 25 else ""
        lab_lines.append(f"BMI:         {bmi:.1f}{flag}")
    if pd.notna(glu) and glu > 0:
        flag = " elevated" if glu >= 100 else ""
        lab_lines.append(f"Glucose:     {glu:.1f} mg/dL{flag}")

    lab_block = (
        "\n".join(lab_lines) if lab_lines
        else "No lab values recorded"
    )

    summary = (
        f"    {age} yrs  ·  {gender_symbol}  ·  {race}\n\n"
        f"    Clinical history\n"
        f"    Conditions:  {n_cond}\n"
        f"    Medications: {n_med}\n"
        f"    Procedures:  {n_proc}\n\n"
        f"    Healthcare utilisation\n"
        f"    Encounters:  {enc} across {e_type} encounter types\n\n"
        f"    Lab values\n"
        f"    {lab_block.replace(chr(10), chr(10) + '    ')}\n\n"
        f"    Outcome\n"
        f"    {outcome_text}"
    )
    return summary

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
