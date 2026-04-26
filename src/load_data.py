"""
load_data.py
------------
Reads Synthea CSV files from the ./data/ folder.
Returns a dictionary of DataFrames, one per file.
Handles missing files gracefully.
"""

import os
import pandas as pd
from config import DATA_DIR, CSV_FILES


def load_all_csv() -> dict[str, pd.DataFrame]:
    """
    Load every Synthea CSV defined in config.CSV_FILES.

    Returns
    -------
    dict mapping table name -> pd.DataFrame
    Missing files are skipped with a warning printed to the console.
    """
    data = {}
    for table_name, filename in CSV_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[load_data] WARNING: {filename} not found - skipping {table_name}")
            continue
        try:
            df = pd.read_csv(filepath, low_memory=False)
            # Standardise column names to uppercase for consistency
            df.columns = [c.strip().upper() for c in df.columns]
            data[table_name] = df
            print(f"[load_data] Loaded {table_name}: {len(df):,} rows × {len(df.columns)} cols")
        except Exception as e:
            print(f"[load_data] ERROR reading {filename}: {e}")
    return data


def get_patient_ids(data: dict) -> list[str]:
    """Return the list of unique patient IDs from patients.csv."""
    if "patients" not in data:
        raise ValueError("patients.csv was not loaded - cannot get patient IDs.")
    return data["patients"]["ID"].dropna().unique().tolist()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_all_csv()
    print("\nLoaded tables:", list(data.keys()))
    ids = get_patient_ids(data)
    print(f"Total patients: {len(ids)}")
    if "patients" in data:
        print(data["patients"].head(3))
