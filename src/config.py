"""
config.py
---------
Central configuration for the Clinical RAG Predictor.
Change paths and settings here - no need to edit other files.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# ── Paths ────────────────────────────────────────────────────────────────────
# Root of the project (one level above src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(ROOT_DIR, "data")       # Synthea CSV files go here
MODELS_DIR  = os.path.join(ROOT_DIR, "models")     # Trained models saved here
CHROMA_DIR  = os.path.join(ROOT_DIR, "chroma_db")  # ChromaDB vector store

# Create directories if they don't exist yet
for d in [DATA_DIR, MODELS_DIR, CHROMA_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Synthea CSV filenames ─────────────────────────────────────────────────────
CSV_FILES = {
    "patients":      "patients.csv",
    "encounters":    "encounters.csv",
    "conditions":    "conditions.csv",
    "medications":   "medications.csv",
    "observations":  "observations.csv",
    "procedures":    "procedures.csv",
    "allergies":     "allergies.csv",
    "careplans":     "careplans.csv",
    "immunizations": "immunizations.csv",
}

# ── Prediction target ─────────────────────────────────────────────────────────
# Options: "diabetes_complication", "heart_disease", "readmission"
PREDICTION_TARGET = "diabetes_complication"

# Condition codes used to build the label (SNOMED-CT codes from Synthea)
DIABETES_CODES = [
    "44054006",   # Diabetes mellitus type 2
    "73211009",   # Diabetes mellitus
    "314893005",  # Diabetes with renal manifestation
    "230572002",  # Diabetic neuropathy
    "4855003",    # Diabetic retinopathy
]

# ── ML model ─────────────────────────────────────────────────────────────────
MODEL_FILENAME  = "risk_model.joblib"
SCALER_FILENAME = "scaler.joblib"
TEST_SIZE       = 0.2    # 20 % held out for evaluation
RANDOM_STATE    = 42

# ── ChromaDB / RAG ───────────────────────────────────────────────────────────
CHROMA_COLLECTION = "patient_summaries"
EMBED_MODEL       = "nomic-embed-text"   # Ollama embedding model
TOP_K             = 3                    # Number of similar patients to retrieve

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL = "mistral"   
LLM_TEMPERATURE = 0.3   # Lower = more focused/factual

# ── Risk thresholds ──────────────────────────────────────────────────────────
RISK_LOW    = 0.15   # probability < 30 % -> Low
RISK_MEDIUM = 0.35   # 30–60 % -> Medium, ≥ 60 % -> High
