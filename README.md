# Clinical Diabetes Risk Predictor & Chatbot

A local AI system that combines **predictive ML**, **Retrieval-Augmented Generation (RAG)**, and a **local LLM** to analyse synthetic patient records from Synthea.

---

## What this system does

1. Loads Synthea synthetic EHR data (CSV files)
2. Cleans, joins, and engineers patient-level features
3. Trains an XGBoost classifier to predict diabetes complication risk
4. Generates natural-language summaries for each patient
5. Embeds summaries with `nomic-embed-text` and stores them in ChromaDB
6. For a selected patient: predicts risk, retrieves similar cases, and generates a clinical explanation using a local Mistral model via Ollama
7. Displays everything in a Streamlit web app

---

## Quick start (Mac)

### 1 - Install dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2 - Get Synthea data

Download pre-generated Synthea data from:
https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip

Unzip and place the CSV files in `./data/`:
```
clinical-rag-predictor/data/
    patients.csv
    encounters.csv
    conditions.csv
    medications.csv
    observations.csv
    procedures.csv
    allergies.csv
    careplans.csv
    immunizations.csv
```

### 3 - Set up Ollama models

Make sure Ollama is installed: https://ollama.com

```bash
# Start the Ollama server
ollama serve

# In a second terminal, pull the required models
ollama pull mistral            # LLM for explanations
ollama pull nomic-embed-text   # Embeddings for ChromaDB
```

### 4 - Train the model

```bash
cd src
python train_model.py
```

This will print accuracy, F1, ROC-AUC metrics and save:
- `models/risk_model.joblib`
- `models/scaler.joblib`

### 5 - Build the ChromaDB vector store

```bash
cd src
python vector_store.py
```

This embeds all patient summaries and saves them to `./chroma_db/`.  

### 6 - Launch the app

```bash
# From the project root
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## File structure

```
clinical-rag-predictor/

├── requirements.txt          <- Python dependencies
├── README.md
├── data/                     <- Place Synthea CSV files here
├── models/                   <- Trained model & scaler saved here
├── chroma_db/                <- ChromaDB vector store saved here
├── app.py                    <- Streamlit UI (entry point)  
└── src/
    ├── config.py             <- All paths, settings, and constants
    ├── load_data.py          <- Reads CSV files into DataFrames
    ├── preprocess.py         <- Cleans tables and joins them
    ├── features.py           <- Builds the ML feature matrix + label
    ├── train_model.py        <- Trains XGBoost, saves model, prints metrics
    ├── build_rag_documents.py<- Creates patient narrative summaries
    ├── vector_store.py       <- Embeds summaries and stores in ChromaDB
    ├── retriever.py          <- Retrieves top-k similar patients
    └── llm_explainer.py      <- Calls Ollama LLM to generate explanation
```

---

## Synthea dataset schema mapping

| File | Key columns | Engineered features | RAG summary use |
|------|-------------|---------------------|-----------------|
| patients.csv | ID, BIRTHDATE, GENDER, RACE | AGE_YEARS, IS_DECEASED | Age, gender, race |
| encounters.csv | PATIENT, ENCOUNTERCLASS | ENCOUNTER_COUNT, UNIQUE_ENCOUNTER_TYPES | Utilisation narrative |
| conditions.csv | PATIENT, CODE, DESCRIPTION | CONDITION_COUNT, diabetes label | Comorbidity burden |
| medications.csv | PATIENT, CODE, DESCRIPTION | MED_COUNT | Medication burden |
| observations.csv | PATIENT, CODE, VALUE | HBA1C, SYSTOLIC_BP, BMI, GLUCOSE | Lab values text |
| procedures.csv | PATIENT, CODE | PROCEDURE_COUNT | Procedure burden |
| allergies.csv | PATIENT, CODE | (optional feature) | Allergy mention |
| careplans.csv | PATIENT, CODE | (optional feature) | Care context |
| immunizations.csv | PATIENT, CODE | IMMUNIZATION_COUNT | Preventive care |

---

## Evaluation plan

### ML metrics
- Accuracy, Precision, Recall, F1-score (classification_report)
- ROC-AUC curve
- Feature importance chart

### RAG evaluation (qualitative)
- Do retrieved patients have similar demographics and labs to the query?
- Is the LLM explanation grounded in the retrieved evidence?
- Does the explanation mention specific features driving the prediction?

### Limitations
- Synthea data is entirely synthetic - does not reflect real-world distributions
- Lab values are often sparse or zeroed out for patients with few encounters
- Class imbalance may affect recall for the positive (complication) class
- Ollama LLM may hallucinate - always treat explanations as illustrative only

---

**Title:** A Predictive and Generative Clinical Decision Support System Using Synthetic EHR Data, Retrieval-Augmented Generation, and Local Large Language Models

**Research question:** Can a fully local AI pipeline - combining XGBoost risk prediction, ChromaDB-based patient similarity retrieval, and a locally-deployed LLM - generate interpretable, evidence-grounded clinical explanations for patient risk from synthetic EHR data?

**Methodology:** Synthea CSV data is preprocessed into a patient-level feature matrix. An XGBoost classifier predicts diabetes complication risk. Natural-language patient summaries are embedded using nomic-embed-text and stored in ChromaDB. For each query patient, the top-k most similar historical patients are retrieved and provided as context to a local Mistral LLM, which generates a plain-English clinical explanation.

**Expected results:** The XGBoost model should achieve ROC-AUC > 0.75 on Synthea data given strong lab-value signals. The RAG system should retrieve clinically meaningful similar cases, and the LLM should produce coherent explanations grounded in those cases.

**Limitations:** Synthetic data lacks real-world clinical noise; results are not generalisable to actual patient populations without validation on real EHR data.
