"""
app.py
------
Clinical Diabetes Risk Predictor- 6-tab Streamlit dashboard.

Tabs:
  1.  Risk Predictor + Chatbot  - patient risk prediction and LLM explanation
  2.  Dataset Overview           - Synthea CSV statistics and charts
  3.  Training Data              - feature distributions, correlation, balance
  4.  Model Performance          - accuracy, ROC, confusion matrix, feature importance
  5.  RAG Pipeline Metrics       - ChromaDB stats, similarity distances, summaries
  6.  System Pipeline            - visual architecture diagram

Launch:
    cd clinical-diabetes-risk-predictor
    streamlit run app.py
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ollama

# ── Project imports (graceful fallback if src/ not found) ─────────────────────
try:
    from load_data           import load_all_csv
    from preprocess          import merge_all
    from features            import build_feature_matrix
    from train_model         import load_model, predict_patient
    from build_rag_documents import build_all_summaries, build_patient_summary
    from retriever           import retrieve_similar_patients
    from llm_explainer       import generate_explanation
    from config              import (TOP_K, RISK_LOW, RISK_MEDIUM, LLM_MODEL,
                                     EMBED_MODEL, CHROMA_COLLECTION, CHROMA_DIR,
                                     DATA_DIR, MODELS_DIR)
    SRC_OK = True
except Exception:
    SRC_OK = False
    TOP_K = 3; RISK_LOW = 0.3; RISK_MEDIUM = 0.6; LLM_MODEL = "mistral"
    EMBED_MODEL = "nomic-embed-text"; CHROMA_COLLECTION = "patient_summaries"
    CHROMA_DIR = "./chroma_db"; DATA_DIR = "./data"; MODELS_DIR = "./models"

# ── Colour palette: BLUE and GREY only ────────────────────────────────────────
C1 = "#1F3A5F"   # dark navy
C2 = "#2E6DA4"   # medium blue
C3 = "#5A9FD4"   # mid-light blue
C4 = "#8ABFE8"   # light blue
C5 = "#B8D9F2"   # very light blue
G1 = "#2C2C2C"; G2 = "#5A5A5A"; G3 = "#8C8C8C"; G4 = "#C0C0C0"; G5 = "#E8E8E8"
BLUE_PALETTE  = [C1, C2, C3, C4, C5]
GREY_PALETTE  = [G1, G2, G3, G4, G5]
MIXED_PALETTE = [C1, G2, C2, G3, C3, G4, C4, G5, C5]

def _blue_cmap():
    return mcolors.LinearSegmentedColormap.from_list("bluegrey", [C5, C3, C1])

# ── File paths ─────────────────────────────────────────────────────────────────
DATA_FILES = {
    "patients":     os.path.join(DATA_DIR, "patients.csv"),
    "encounters":   os.path.join(DATA_DIR, "encounters.csv"),
    "conditions":   os.path.join(DATA_DIR, "conditions.csv"),
    "medications":  os.path.join(DATA_DIR, "medications.csv"),
    "observations": os.path.join(DATA_DIR, "observations.csv"),
    "procedures":   os.path.join(DATA_DIR, "procedures.csv"),
}
PROC_TRAINING   = os.path.join("processed", "training_dataset.csv")
MODEL_PKL       = os.path.join(MODELS_DIR, "risk_model.joblib")
METRICS_JSON    = os.path.join(MODELS_DIR, "metrics.json")
FEAT_IMP_CSV    = os.path.join(MODELS_DIR, "feature_importance.csv")
PREDICTIONS_CSV = os.path.join(MODELS_DIR, "predictions.csv")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_csv_safely(path: str, label: str = ""):
    """Load a CSV, return None if missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        return df
    except Exception as e:
        st.warning(f"Could not load {label or path}: {e}")
        return None


def load_training_data():
    """Load training CSV or rebuild from raw CSVs."""
    if os.path.exists(PROC_TRAINING):
        return load_csv_safely(PROC_TRAINING, "training_dataset.csv")
    if SRC_OK:
        try:
            return merge_all(load_all_csv())
        except Exception:
            pass
    return None


def load_model_metrics():
    """Load metrics.json- returns None if missing."""
    if not os.path.exists(METRICS_JSON):
        return None
    try:
        with open(METRICS_JSON) as f:
            return json.load(f)
    except Exception:
        return None


def plot_bar_chart(ax, labels, values, title="", xlabel="", ylabel="Count",
                   horizontal=False, color_list=None):
    """Generic bar/barh chart in blue colour scheme."""
    if color_list is None:
        color_list = [BLUE_PALETTE[i % len(BLUE_PALETTE)] for i in range(len(labels))]
    if horizontal:
        bars = ax.barh(labels, values, color=color_list, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(ylabel, fontsize=9); ax.set_ylabel(xlabel, fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{int(val):,}", va="center", fontsize=8, color=G1)
    else:
        bars = ax.bar(labels, values, color=color_list, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.01,
                    f"{int(val):,}", ha="center", fontsize=8, color=G1)
    ax.set_title(title, fontsize=10, fontweight="bold", color=C1, pad=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(G4); ax.spines["bottom"].set_color(G4)
    ax.set_facecolor("#FAFCFF")


def plot_confusion_matrix(ax, cm, labels=("No comp.", "Complication")):
    """Confusion matrix heatmap in blue/grey."""
    im = ax.imshow(cm, interpolation="nearest", cmap=_blue_cmap())
    ax.set_title("Confusion Matrix", fontsize=10, fontweight="bold", color=C1, pad=8)
    ticks = range(len(labels))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[i, j] > thresh else C1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_roc_curve(ax, fpr, tpr, auc_score):
    """ROC curve in dark blue."""
    ax.plot(fpr, tpr, color=C1, lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0,1],[0,1], color=G3, lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=C3)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title("ROC Curve", fontsize=10, fontweight="bold", color=C1, pad=8)
    ax.legend(fontsize=8); ax.set_facecolor("#FAFCFF")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def plot_precision_recall_curve(ax, precision, recall, avg_prec):
    """Precision-recall curve in medium blue."""
    ax.plot(recall, precision, color=C2, lw=2, label=f"Avg prec = {avg_prec:.3f}")
    ax.fill_between(recall, precision, alpha=0.08, color=C3)
    ax.set_xlabel("Recall", fontsize=9); ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision-Recall Curve", fontsize=10, fontweight="bold", color=C1, pad=8)
    ax.legend(fontsize=8); ax.set_facecolor("#FAFCFF")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def plot_feature_importance(ax, feature_names, importances, top_n=12):
    """Horizontal bar chart of feature importances."""
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names, vals = [p[0] for p in pairs], [p[1] for p in pairs]
    n = len(names)
    colors = [BLUE_PALETTE[min(int(i/n*len(BLUE_PALETTE)), len(BLUE_PALETTE)-1)]
              for i in range(n)]
    ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(v + max(vals)*0.01, i, f"{v:.3f}", va="center", fontsize=8, color=G1)
    ax.set_xlabel("Importance (gain)", fontsize=9)
    ax.set_title(f"Top {n} Feature Importances", fontsize=10,
                 fontweight="bold", color=C1, pad=8)
    ax.tick_params(axis="y", labelsize=8); ax.set_facecolor("#FAFCFF")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def render_pipeline_diagram():
    """Renders a blue/grey HTML pipeline diagram inside Streamlit."""
    steps = [
        ("","Synthea\nCSV Files"),("","Data\nPreprocessing"),
        ("","Feature\nEngineering"),("","XGBoost\nClassifier"),
        ("","Patient\nSummaries"),("","nomic-embed\nEmbeddings"),
        ("","ChromaDB\nStore"),("","Similar\nRetrieval"),
        ("","Ollama LLM\nExplanation"),("","Streamlit\nUI"),
    ]
    box = ("display:inline-flex;flex-direction:column;align-items:center;"
           "justify-content:center;background:#EAF3FB;"
           "border:1.5px solid #5A9FD4;border-radius:8px;"
           "padding:10px 14px;min-width:88px;max-width:108px;"
           "font-size:12px;color:#1F3A5F;font-weight:600;"
           "text-align:center;line-height:1.35;"
           "box-shadow:0 2px 4px rgba(0,0,0,0.07);")
    arr  = ("display:inline-flex;align-items:center;color:#2E6DA4;"
            "font-size:20px;padding:0 3px;margin-top:8px;")
    ico  = "font-size:20px;margin-bottom:4px;"
    parts = ['<div style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:4px;padding:12px 0;">']
    for i,(icon,lbl) in enumerate(steps):
        lbl_html = lbl.replace("\n","<br>")
        parts.append(f'<div style="{box}"><span style="{ico}">{icon}</span>{lbl_html}</div>')
        if i < len(steps)-1:
            parts.append(f'<div style="{arr}">&#8594;</div>')
    parts.append("</div>")
    st.markdown("".join(parts)
        + '<p style="font-size:11px;color:#8C8C8C;margin-top:4px;">→ data flows left to right</p>',
        unsafe_allow_html=True)


def _fig(h=4):
    fig, ax = plt.subplots(figsize=(7, h))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#FAFCFF")
    return fig, ax

def _show(fig):
    st.pyplot(fig, use_container_width=True); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Clinical Diabetes Risk Predictor", page_icon="🏥", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"]{font-size:1.4rem;color:#1F3A5F;}
[data-testid="stMetricLabel"]{font-size:0.8rem;color:#5A5A5A;}
</style>""", unsafe_allow_html=True)
# st.title("Clinical Diabetes Risk Predictor")
# st.caption("Synthea EHR - XGBoost - ChromaDB - Ollama - Streamlit ")

# ── Diabetes logo + app header ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:18px;padding:12px 0 6px 0;">
 
  <!-- Diabetes symbolic logo: blue circle + drop + ribbon -->
  <svg width="68" height="68" viewBox="0 0 68 68" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- Outer ring -->
    <circle cx="34" cy="34" r="32" stroke="#2E6DA4" stroke-width="3" fill="#EAF3FB"/>
    <!-- Blue awareness ribbon loop (left arc) -->
    <path d="M28 18 C22 24 22 36 28 40 C24 46 22 52 26 56"
          stroke="#1F3A5F" stroke-width="2.5" stroke-linecap="round" fill="none"/>
    <!-- Blue awareness ribbon loop (right arc) -->
    <path d="M40 18 C46 24 46 36 40 40 C44 46 46 52 42 56"
          stroke="#1F3A5F" stroke-width="2.5" stroke-linecap="round" fill="none"/>
    <!-- Ribbon cross bar -->
    <path d="M28 40 C31 44 37 44 40 40" stroke="#1F3A5F" stroke-width="2.5" stroke-linecap="round" fill="none"/>
    <!-- Blood drop shape (centre) -->
    <path d="M34 22 C34 22 27 31 27 36 C27 40.4 30.1 44 34 44 C37.9 44 41 40.4 41 36 C41 31 34 22 34 22Z"
          fill="#2E6DA4" opacity="0.85"/>
    <!-- Highlight on drop -->
    <ellipse cx="31" cy="35" rx="2" ry="3.5" fill="white" opacity="0.4" transform="rotate(-15 31 35)"/>
    <!-- Glucose ring accent -->
    <circle cx="34" cy="34" r="32" stroke="#5A9FD4" stroke-width="1" fill="none" opacity="0.4"/>
  </svg>
 
  <div>
    <div style="font-size:1.7rem;font-weight:600;color:#1F3A5F;line-height:1.2;">
      Clinical Diabetes Risk Predictor
    </div>
    <div style="font-size:0.85rem;color:#5A5A5A;margin-top:3px;">
      Synthea EHR &nbsp;-&nbsp; XGBoost &nbsp;-&nbsp; ChromaDB &nbsp;-&nbsp; Ollama &nbsp;-&nbsp; Streamlit &nbsp;-&nbsp; fully local
    </div>
  </div>
</div>
<hr style="border:none;border-top:1.5px solid #B8D9F2;margin:4px 0 16px 0;">
""", unsafe_allow_html=True)
top_k        = TOP_K #st.slider("Similar patients to retrieve", 1, 10, TOP_K)
generate_llm =  True #st.toggle("Enable LLM explanation", value=True)
# with st.sidebar:
#     # st.header("Settings")
#     top_k        = TOP_K #st.slider("Similar patients to retrieve", 1, 10, TOP_K)
#     generate_llm =  True #st.toggle("Enable LLM explanation", value=True)
#     st.divider()
#     st.markdown("""
# """)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Synthea data ...")
def _get_merged():
    if not SRC_OK: return pd.DataFrame()
    try:    return merge_all(load_all_csv())
    except: return pd.DataFrame()

@st.cache_data(show_spinner="Building feature matrix ...")
def _get_features(merged):
    if merged.empty or not SRC_OK: return pd.DataFrame(), pd.Series(dtype=int)
    try:    return build_feature_matrix(merged)
    except: return pd.DataFrame(), pd.Series(dtype=int)

@st.cache_resource(show_spinner="Loading model ...")
def _get_model():
    if not SRC_OK: return None, None
    try:    return load_model()
    except: return None, None

@st.cache_data(show_spinner="Building patient summaries ...")
def _get_summaries(merged):
    if merged.empty or not SRC_OK: return pd.DataFrame(columns=["PATIENT","SUMMARY"])
    try:    return build_all_summaries(merged)
    except: return pd.DataFrame(columns=["PATIENT","SUMMARY"])

merged        = _get_merged()
X, y          = _get_features(merged)
model, scaler = _get_model()
summaries_df  = _get_summaries(merged)
patient_ids   = (merged["PATIENT"].tolist()
                 if not merged.empty and "PATIENT" in merged.columns else [])

PROMPT_LIBRARY = {
    "Patient lookup": [
        ("Full dataset summary",
         "Give me an overview: total patients, average age, gender breakdown, complication rate."),
        ("High-HbA1c patients",
         "Which patients likely have HbA1c above 8%? What do they have in common?"),
        ("Demographics overview",
         "Describe the demographic profile- age range, gender split, highest complication group."),
    ],
    "Risk analysis": [
        ("Explain High risk",
         "Describe a typical High-risk patient. Which features push a patient into High risk?"),
        ("Risk distribution",
         "Summarise the risk category distribution. What percentage are High, Medium, Low?"),
        ("What lowers risk?",
         "For a Medium-risk patient, which single improvement most reduces predicted risk?"),
    ],
    "Explain the AI": [
        ("Why HbA1c is top predictor",
         "Explain HbA1c, its normal range, and why it dominates this model."),
        ("How RAG retrieval works",
         "Explain in plain English how ChromaDB finds similar patients using embeddings."),
        ("RAG vs plain classifier",
         "Compare XGBoost alone vs the full RAG pipeline. What does each add?"),
    ],
    "Assignment prompts": [
        ("Project abstract",
         "Write a 150-word academic abstract covering what was built, methods, and results."),
        ("Limitations",
         "List 5 limitations with a concrete improvement for each."),
        ("Ethical considerations",
         "Discuss 3 ethical concerns for real clinical deployment."),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Risk Predictor",
    "Dataset Overview",
    "Training Data",
    "Model Performance",
    "RAG Metrics",
    "System Pipeline",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1- RISK PREDICTOR + CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([1,1], gap="large")

    with left_col:
        st.subheader("Diabetes Risk Predictor: ")
        if not patient_ids:
            st.warning("No patient data loaded. Place Synthea CSVs in `./data/` and restart.")
        else:
            # selected_id = st.selectbox("Select patient ID", patient_ids, key="t1_pid")
            # pat_df = (merged[merged["PATIENT"]==selected_id]
            #           if "PATIENT" in merged.columns else pd.DataFrame())
            # ── Patient selector: Index + ID ─────────────────────────────────────────
            if merged.empty or "PATIENT" not in merged.columns:
                st.warning("Patient data not loaded or PATIENT column missing.")
                st.stop()

            # Build display labels: "Patient Index: 0 | Patient ID: 9f3a..."
            patient_labels = [
                f"Patient Index: {i}  |  Patient ID: {str(pid)}"
                for i, pid in enumerate(merged["PATIENT"].tolist())
            ]

            selected_label = st.selectbox(
                "Select patient",
                options=patient_labels,
                index=0,
                key="t1_pid",
            )

            # Map label back to the actual row
            selected_index = patient_labels.index(selected_label)
            selected_id    = merged["PATIENT"].iloc[selected_index]
            pat_df         = merged.iloc[[selected_index]]

            if pat_df.empty:
                st.warning("Patient not found.")
            else:
                pat_row = pat_df.iloc[0]
                summary_text = build_patient_summary(pat_row) if SRC_OK else f"Patient {selected_id}"
                st.info(summary_text)    
                with st.expander("Raw feature values"):
                    if selected_id in X.index:
                        st.dataframe(X.loc[selected_id].to_frame("value"), use_container_width=True)

                st.subheader("Risk score")
                prob, category, similar_patients = 0.5, "Medium", []
                if model is not None and scaler is not None and selected_id in X.index:
                    prob, category = predict_patient(X.loc[selected_id], model, scaler)
                    m1,m2,m3 = st.columns(3)
                    with m1: st.metric("Probability", f"{prob:.1%}")
                    with m2: st.metric("Category", category)
                    with m3:
                        tl = int(pat_row.get("DIABETES_COMPLICATION",-1))
                        st.metric("Condition",
                                  "Complication" if tl==1 else "No complication" if tl==0 else "Unknown")
                    st.progress(prob, text=f"Risk: {prob:.1%}")
                else:
                    st.warning("Model not trained. Run `cd src && python train_model.py`")

                st.subheader("Similar patients (RAG)")
                if SRC_OK:
                    try:
                        similar_patients = retrieve_similar_patients(summary_text, top_k=top_k)
                        for i,sp in enumerate(similar_patients,1):
                            with st.expander(f"Patient {i}  -  dist {sp['distance']:.4f}  -  {sp['patient_id']}"):
                                st.write(sp["summary"])
                    except RuntimeError as e:
                        st.warning(str(e))

    with right_col:
        st.subheader("Chatbot:")
        # st.caption(f"Powered by **{LLM_MODEL}** via Ollama- fully local")

        if st.button("Generate LLM explanation", disabled=(not generate_llm),
                     use_container_width=True):
            if SRC_OK and similar_patients:
                with st.spinner("Generating ..."):
                    exp = generate_explanation(summary_text, prob, category, similar_patients)
                st.markdown(exp)
            else:
                st.warning("Enable LLM explanation and ensure ChromaDB is built...")

        st.divider()

        def _ctx():
            if merged.empty: return "Dataset not loaded."
            n    = len(merged)
            comp = int(merged["DIABETES_COMPLICATION"].sum()) if "DIABETES_COMPLICATION" in merged.columns else 0
            age  = f"avg age {merged['AGE_YEARS'].mean():.0f}" if "AGE_YEARS" in merged.columns else ""
            return f"{n} patients, {comp} complications, {age}. Model:XGBoost RAG:ChromaDB LLM:{LLM_MODEL}."

        SYS = "You are a helpful clinical AI assistant. " + _ctx() + " Data is SYNTHETIC Synthea only."

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "chat_context" not in st.session_state:
            st.session_state.chat_context = [
                {"role":"user","content":SYS},
                {"role":"assistant","content":"Ready- ask anything about the project."},
            ]

        with st.expander("Prompt library", expanded=False):
            for cat_name, prompts in PROMPT_LIBRARY.items():
                st.markdown(f"**{cat_name}**")
                cols = st.columns(3)
                for idx,(lbl,ptxt) in enumerate(prompts):
                    with cols[idx%3]:
                        if st.button(lbl, key=f"pl_{cat_name}_{idx}", use_container_width=True):
                            st.session_state.pending_prompt = ptxt; st.rerun()

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        injected    = st.session_state.pop("pending_prompt", None)
        user_input  = st.chat_input("Ask about patients, predictions, or the pipeline ...")
        final_input = user_input or injected

        if final_input:
            st.session_state.chat_messages.append({"role":"user","content":final_input})
            with st.chat_message("user"): st.markdown(final_input)
            st.session_state.chat_context.append({"role":"user","content":final_input})
            with st.chat_message("assistant"):
                with st.spinner(f"{LLM_MODEL} thinking ..."):
                    try:
                        resp  = ollama.chat(model=LLM_MODEL,
                                            messages=st.session_state.chat_context,
                                            options={"temperature":0.4})
                        reply = resp["message"]["content"].strip()
                    except Exception as e:
                        reply = f"Ollama not reachable.\n```\nollama serve\nollama pull {LLM_MODEL}\n```\nError: {e}"
                st.markdown(reply)
            st.session_state.chat_messages.append({"role":"assistant","content":reply})
            st.session_state.chat_context.append({"role":"assistant","content":reply})

        if st.session_state.chat_messages:
            if st.button("🗑️ Clear chat"):
                st.session_state.chat_messages = []
                st.session_state.chat_context  = [{"role":"user","content":SYS},
                    {"role":"assistant","content":"Cleared. Ready."}]
                st.rerun()

    st.divider()
    st.caption("")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2- DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Dataset Overview")
    st.caption("Summary statistics from the Synthea CSV files in `./data/`")

    raw = {k: load_csv_safely(v, k) for k,v in DATA_FILES.items()}

    if not any(v is not None for v in raw.values()):
        st.warning("No CSV files found in `./data/`. Download Synthea data first.")
        st.code("# https://synthetichealth.github.io/synthea-sample-data/")
        st.stop()

    # Record count cards
    st.markdown("#### Record counts")
    cols = st.columns(6)
    for col,(lbl,df) in zip(cols,[
        ("Patients",raw["patients"]),("Encounters",raw["encounters"]),
        ("Conditions",raw["conditions"]),("Medications",raw["medications"]),
        ("Observations",raw["observations"]),("Procedures",raw["procedures"])]):
        with col: st.metric(lbl, f"{len(df):,}" if df is not None else "—")

    st.divider()

    # Age distribution
    if raw["patients"] is not None and "BIRTHDATE" in raw["patients"].columns:
        st.markdown("#### Age distribution")
        pts = raw["patients"].copy()
        pts["BIRTHDATE"] = pd.to_datetime(pts["BIRTHDATE"], errors="coerce")
        pts["AGE"] = ((pd.Timestamp.today() - pts["BIRTHDATE"]).dt.days / 365.25).astype(float)
        pts = pts.dropna(subset=["AGE"])
        fig,ax = _fig(3.5)
        ax.hist(pts["AGE"], bins=20, color=C2, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Age (years)",fontsize=9); ax.set_ylabel("Patients",fontsize=9)
        ax.set_title("Patient Age Distribution",fontsize=10,fontweight="bold",color=C1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        _show(fig)

    col_a, col_b = st.columns(2)

    with col_a:
        if raw["patients"] is not None and "GENDER" in raw["patients"].columns:
            gc = raw["patients"]["GENDER"].value_counts()
            fig,ax = _fig(3.2)
            plot_bar_chart(ax, gc.index.tolist(), gc.values.tolist(),
                title="Gender Distribution", ylabel="Patients", color_list=[C1,C3])
            _show(fig)

    with col_b:
        if raw["encounters"] is not None and "ENCOUNTERCLASS" in raw["encounters"].columns:
            ec = raw["encounters"]["ENCOUNTERCLASS"].value_counts().head(8)
            fig,ax = _fig(3.2)
            plot_bar_chart(ax, ec.index.tolist(), ec.values.tolist(),
                title="Encounter Classes", horizontal=True,
                color_list=[BLUE_PALETTE[i%len(BLUE_PALETTE)] for i in range(len(ec))])
            _show(fig)

    if raw["conditions"] is not None:
        dcol = next((c for c in ["DESCRIPTION","CODE"] if c in raw["conditions"].columns), None)
        if dcol:
            st.markdown("#### Top 10 conditions")
            tc = raw["conditions"][dcol].value_counts().head(10)
            fig,ax = _fig(3.8)
            plot_bar_chart(ax, tc.index.tolist(), tc.values.tolist(),
                title="Top 10 Conditions", horizontal=True,
                color_list=[BLUE_PALETTE[i%len(BLUE_PALETTE)] for i in range(10)])
            _show(fig)

    if raw["medications"] is not None:
        dcol = next((c for c in ["DESCRIPTION","CODE"] if c in raw["medications"].columns), None)
        if dcol:
            st.markdown("#### Top 10 medications")
            tm = raw["medications"][dcol].value_counts().head(10)
            fig,ax = _fig(3.8)
            plot_bar_chart(ax, tm.index.tolist(), tm.values.tolist(),
                title="Top 10 Medications", horizontal=True,
                color_list=[GREY_PALETTE[i%len(GREY_PALETTE)] for i in range(10)])
            _show(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3- TRAINING DATA VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Training Dataset Visualisation")
    train_df = load_training_data()

    if train_df is None or train_df.empty:
        st.warning("Training dataset not found. Run `cd src && python train_model.py` first.")
        st.stop()

    TARGET_COL = "DIABETES_COMPLICATION"
    NUM_FEATS  = [c for c in ["AGE_YEARS","HBA1C","BMI","GLUCOSE","SYSTOLIC_BP",
                               "CONDITION_COUNT","MED_COUNT","ENCOUNTER_COUNT",
                               "PROCEDURE_COUNT"] if c in train_df.columns]

    n_total = len(train_df); n_train = int(n_total*0.8); n_test = n_total-n_train
    m1,m2,m3 = st.columns(3)
    with m1: st.metric("Total patients", f"{n_total:,}")
    with m2: st.metric("Training rows (80%)", f"{n_train:,}")
    with m3: st.metric("Test rows (20%)", f"{n_test:,}")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        if TARGET_COL in train_df.columns:
            vc = train_df[TARGET_COL].value_counts().sort_index()
            fig,ax = _fig(3.2)
            plot_bar_chart(ax,["No complication (0)","Complication (1)"],vc.values.tolist(),
                title="Target Label Distribution",ylabel="Patients",color_list=[C3,C1])
            _show(fig)
            n_pos=int(vc.get(1,0)); n_neg=int(vc.get(0,0))
            ratio = n_neg/n_pos if n_pos>0 else float("inf")
            st.info(f"**Class balance**- Negative: {n_neg} - Positive: {n_pos} - Ratio {ratio:.1f}:1")

    with col_b:
        miss = train_df[NUM_FEATS].isnull().sum(); miss = miss[miss>0]
        if miss.empty:
            st.success("No missing values in numeric features.")
        else:
            fig,ax = _fig(3.2)
            plot_bar_chart(ax, miss.index.tolist(), miss.values.tolist(),
                title="Missing Values per Feature", ylabel="Missing", horizontal=True,
                color_list=[G2]*len(miss))
            _show(fig)

    st.markdown("#### Numeric feature distributions")
    feats_to_show = NUM_FEATS[:6]
    for row_start in range(0, len(feats_to_show), 3):
        row_feats = feats_to_show[row_start:row_start+3]
        cols = st.columns(len(row_feats))
        for col,feat in zip(cols,row_feats):
            with col:
                vals = train_df[feat].replace(0, np.nan).dropna()
                if vals.empty: st.caption(f"{feat}: all zeros"); continue
                fig,ax = _fig(2.6)
                ax.hist(vals, bins=15, color=C2, edgecolor="white", linewidth=0.3)
                ax.set_title(feat, fontsize=9, fontweight="bold", color=C1)
                ax.set_ylabel("Count", fontsize=8); ax.tick_params(labelsize=7)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                _show(fig)

    if len(NUM_FEATS) >= 3:
        st.markdown("#### Feature correlation heatmap")
        corr = train_df[NUM_FEATS].corr()
        fig,ax = plt.subplots(figsize=(8,6)); fig.patch.set_facecolor("white")
        im = ax.imshow(corr.values, cmap=_blue_cmap(), vmin=-1, vmax=1)
        ax.set_xticks(range(len(NUM_FEATS))); ax.set_xticklabels(NUM_FEATS,rotation=45,ha="right",fontsize=8)
        ax.set_yticks(range(len(NUM_FEATS))); ax.set_yticklabels(NUM_FEATS,fontsize=8)
        for i in range(len(NUM_FEATS)):
            for j in range(len(NUM_FEATS)):
                v = corr.values[i,j]
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=7,
                        color="white" if abs(v)>0.5 else C1)
        plt.colorbar(im,ax=ax,fraction=0.035,pad=0.03)
        ax.set_title("Pearson Correlation Matrix",fontsize=10,fontweight="bold",color=C1,pad=8)
        plt.tight_layout(); _show(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4- MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Performance")
    st.caption("XGBoost classifier evaluation on the held-out 20% test split")

    metrics  = load_model_metrics()
    feat_imp = load_csv_safely(FEAT_IMP_CSV, "feature_importance.csv")
    preds_df = load_csv_safely(PREDICTIONS_CSV, "predictions.csv")

    if metrics is None and feat_imp is None and preds_df is None and model is None:
        st.warning("No model files found. Train first:\n```\ncd src\npython train_model.py\n```")
        st.stop()

    # Metric cards
    st.markdown("#### Evaluation metrics")
    DEFAULT = {"accuracy":None,"precision":None,"recall":None,"f1":None,"roc_auc":None}
    m = {**DEFAULT, **(metrics or {})}

    # Live recompute if no metrics.json
    if metrics is None and model is not None and not X.empty:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (accuracy_score,precision_score,recall_score,
                                         f1_score,roc_auc_score)
            Xn,yn = X.values, y.values
            _,Xte,_,yte = train_test_split(Xn,yn,test_size=0.2,random_state=42,
                                            stratify=yn if yn.sum()>5 else None)
            ypred = model.predict(scaler.transform(Xte))
            yprob = model.predict_proba(scaler.transform(Xte))[:,1]
            m = {
                "accuracy":  round(accuracy_score(yte,ypred),4),
                "precision": round(precision_score(yte,ypred,zero_division=0),4),
                "recall":    round(recall_score(yte,ypred,zero_division=0),4),
                "f1":        round(f1_score(yte,ypred,zero_division=0),4),
                "roc_auc":   round(roc_auc_score(yte,yprob) if yte.sum()>0 else 0,4),
            }
            st.info("Metrics computed live from loaded model.")
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(k,lbl) in zip([c1,c2,c3,c4,c5],[
        ("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),
        ("f1","F1 Score"),("roc_auc","ROC-AUC")]):
        with col:
            v = m.get(k)
            st.metric(lbl, f"{v:.4f}" if v is not None else "—")
    st.divider()

    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        cm_drawn = False
        if preds_df is not None and {"Y_TRUE","Y_PRED"}.issubset(preds_df.columns):
            from sklearn.metrics import confusion_matrix as sk_cm
            cm = sk_cm(preds_df["Y_TRUE"], preds_df["Y_PRED"])
            fig,ax = _fig(3.8); plot_confusion_matrix(ax,cm); _show(fig); cm_drawn=True
        if not cm_drawn and model is not None and not X.empty:
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import confusion_matrix as sk_cm
                Xn,yn = X.values,y.values
                _,Xte,_,yte = train_test_split(Xn,yn,test_size=0.2,random_state=42,
                                                stratify=yn if yn.sum()>5 else None)
                ypred = model.predict(scaler.transform(Xte))
                fig,ax = _fig(3.8); plot_confusion_matrix(ax,sk_cm(yte,ypred)); _show(fig)
            except Exception as e: st.caption(f"Confusion matrix: {e}")

    # ROC curve
    with col_right:
        try:
            if preds_df is not None and {"Y_TRUE","Y_PROB"}.issubset(preds_df.columns):
                from sklearn.metrics import roc_curve, roc_auc_score
                fpr,tpr,_ = roc_curve(preds_df["Y_TRUE"],preds_df["Y_PROB"])
                auc_v = roc_auc_score(preds_df["Y_TRUE"],preds_df["Y_PROB"]) if preds_df["Y_TRUE"].sum()>0 else 0
            elif model is not None and not X.empty:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import roc_curve, roc_auc_score
                Xn,yn = X.values,y.values
                _,Xte,_,yte = train_test_split(Xn,yn,test_size=0.2,random_state=42,
                                                stratify=yn if yn.sum()>5 else None)
                yprob = model.predict_proba(scaler.transform(Xte))[:,1]
                fpr,tpr,_ = roc_curve(yte,yprob)
                auc_v = roc_auc_score(yte,yprob) if yte.sum()>0 else 0
            else: raise ValueError("No data")
            fig,ax = _fig(3.8); plot_roc_curve(ax,fpr,tpr,auc_v); _show(fig)
        except Exception as e: st.caption(f"ROC curve: {e}")

    col_pr, col_fi = st.columns(2)

    with col_pr:
        try:
            if preds_df is not None and {"Y_TRUE","Y_PROB"}.issubset(preds_df.columns):
                from sklearn.metrics import precision_recall_curve, average_precision_score
                prec,rec,_ = precision_recall_curve(preds_df["Y_TRUE"],preds_df["Y_PROB"])
                ap = average_precision_score(preds_df["Y_TRUE"],preds_df["Y_PROB"])
            elif model is not None and not X.empty:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import precision_recall_curve, average_precision_score
                Xn,yn = X.values,y.values
                _,Xte,_,yte = train_test_split(Xn,yn,test_size=0.2,random_state=42,
                                                stratify=yn if yn.sum()>5 else None)
                yprob = model.predict_proba(scaler.transform(Xte))[:,1]
                prec,rec,_ = precision_recall_curve(yte,yprob)
                ap = average_precision_score(yte,yprob)
            else: raise ValueError("No data")
            fig,ax = _fig(3.8); plot_precision_recall_curve(ax,prec,rec,ap); _show(fig)
        except Exception as e: st.caption(f"PR curve: {e}")

    with col_fi:
        fi_names,fi_vals = [],[]
        if feat_imp is not None:
            nc = next((c for c in feat_imp.columns if "feature" in c.lower()),None)
            vc2= next((c for c in feat_imp.columns if any(w in c.lower() for w in ["import","gain","value"])),None)
            if nc and vc2: fi_names=feat_imp[nc].tolist(); fi_vals=feat_imp[vc2].tolist()
        elif model is not None and hasattr(model,"feature_importances_"):
            fi_vals  = model.feature_importances_.tolist()
            fi_names = X.columns.tolist() if not X.empty else [f"feat_{i}" for i in range(len(fi_vals))]
        if fi_names and fi_vals:
            fig,ax = _fig(4.0); plot_feature_importance(ax,fi_names,fi_vals,top_n=12); _show(fig)
        else: st.caption("Feature importance: train the model first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5- RAG PIPELINE METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("RAG Pipeline Metrics")
    st.caption("ChromaDB vector store statistics and retrieval diagnostics")

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Embedding model", EMBED_MODEL)
    with c2: st.metric("LLM model", LLM_MODEL)
    with c3: st.metric("Top-k retrieval", top_k)
    with c4: st.metric("Collection", CHROMA_COLLECTION)
    st.divider()

    st.markdown("#### ChromaDB vector store")
    chroma_count = 0
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        coll   = client.get_or_create_collection(CHROMA_COLLECTION)
        chroma_count = coll.count()
        ca,cb = st.columns(2)
        with ca: st.metric("Stored embeddings", f"{chroma_count:,}")
        with cb: st.metric("Embedding dimensions", "768")
        if chroma_count == 0:
            st.warning("ChromaDB empty. Run `cd src && python vector_store.py`")
    except Exception as e:
        st.warning(f"ChromaDB not reachable: {e}")

    st.markdown("#### Patient summaries")
    n_sum = len(summaries_df) if not summaries_df.empty else 0
    st.metric("Summaries generated", f"{n_sum:,}")

    if not summaries_df.empty and "SUMMARY" in summaries_df.columns:
        summaries_df["SUMMARY_LEN"] = summaries_df["SUMMARY"].str.len()
        avg_len = summaries_df["SUMMARY_LEN"].mean()
        ca,cb = st.columns(2)
        with ca: st.metric("Avg summary length", f"{avg_len:.0f} chars")
        with cb: st.metric("Total characters", f"{summaries_df['SUMMARY_LEN'].sum():,}")

        fig,ax = _fig(3)
        ax.hist(summaries_df["SUMMARY_LEN"], bins=20, color=C2, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Summary length (chars)",fontsize=9); ax.set_ylabel("Patients",fontsize=9)
        ax.set_title("Patient Summary Length Distribution",fontsize=10,fontweight="bold",color=C1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        _show(fig)

    st.markdown("#### Retrieval similarity distances")
    if st.button("Run sample retrieval diagnostics", use_container_width=False):
        if chroma_count == 0:
            st.warning("Build ChromaDB first.")
        elif not summaries_df.empty and SRC_OK:
            distances = []
            sample_ids = summaries_df["PATIENT"].tolist()[:20]
            prog = st.progress(0, text="Querying ...")
            for i,pid in enumerate(sample_ids):
                row = summaries_df[summaries_df["PATIENT"]==pid]
                if row.empty: continue
                try:
                    res = retrieve_similar_patients(row.iloc[0]["SUMMARY"], top_k=top_k)
                    distances.extend([r["distance"] for r in res])
                except Exception: pass
                prog.progress((i+1)/len(sample_ids), text=f"Patient {i+1}/{len(sample_ids)}")
            prog.empty()
            if distances:
                fig,ax = _fig(3)
                ax.hist(distances,bins=15,color=C1,edgecolor="white",linewidth=0.3)
                ax.set_xlabel("Cosine distance",fontsize=9); ax.set_ylabel("Count",fontsize=9)
                ax.set_title("Similarity Distance Distribution",fontsize=10,fontweight="bold",color=C1)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                _show(fig)
                st.caption(f"Mean: {np.mean(distances):.4f} | Min: {min(distances):.4f} | Max: {max(distances):.4f}")

    st.markdown("#### Example patient summaries")
    n_show = st.slider("Examples to display", 1, 10, 3, key="rag_n")
    if not summaries_df.empty:
        for _,row in summaries_df.head(n_show).iterrows():
            with st.expander(f"Patient {str(row['PATIENT'])[:12]}..."):
                st.write(row["SUMMARY"])
    else:
        st.caption("No summaries loaded. Run training first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6- SYSTEM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("System Pipeline")
    st.caption("End-to-end architecture of the Clinical RAG Predictor")

    render_pipeline_diagram()
    st.divider()

    st.markdown("#### Stage descriptions")
    stages = [
        ("Synthea CSV Files",   "load_data.py",
         "Nine CSV files from the Synthea synthetic EHR generator. Each file is one table: "
         "patients, encounters, conditions, medications, observations, procedures, "
         "allergies, careplans, immunizations."),
        ("Data Preprocessing",  "preprocess.py",
         "Each table is cleaned individually then joined on the PATIENT UUID. Dates are "
         "parsed, ages computed, duplicate rows removed, missing numerics filled with zero. "
         "Output: one wide DataFrame- 117 rows × 16 columns."),
        ("Feature Engineering", "features.py",
         "Selects 11 numeric features (AGE_YEARS, HBA1C, BMI, GLUCOSE, SYSTOLIC_BP, "
         "CONDITION_COUNT, MED_COUNT, PROCEDURE_COUNT, ENCOUNTER_COUNT, "
         "UNIQUE_ENCOUNTER_TYPES, IS_DECEASED) plus label-encoded GENDER and RACE. "
         "Extracts the binary DIABETES_COMPLICATION label."),
        ("XGBoost Classifier",  "train_model.py",
         "XGBoost (200 estimators, depth 4, lr 0.05) trained on the 80/20 split. "
         "StandardScaler normalises features. Saved to ./models/risk_model.joblib. "
         "Returns probability + Low/Medium/High category at inference."),
        ("Patient Summaries",   "build_rag_documents.py",
         "Each patient row becomes a natural-language paragraph: age, gender, conditions, "
         "lab values, and outcome label. These are the RAG retrieval documents."),
        ("Embeddings",          "vector_store.py",
         "nomic-embed-text via Ollama converts each summary to a 768-dimensional vector. "
         "Run once offline; takes ~5-10 minutes for 117 patients."),
        ("ChromaDB",            "vector_store.py",
         "All 117 embeddings are upserted into a local ChromaDB persistent collection "
         "in ./chroma_db/. Supports cosine similarity search."),
        ("Retrieval",           "retriever.py",
         "At inference time the query patient summary is embedded and ChromaDB returns "
         "the top-k nearest summaries by cosine distance. Default k=3."),
        ("Ollama LLM",          "llm_explainer.py",
         "Mistral-7B-Instruct (local Ollama) receives: patient summary, risk probability, "
         "risk category, and the top-3 retrieved patient histories. Generates a grounded "
         "plain-English clinical explanation at temperature 0.3."),
        ("Streamlit UI",        "app.py",
         "Six-tab dashboard: Risk Predictor, Dataset Overview, Training Data, "
         "Model Performance, RAG Metrics, and System Pipeline. "
         "All processing is local- no data leaves the machine."),
    ]
    for lbl,mod,desc in stages:
        with st.expander(f"{lbl} -  `{mod}`"):
            st.write(desc)

    st.divider()
    st.markdown("#### Expected file locations")
    file_table = {
        "File / Directory": [
            "./data/patients.csv","./data/encounters.csv","./data/conditions.csv",
            "./data/medications.csv","./data/observations.csv","./data/procedures.csv",
            "./processed/training_dataset.csv",
            "./models/risk_model.joblib","./models/scaler.joblib",
            "./models/metrics.json","./models/feature_importance.csv","./models/predictions.csv",
            "./chroma_db/",
        ],
        "Created by": [
            "Download Synthea","Download Synthea","Download Synthea",
            "Download Synthea","Download Synthea","Download Synthea",
            "Optional manual export",
            "train_model.py","train_model.py",
            "train_model.py (optional)","train_model.py (optional)","train_model.py (optional)",
            "vector_store.py",
        ],
        "Required?": [
            "Yes","Yes","Yes","Yes","Yes","Optional",
            "Optional (auto-rebuilt)",
            "Yes (for predictions)","Yes (for predictions)",
            "Optional","Optional","Optional",
            "Yes (for RAG)",
        ],
    }
    st.dataframe(
        pd.DataFrame(file_table).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()
    st.caption("-----")