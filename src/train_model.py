"""
train_model.py
--------------
Trains an XGBoost classifier on the patient feature matrix.
Saves the trained model and scaler to ./models/.
Prints evaluation metrics.

Run this script once before launching the Streamlit app:
    cd src && python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

try:
    from xgboost import XGBClassifier
    MODEL_TYPE = "xgboost"
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    MODEL_TYPE = "randomforest"
    print("[train_model] XGBoost not found — falling back to RandomForest.")

from config import (
    MODELS_DIR, MODEL_FILENAME, SCALER_FILENAME,
    TEST_SIZE, RANDOM_STATE,
)
from load_data   import load_all_csv
from preprocess  import merge_all
from features    import build_feature_matrix


def train_and_evaluate():
    """End-to-end training pipeline. Returns the trained model."""

    # 1. Load and preprocess data
    print("=" * 55)
    print("STEP 1: Loading Synthea CSV files …")
    data   = load_all_csv()
    merged = merge_all(data)

    # 2. Build feature matrix
    print("\nSTEP 2: Building feature matrix …")
    X, y = build_feature_matrix(merged)
    patient_ids = X.index.tolist()

    # 3. Train / test split
    X_np  = X.values
    y_np  = y.values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_np, y_np, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_np if y_np.sum() > 10 else None
    )
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # 4. Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 5. Train model
    print(f"\nSTEP 3: Training {MODEL_TYPE} …")
    if MODEL_TYPE == "xgboost":
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=RANDOM_STATE,
        )

    model.fit(X_tr_s, y_tr)

    # 6. Evaluate
    print("\nSTEP 4: Evaluating on test set …")
    y_pred  = model.predict(X_te_s)
    y_proba = model.predict_proba(X_te_s)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_te, y_pred, zero_division=0))

    # Feature importance (top 10)
    feature_names = X.columns.tolist()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 feature importances:")
        for name, imp in top:
            print(f"  {name:<30} {imp:.4f}")

    # 7. Save model and scaler
    model_path  = os.path.join(MODELS_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME)
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[train_model] Model saved  -> {model_path}")
    print(f"[train_model] Scaler saved -> {scaler_path}")

    return model, scaler, X.columns.tolist()


def load_model():
    """Load a previously trained model and scaler from ./models/."""
    model_path  = os.path.join(MODELS_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Please run:  cd src && python train_model.py"
        )
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_patient(patient_row: pd.Series, model, scaler) -> tuple[float, str]:
    """
    Predict risk probability for a single patient.

    Parameters
    ----------
    patient_row : pd.Series — one row from the feature matrix (X)
    model, scaler : trained objects

    Returns
    -------
    (probability, risk_category)
    """
    from config import RISK_LOW, RISK_MEDIUM
    X_single = scaler.transform(patient_row.values.reshape(1, -1))
    prob = model.predict_proba(X_single)[0][1]
    if prob < RISK_LOW:
        category = "Low"
    elif prob < RISK_MEDIUM:
        category = "Medium"
    else:
        category = "High"
    return round(float(prob), 4), category


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_evaluate()
