"""
train_model.py
--------------
Trains an XGBoost (or RandomForest fallback) classifier.

Strategy for tiny imbalanced datasets (8 positives / 117 total):
  - Use Leave-One-Out or Stratified K-Fold cross-validation for ALL metrics.
    A single 80/20 split routinely puts 0 positives in the test set, making
    every metric undefined.
  - Train the FINAL model on ALL 117 patients (no holdout) so it learns
    from every positive case.
  - Save cross-validated predictions as predictions.csv so Tab 4 charts
    are drawn from real evaluation data, not a broken single split.

Saved files
-----------
./models/risk_model.joblib        trained model (fit on all data)
./models/scaler.joblib            fitted StandardScaler
./models/metrics.json             CV-averaged metrics
./models/feature_importance.csv   feature name + importance score
./models/predictions.csv          Y_TRUE, Y_PROB from cross-validation
                                   (used by Tab 4 ROC / PR / confusion charts)

Run:
    cd src && python train_model.py
"""

import os, json, warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, LeaveOneOut
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
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
    RANDOM_STATE,
)
from load_data  import load_all_csv
from preprocess import merge_all
from features   import build_feature_matrix


METRICS_JSON    = os.path.join(MODELS_DIR, "metrics.json")
FEAT_IMP_CSV    = os.path.join(MODELS_DIR, "feature_importance.csv")
PREDICTIONS_CSV = os.path.join(MODELS_DIR, "predictions.csv")


def _make_model(n_pos, n_neg):
    """Build a fresh (unfitted) model with imbalance correction."""
    spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
    if MODEL_TYPE == "xgboost":
        return XGBClassifier(
            n_estimators     = 300,
            max_depth        = 3,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            scale_pos_weight = spw,
            eval_metric      = "logloss",
            random_state     = RANDOM_STATE,
            verbosity        = 0,
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators = 300,
            max_depth    = 4,
            class_weight = "balanced",
            random_state = RANDOM_STATE,
        )


def train_and_evaluate():
    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Synthea CSV files ...")
    merged = merge_all(load_all_csv())

    # ── STEP 2: Features ──────────────────────────────────────────────────────
    print("\nSTEP 2: Building feature matrix ...")
    X, y          = build_feature_matrix(merged)
    feature_names = X.columns.tolist()
    X_np, y_np    = X.values, y.values

    n_pos = int(y_np.sum())
    n_neg = int(len(y_np) - n_pos)
    print(f"  Total: {len(y_np)} | Positive: {n_pos} | Negative: {n_neg}")
    print(f"  Imbalance ratio: {n_neg/n_pos:.1f}:1" if n_pos else "  No positives found!")

    if n_pos == 0:
        print("ERROR: No positive samples found. Check label derivation.")
        return None, None, feature_names

    # ── STEP 3: Cross-validation strategy ────────────────────────────────────
    # With only 8 positives, use Leave-One-Out (LOO) so every positive
    # appears in the test fold exactly once.
    # LOO gives 117 folds of 116 train / 1 test — robust for tiny datasets.
    print("\nSTEP 3: Cross-validated evaluation (Leave-One-Out) ...")

    scaler_cv = StandardScaler()
    X_scaled  = scaler_cv.fit_transform(X_np)

    model_cv  = _make_model(n_pos, n_neg)
    cv        = LeaveOneOut()

    # cross_val_predict returns one prediction per sample across all LOO folds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_prob_cv = cross_val_predict(
            model_cv, X_scaled, y_np,
            cv     = cv,
            method = "predict_proba",
        )[:, 1]
        y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    # ── STEP 4: Compute metrics from CV predictions ───────────────────────────
    acc  = accuracy_score(y_np, y_pred_cv)
    prec = precision_score(y_np, y_pred_cv, zero_division=0)
    rec  = recall_score(y_np, y_pred_cv, zero_division=0)
    f1   = f1_score(y_np, y_pred_cv, zero_division=0)
    try:
        auc = roc_auc_score(y_np, y_prob_cv)
    except ValueError:
        auc = float("nan")

    print(f"\n  LOO Cross-Validation Results (all {len(y_np)} samples)")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\nClassification Report (LOO CV):")
    print(classification_report(y_np, y_pred_cv, zero_division=0))
    cm = confusion_matrix(y_np, y_pred_cv)
    print(f"Confusion matrix (LOO CV):\n{cm}")
    print(f"  TP={cm[1,1]}  FP={cm[0,1]}  TN={cm[0,0]}  FN={cm[1,0]}")

    # Also report 5-fold stratified CV AUC as a sanity check
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv5_auc = cross_val_score(
            _make_model(n_pos, n_neg), X_scaled, y_np,
            cv=skf, scoring="roc_auc"
        )
    print(f"\n  5-fold Stratified CV AUC: {cv5_auc.mean():.4f} +/- {cv5_auc.std():.4f}")

    # ── STEP 5: Train FINAL model on ALL data ─────────────────────────────────
    print("\nSTEP 5: Training final model on ALL 117 patients ...")
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_np)

    model = _make_model(n_pos, n_neg)
    model.fit(X_full, y_np)
    print("  Final model trained.")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        top = sorted(zip(feature_names, model.feature_importances_),
                     key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 feature importances:")
        for name, imp in top:
            print(f"  {name:<30} {imp:.4f}")

    # ── STEP 6: Save model + scaler ───────────────────────────────────────────
    model_path  = os.path.join(MODELS_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME)
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[train_model] Model saved  -> {model_path}")
    print(f"[train_model] Scaler saved -> {scaler_path}")

    # ── STEP 7: Save metrics.json ─────────────────────────────────────────────
    metrics = {
        "accuracy":        round(float(acc),  4),
        "precision":       round(float(prec), 4),
        "recall":          round(float(rec),  4),
        "f1":              round(float(f1),   4),
        "roc_auc":         round(float(auc),  4) if not np.isnan(auc) else None,
        "cv_strategy":     "Leave-One-Out",
        "cv_auc_5fold":    round(float(cv5_auc.mean()), 4),
        "cv_auc_5fold_std":round(float(cv5_auc.std()),  4),
        "model_type":      MODEL_TYPE,
        "n_estimators":    300,
        "max_depth":       3,
        "learning_rate":   0.05,
        "n_samples":       int(len(y_np)),
        "n_features":      int(X.shape[1]),
        "n_positive":      int(n_pos),
        "n_negative":      int(n_neg),
        "note": (
            "Metrics from Leave-One-Out CV on all 117 patients. "
            "Final model trained on all data (no holdout) to maximise "
            "learning from 8 positive cases."
        ),
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_model] Metrics saved -> {METRICS_JSON}")

    # ── STEP 8: Save feature_importance.csv ───────────────────────────────────
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_.tolist(),
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(FEAT_IMP_CSV, index=False)
        print(f"[train_model] Feature importance saved -> {FEAT_IMP_CSV}")

    # ── STEP 9: Save predictions.csv ──────────────────────────────────────────
    # Uses LOO CV probabilities — every patient appears in the test fold
    # exactly once, so all 117 rows are present including all 8 positives.
    # Tab 4 charts (ROC, PR, confusion matrix) read from this file.
    pred_df = pd.DataFrame({
        "Y_TRUE": y_np.tolist(),
        "Y_PRED": y_pred_cv.tolist(),
        "Y_PROB": [round(float(p), 6) for p in y_prob_cv],
    })
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[train_model] Predictions saved -> {PREDICTIONS_CSV}")
    print(f"             ({len(pred_df)} rows | "
          f"{int(pred_df['Y_TRUE'].sum())} positives | "
          f"{int(len(pred_df) - pred_df['Y_TRUE'].sum())} negatives)")

    print("\n" + "=" * 60)
    print("Training complete. Run:  streamlit run app.py")
    print("=" * 60)
    return model, scaler, feature_names


def load_model():
    """Load a previously trained model and scaler from ./models/."""
    model_path  = os.path.join(MODELS_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Please run:  cd src && python train_model.py"
        )
    return joblib.load(model_path), joblib.load(scaler_path)


def predict_patient(patient_row: pd.Series, model, scaler) -> tuple[float, str]:
    """Return (probability, risk_category) for a single patient row."""
    from config import RISK_LOW, RISK_MEDIUM
    X_single = scaler.transform(patient_row.values.reshape(1, -1))
    prob     = model.predict_proba(X_single)[0][1]
    if prob < RISK_LOW:
        category = "Low"
    elif prob < RISK_MEDIUM:
        category = "Medium"
    else:
        category = "High"
    return round(float(prob), 4), category


if __name__ == "__main__":
    train_and_evaluate()