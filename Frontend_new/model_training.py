"""
model_training.py
-----------------
Trains the XGBoost delay-prediction model from the CSV dataset.
Called once at Flask app startup — do NOT modify model.ipynb.

Returns a dict with:
  model            – trained XGBClassifier
  le_origin        – LabelEncoder for origin
  le_dest          – LabelEncoder for destination
  le_weather       – LabelEncoder for weather
  le_traffic       – LabelEncoder for traffic
  le_carrier       – LabelEncoder for Carrier_History
  accuracy         – float, e.g. 0.88
  feature_names    – list of feature column names
  feature_importances – list of float importances (same order as feature_names)
  confusion_matrix – 2-D list [[TN,FP],[FN,TP]]
  roc_auc          – float
  n_samples        – int, total rows in dataset
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier


# -------------------------------------------------------------------------
# Locate the CSV relative to this file (works regardless of cwd)
# -------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "data", "shipment_dataset_odisha_modified.csv")


def train_model():
    """Load dataset, train XGBoost, return metadata dict."""
    # ------------------------------------------------------------------
    # Load & clean
    # ------------------------------------------------------------------
    data = pd.read_csv(CSV_PATH).ffill().bfill()
    n_samples = len(data)

    # ------------------------------------------------------------------
    # Encode categoricals (exact same step order as notebook)
    # ------------------------------------------------------------------
    le_origin  = LabelEncoder()
    le_dest    = LabelEncoder()
    le_weather = LabelEncoder()
    le_traffic = LabelEncoder()
    le_carrier = LabelEncoder()

    data["origin"]          = le_origin.fit_transform(data["origin"].astype(str))
    data["destination"]     = le_dest.fit_transform(data["destination"].astype(str))
    data["weather"]         = le_weather.fit_transform(data["weather"].astype(str))
    data["traffic"]         = le_traffic.fit_transform(data["traffic"].astype(str))
    data["Carrier_History"] = le_carrier.fit_transform(data["Carrier_History"].astype(str))

    X = data.drop(["shipment_id", "delay"], axis=1).astype(np.float32)
    y = data["delay"]

    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # Train / test split (same params as notebook)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------------------
    # XGBoost (same hyper-params as notebook)
    # ------------------------------------------------------------------
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.6,
        max_depth=12,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = round(accuracy_score(y_test, y_pred), 4)

    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()   # [[TN, FP], [FN, TP]]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = round(auc(fpr, tpr), 4)

    importances = list(model.feature_importances_.astype(float))

    print(f"[model_training] Model loaded. Accuracy: {acc}  |  ROC-AUC: {roc_auc_val}")

    return {
        "model":               model,
        "le_origin":           le_origin,
        "le_dest":             le_dest,
        "le_weather":          le_weather,
        "le_traffic":          le_traffic,
        "le_carrier":          le_carrier,
        "accuracy":            acc,
        "feature_names":       feature_names,
        "feature_importances": importances,
        "confusion_matrix":    cm_list,
        "roc_auc":             roc_auc_val,
        "n_samples":           n_samples,
    }
