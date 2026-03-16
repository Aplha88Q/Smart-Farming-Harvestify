"""
crop_recommendation.py
----------------------
Trains a CatBoost classifier on the Crop Recommendation dataset and persists
the trained model + feature scaler for inference.

Dataset  : 2,200 samples | 7 features | 22 crop classes
Source   : https://github.com/Gladiator07/Harvestify
Best CV  : 99.38% (CatBoost, 5-fold cross-validation)

Usage:
    python crop_recommendation.py
Output:
    model/catboost_model.cbm   - Trained CatBoost model
    model/scaler.pkl           - Fitted StandardScaler
"""

import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/"
    "Data-processed/crop_recommendation.csv"
)

print("[INFO] Loading dataset...")
data = pd.read_csv(DATASET_URL)
print(f"[INFO] Dataset shape: {data.shape}")
print(f"[INFO] Crop classes  : {sorted(data['label'].unique())}\n")

# Features: N, P, K, temperature, humidity, ph, rainfall
X = data.drop(columns=["label"])
y = data["label"]

# ---------------------------------------------------------------------------
# 2. Train / test split  (80 / 20, stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Feature scaling
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Model training — CatBoost
# ---------------------------------------------------------------------------
print("[INFO] Training CatBoost model...")
model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=0,
)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"[RESULT] Test Accuracy : {test_acc * 100:.2f}%")

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"[RESULT] CV-5 Accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

print("\n[REPORT] Per-class metrics:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 6. Persist model artefacts
# ---------------------------------------------------------------------------
os.makedirs("model", exist_ok=True)

model.save_model("model/catboost_model.cbm")
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n[INFO] Saved: model/catboost_model.cbm")
print("[INFO] Saved: model/scaler.pkl")

# ---------------------------------------------------------------------------
# 7. Quick inference demo
# ---------------------------------------------------------------------------
SAMPLE = {
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.88, "humidity": 82.00,
    "ph": 6.50, "rainfall": 202.94,
}

sample_df     = pd.DataFrame([SAMPLE])
sample_scaled = scaler.transform(sample_df)
prediction    = model.predict(sample_scaled)[0]

print(f"\n[DEMO] Input  : {SAMPLE}")
print(f"[DEMO] Recommended crop: {prediction}")
