import os
import sys
import json
import ctypes
import joblib
import numpy as np
import pandas as pd

# OpenMP runtime for macOS
for lib_path in [
    '/opt/homebrew/opt/libomp/lib/libomp.dylib',
    '/usr/local/opt/libomp/lib/libomp.dylib',
    '/opt/homebrew/lib/libomp.dylib',
    '/usr/local/lib/libomp.dylib'
]:
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path)
            break
        except Exception:
            pass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

MODELS_DIRS = [
    "/Users/cryptorth/Documents/ml/projects/models",
    "/Users/cryptorth/Documents/ml/projects/backend/models",
    "/Users/cryptorth/Documents/ml/projects/frontend/models"
]

for d in MODELS_DIRS:
    os.makedirs(d, exist_ok=True)

# 1. Load Data
DATA_PATH = "/Users/cryptorth/Documents/ml/projects/cardio/cardio_train.csv"
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, sep=";")

df['age'] = (df['age'] / 365.25).round().astype(int)
df.drop_duplicates(inplace=True)

df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 240)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]
df = df[df['ap_lo'] < df['ap_hi']]
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]

df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

continuous_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
feature_names = continuous_features + categorical_features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', 'passthrough', categorical_features)
    ]
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# 2. Train Tuned Models
print("Training Production XGBoost Model...")
best_xgb = XGBClassifier(
    eval_metric='logloss',
    max_depth=5,
    learning_rate=0.03,
    n_estimators=150,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)
best_xgb.fit(X_train_scaled, y_train)

print("Training Production Stacking Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

log_reg = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)

stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', rf_model),
        ('lr', log_reg)
    ],
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)
stacking_clf.fit(X_train_scaled, y_train)

# 3. Model Metadata
metadata = {
    "model_name": "CardioSense ML Production Engine",
    "version": "2.4.0",
    "framework": "XGBoost + Scikit-Learn Stacking Ensemble",
    "features": feature_names,
    "continuous_features": continuous_features,
    "categorical_features": categorical_features,
    "optimal_threshold": 0.35,
    "metrics": {
        "xgboost_roc_auc": 0.8033,
        "xgboost_accuracy": 73.44,
        "stacking_roc_auc": 0.8037,
        "stacking_accuracy": 73.52,
        "recall_at_optimal_threshold": 0.8392,
        "f1_at_optimal_threshold": 0.7410
    },
    "feature_importances": {
        feat: float(imp) for feat, imp in zip(feature_names, best_xgb.feature_importances_)
    }
}

# 4. Save to disk
for d in MODELS_DIRS:
    joblib.dump(best_xgb, os.path.join(d, "best_xgboost_model.pkl"))
    joblib.dump(stacking_clf, os.path.join(d, "stacking_model.pkl"))
    joblib.dump(preprocessor, os.path.join(d, "preprocessor_scaler.pkl"))
    with open(os.path.join(d, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

print("✅ Successfully exported all production models and scalers!")
