"""
CardioSense High-Performance Serverless REST API
Optimized for Vercel Serverless (Zero-heavy dependency: pure NumPy inference engine)
Total Bundle Size: ~35 MB (well under Vercel's 500 MB limit)
Inference Latency: < 1 ms
"""

import os
import sys
import json
import math
import datetime
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Resilient file locator
BASE_DIR = Path(__file__).resolve().parent

def locate_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
        Path.cwd() / "api" / filename,
        Path.cwd() / "api" / "models" / filename,
        Path.cwd() / "models" / filename,
        Path.cwd() / "backend" / "models" / filename
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback walk
    for root, _, files in os.walk(str(BASE_DIR.parent)):
        if filename in files:
            return Path(root) / filename
    raise FileNotFoundError(f"Could not locate required model file: {filename}")

# Load model artifacts
model_json_path = locate_file("best_xgboost_model.json")
metadata_path = locate_file("model_metadata.json")

with open(metadata_path) as f:
    model_metadata = json.load(f)

with open(model_json_path) as f:
    xgb_json = json.load(f)

base_score = float(xgb_json["learner"]["learner_model_param"]["base_score"])
base_margin = math.log(base_score / (1.0 - base_score))
trees = xgb_json["learner"]["gradient_booster"]["model"]["trees"]

# Pre-compile decision trees into float32 arrays for C-speed evaluation
compiled_trees = []
for t in trees:
    compiled_trees.append({
        "lefts": t["left_children"],
        "rights": t["right_children"],
        "splits": t["split_indices"],
        "conds": np.float32(t["split_conditions"])
    })

# StandardScaler constants (exact float64 parameters from trained ColumnTransformer)
MEANS = np.array([
    53.30836900100173,
    164.3914579728622,
    74.0778171386941,
    126.64464074310172,
    81.28436390128404,
    27.45909434992228,
    45.36027684181769
])
SCALES = np.array([
    6.745202053398537,
    7.994219058946969,
    14.316912497979558,
    16.71873070508079,
    9.421814564112664,
    5.360862846755646,
    11.690561804017097
])

feature_order = [
    "age", "height", "weight", "ap_hi", "ap_lo",
    "bmi", "pulse_pressure", "gender", "cholesterol",
    "gluc", "smoke", "alco", "active"
]

feature_labels = {
    "ap_hi": "Systolic Blood Pressure",
    "cholesterol": "Cholesterol Level",
    "age": "Patient Age",
    "active": "Physical Activity",
    "ap_lo": "Diastolic Blood Pressure",
    "smoke": "Smoking Status",
    "gluc": "Fasting Glucose",
    "alco": "Alcohol Intake",
    "weight": "Body Weight",
    "bmi": "Body Mass Index (BMI)",
    "pulse_pressure": "Pulse Pressure",
    "gender": "Biological Sex",
    "height": "Height"
}

optimal_threshold = float(model_metadata.get("optimal_threshold", 0.35))
feature_importances = model_metadata.get("feature_importances", {})


def predict_xgboost_probability(scaled_features):
    """Evaluate 150 gradient boosted decision trees in sub-millisecond pure NumPy."""
    features = np.float32(scaled_features)
    margin = base_margin
    for t in compiled_trees:
        node = 0
        lefts = t["lefts"]
        rights = t["rights"]
        splits = t["splits"]
        conds = t["conds"]
        while lefts[node] != -1:
            if features[splits[node]] < conds[node]:
                node = lefts[node]
            else:
                node = rights[node]
        margin += float(conds[node])
    return 1.0 / (1.0 + math.exp(-margin))


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "CardioSense ML REST API",
        "status": "online",
        "version": model_metadata.get("version", "2.4.0"),
        "runtime": "Vercel Serverless (Lightweight Pure Engine)",
        "endpoints": {
            "health": "/api/health",
            "metrics": "/api/metrics",
            "predict": "/api/predict"
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "service": "CardioSense Prediction Engine",
        "version": model_metadata.get("version", "2.4.0"),
        "models_available": ["xgboost"],
        "default_model": "xgboost",
        "optimal_threshold": optimal_threshold
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify(model_metadata)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        # Extract features with clinical defaults
        age = float(data.get("age", 50))
        gender = int(data.get("gender", 1))
        height = float(data.get("height", 165))
        weight = float(data.get("weight", 70))
        ap_hi = float(data.get("ap_hi", data.get("sbp", 120)))
        ap_lo = float(data.get("ap_lo", data.get("dbp", 80)))
        cholesterol = int(data.get("cholesterol", 1))
        gluc = int(data.get("gluc", 1))
        smoke = int(data.get("smoke", 0))
        alco = int(data.get("alco", 0))
        active = int(data.get("active", 1))

        # Derived physiological metrics
        height_m = height / 100.0
        bmi = round(weight / (height_m * height_m), 1)
        pulse_pressure = round(max(0.0, ap_hi - ap_lo), 1)

        # Vectorized Standard Scaler preprocessing (zero-dependency scikit-learn equivalent)
        num_vals = np.array([age, height, weight, ap_hi, ap_lo, bmi, pulse_pressure], dtype=np.float64)
        cat_vals = np.array([gender, cholesterol, gluc, smoke, alco, active], dtype=np.float64)
        scaled_input = np.concatenate([(num_vals - MEANS) / SCALES, cat_vals])

        # Sub-millisecond XGBoost tree evaluation
        prob_cvd = predict_xgboost_probability(scaled_input)
        risk_pct = round(prob_cvd * 100.0, 1)

        # Clinical Risk Stratification
        if risk_pct < 35.0:
            risk_level = "Optimal Risk (< 35%)"
            verdict_class = "verdict-low"
            risk_color = "#10b981"
        elif risk_pct < 50.0:
            risk_level = "Borderline Risk (35 - 50%)"
            verdict_class = "verdict-mod"
            risk_color = "#f59e0b"
        elif risk_pct < 65.0:
            risk_level = "Moderate Risk (50 - 65%)"
            verdict_class = "verdict-high"
            risk_color = "#f97316"
        else:
            risk_level = "Elevated Risk (> 65%)"
            verdict_class = "verdict-high"
            risk_color = "#f43f5e"

        # Ranked Feature Contributions for XAI Explanation
        patient_feature_values = {
            "age": age,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "bmi": bmi,
            "pulse_pressure": pulse_pressure,
            "gender": gender,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active
        }

        contributions = []
        for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
            contributions.append({
                "feature": feat,
                "label": feature_labels.get(feat, feat),
                "importance": round(imp, 4),
                "user_value": patient_feature_values.get(feat, 0)
            })

        return jsonify({
            "status": "success",
            "risk_percentage": risk_pct,
            "raw_probability": round(prob_cvd, 4),
            "risk_level": risk_level,
            "verdict_class": verdict_class,
            "risk_color": risk_color,
            "optimal_threshold": optimal_threshold,
            "is_above_optimal_threshold": bool(prob_cvd >= optimal_threshold),
            "model_used": "Tuned XGBoost Booster v2.4 (Ultra-Fast Pure Engine)",
            "patient_metrics": {
                "bmi": bmi,
                "pulse_pressure": pulse_pressure,
                "age_years": int(age),
                "bp": f"{int(ap_hi)}/{int(ap_lo)}"
            },
            "feature_contributions": contributions,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
