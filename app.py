import os
import sys
import json
import ctypes
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Preload OpenMP for macOS Apple Silicon
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

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Locate models directory
MODELS_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "models"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "models"),
    os.path.join(os.getcwd(), "models"),
    os.path.join(os.getcwd(), "backend", "models"),
    os.path.join(os.getcwd(), "api", "models"),
    "/Users/cryptorth/Documents/ml/projects/models",
    "/Users/cryptorth/Documents/ml/projects/backend/models"
]

models_dir = None
for d in MODELS_DIRS:
    if os.path.exists(os.path.join(d, "best_xgboost_model.pkl")):
        models_dir = d
        break

if models_dir is None:
    raise FileNotFoundError("Could not find trained models directory!")

print(f"Loading ML production artifacts from: {models_dir}")
xgb_model = joblib.load(os.path.join(models_dir, "best_xgboost_model.pkl"))
preprocessor = joblib.load(os.path.join(models_dir, "preprocessor_scaler.pkl"))

with open(os.path.join(models_dir, "model_metadata.json")) as f:
    model_metadata = json.load(f)

# Lazy-loaded Stacking Ensemble (saves ~500MB RAM & avoids cold start timeouts on serverless)
_stacking_model_cache = None

def get_stacking_model():
    global _stacking_model_cache
    if _stacking_model_cache is None:
        stacking_path = os.path.join(models_dir, "stacking_model.pkl")
        if os.path.exists(stacking_path):
            print("⏳ Lazy-loading Stacking Meta-Ensemble on demand...")
            _stacking_model_cache = joblib.load(stacking_path)
            print("✅ Stacking Meta-Ensemble loaded into memory.")
        else:
            print("⚠️ Stacking model artifact not found, falling back to XGBoost.")
    return _stacking_model_cache

print("✅ CardioSense ML Models & Preprocessor loaded successfully (XGBoost eager, Stacking lazy)!")

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "CardioSense ML REST API",
        "status": "online",
        "version": model_metadata.get("version", "2.4.0"),
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
        "models_available": ["xgboost", "stacking"],
        "default_model": "xgboost",
        "optimal_threshold": model_metadata.get("optimal_threshold", 0.35)
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

        # Model selection: 'xgboost' or 'stacking' (with lazy loading)
        model_type = str(data.get("model_type", data.get("model", "xgboost"))).lower()
        if "stack" in model_type:
            stack_m = get_stacking_model()
            if stack_m is not None:
                selected_model = stack_m
                model_name = "Stacking Meta-Ensemble v2.4"
            else:
                selected_model = xgb_model
                model_name = "Tuned XGBoost Booster v2.4 (Stacking Fallback)"
        else:
            selected_model = xgb_model
            model_name = "Tuned XGBoost Booster v2.4"

        # Derived metrics
        height_m = height / 100.0
        bmi = round(weight / (height_m * height_m), 1)
        pulse_pressure = round(max(0.0, ap_hi - ap_lo), 1)

        # Build feature DataFrame matching exact training feature order
        feature_order = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        input_dict = {
            'age': [age],
            'height': [height],
            'weight': [weight],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'bmi': [bmi],
            'pulse_pressure': [pulse_pressure],
            'gender': [gender],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active]
        }
        input_df = pd.DataFrame(input_dict)[feature_order]

        # Standardize features
        input_scaled = preprocessor.transform(input_df)

        # Inference
        probs = selected_model.predict_proba(input_scaled)[0]
        prob_cvd = float(probs[1])
        risk_pct = round(prob_cvd * 100.0, 1)

        # Clinical Risk Stratification
        if risk_pct < 35.0:
            risk_level = "Optimal Risk (< 35%)"
            verdict_class = "verdict-low"
            risk_color = "#10b981"
        elif risk_pct < 65.0:
            risk_level = "Moderate Risk (35 - 65%)"
            verdict_class = "verdict-mod"
            risk_color = "#f59e0b"
        else:
            risk_level = "Elevated Risk (> 65%)"
            verdict_class = "verdict-high"
            risk_color = "#f43f5e"

        # Feature Contributions & Relative Impact
        importances = model_metadata.get("feature_importances", {})
        feature_labels = {
            "ap_hi": "Systolic Blood Pressure",
            "cholesterol": "Cholesterol Level",
            "age": "Patient Age",
            "bmi": "Body Mass Index (BMI)",
            "pulse_pressure": "Pulse Pressure",
            "ap_lo": "Diastolic Blood Pressure",
            "active": "Physical Activity",
            "smoke": "Smoking Status",
            "gluc": "Fasting Glucose",
            "weight": "Body Weight",
            "gender": "Biological Sex",
            "alco": "Alcohol Intake",
            "height": "Height"
        }

        contributions = []
        for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            contributions.append({
                "feature": feat,
                "label": feature_labels.get(feat, feat),
                "importance": round(imp, 4),
                "user_value": input_dict[feat][0]
            })

        optimal_t = model_metadata.get("optimal_threshold", 0.35)

        return jsonify({
            "status": "success",
            "risk_percentage": risk_pct,
            "raw_probability": round(prob_cvd, 4),
            "risk_level": risk_level,
            "verdict_class": verdict_class,
            "risk_color": risk_color,
            "optimal_threshold": optimal_t,
            "is_above_optimal_threshold": bool(prob_cvd >= optimal_t),
            "model_used": model_name,
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
    print(f"Starting CardioSense Flask REST API on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
