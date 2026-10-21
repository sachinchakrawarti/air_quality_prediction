"""
server/app.py
-------------
Flask server that:
  • serves the UI from ../ui/
  • exposes POST /predict  →  {aqi, r2, shap, category}
  • loads trained models from ../models/ at startup

Run from anywhere:
    python server/app.py
Then open http://localhost:5000
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, send_from_directory, request, jsonify

# ----------------------------------------------------------------------
# Paths — resolve from this file's location so cwd doesn't matter
# ----------------------------------------------------------------------
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SERVER_DIR)
UI_DIR     = os.path.join(ROOT_DIR, "ui")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")

if not os.path.isdir(UI_DIR):
    raise SystemExit(f"UI folder not found: {UI_DIR}")
if not os.path.isdir(MODEL_DIR):
    raise SystemExit(f"Models folder not found: {MODEL_DIR} — run notebook cell 20 first")

# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------
app = Flask(__name__, static_folder=UI_DIR, static_url_path="")

# ----------------------------------------------------------------------
# Load all city artifacts once
# ----------------------------------------------------------------------
CITIES = sorted(
    f.replace("rfr_", "").replace(".pkl", "")
    for f in os.listdir(MODEL_DIR)
    if f.startswith("rfr_") and f.endswith(".pkl")
)

if not CITIES:
    raise SystemExit(f"No model files (rfr_*.pkl) found in {MODEL_DIR}")

ARTIFACTS = {}
for city in CITIES:
    try:
        ARTIFACTS[city] = {
            "model":    joblib.load(os.path.join(MODEL_DIR, f"rfr_{city}.pkl")),
            "scaler":   joblib.load(os.path.join(MODEL_DIR, f"scaler_{city}.pkl")),
            "features": joblib.load(os.path.join(MODEL_DIR, f"features_{city}.pkl")),
        }
    except FileNotFoundError as e:
        print(f"  WARNING: missing artifact for {city}: {e}")

# Optional metrics.json (R² per city) — used only for display
METRICS = {}
metrics_path = os.path.join(MODEL_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        METRICS = json.load(f)

print("=" * 60)
print(f"Loaded {len(ARTIFACTS)} city model(s): {list(ARTIFACTS.keys())}")
print("=" * 60)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
SEASON_TO_MONTH = {
    "winter": 1,
    "summer": 4,
    "monsoon": 7,
    "post-monsoon": 10,
}

def aqi_category(aqi: float) -> str:
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"


def build_feature_row(body: dict, features: list) -> pd.DataFrame:
    """Build a 1-row DataFrame matching the exact feature order the model expects."""
    season = body.get("season", "winter")
    row = {}
    for f in features:
        if f in body:
            row[f] = float(body[f])
        elif f.startswith("season_"):
            row[f] = 1.0 if f == f"season_{season}" else 0.0
        elif f == "month":
            row[f] = float(SEASON_TO_MONTH.get(season, 1))
        else:
            row[f] = 0.0
    return pd.DataFrame([row])[features]


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "cities": list(ARTIFACTS.keys()),
        "ui_dir": UI_DIR,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True, silent=True) or {}
        city = body.get("city")

        if not city:
            return jsonify({"error": "Missing 'city' in request body"}), 400
        if city not in ARTIFACTS:
            return jsonify({
                "error": f"No model for city '{city}'. Available: {list(ARTIFACTS.keys())}"
            }), 400

        art    = ARTIFACTS[city]
        feats  = art["features"]
        X_df   = build_feature_row(body, feats)
        X_np   = X_df.to_numpy(dtype=float)

        if np.isnan(X_np).any():
            return jsonify({"error": "NaN in input features — check required fields"}), 400

        X_scaled = art["scaler"].transform(X_np)
        aqi      = float(art["model"].predict(X_scaled)[0])

        # --- SHAP explanation (best-effort; never crashes the request) ---
        shap_dict = {}
        try:
            explainer = shap.TreeExplainer(art["model"])
            sv        = explainer.shap_values(X_scaled)
            sv_row    = sv[0] if isinstance(sv, list) else sv[0]
            shap_dict = {f: float(v) for f, v in zip(feats, sv_row)}
        except Exception as e:
            print(f"  SHAP failed for {city}: {e}")

        # --- Optional per-city R² from metrics.json ---
        r2 = float(METRICS.get(city, {}).get("r2", 0.0)) if METRICS else 0.0

        return jsonify({
            "aqi":       round(aqi, 2),
            "category":  aqi_category(aqi),
            "r2":        round(r2, 3),
            "shap":      shap_dict,
            "city":      city,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"UI    : {UI_DIR}")
    print(f"Models: {MODEL_DIR}")
    print("Open   http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)