"""
FastAPI backend for the UFC fight predictor.

Endpoints:
  GET  /fighters          -> list of fighter names (for the frontend dropdowns)
  GET  /predict?f1=X&f2=Y -> win probability for fighter f1 vs fighter f2
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
MODEL_PATH = BASE / "model" / "trained_model.joblib"
FIGHTER_STATS_PATH = BASE / "data" / "fighter_stats.csv"

app = FastAPI(title="UFC Fight Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_bundle = None
_fighter_stats = None


@app.on_event("startup")
def load_artifacts():
    global _model_bundle, _fighter_stats
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run model/train_model.py first.")
    if not FIGHTER_STATS_PATH.exists():
        raise RuntimeError(f"Fighter stats not found at {FIGHTER_STATS_PATH}. "
                            f"Run model/build_fighter_stats.py first.")
    _model_bundle = joblib.load(MODEL_PATH)
    _fighter_stats_raw = pd.read_csv(FIGHTER_STATS_PATH)
    _fighter_stats_raw["fighter_name_lower"] = _fighter_stats_raw["fighter_name"].str.lower()
    _fighter_stats = _fighter_stats_raw.set_index("fighter_name_lower")


@app.get("/fighters")
def list_fighters():
    return {"fighters": sorted(_fighter_stats["fighter_name"].tolist())}


@app.get("/predict")
def predict(
    f1: str = Query(..., description="Fighter 1 name (red corner)"),
    f2: str = Query(..., description="Fighter 2 name (blue corner)"),
):
    f1_key, f2_key = f1.lower(), f2.lower()

    if f1_key not in _fighter_stats.index:
        raise HTTPException(404, f"Fighter not found: {f1}")
    if f2_key not in _fighter_stats.index:
        raise HTTPException(404, f"Fighter not found: {f2}")

    stats1 = _fighter_stats.loc[f1_key]
    stats2 = _fighter_stats.loc[f2_key]

    model = _model_bundle["model"]
    feature_cols = _model_bundle["feature_cols"]

    row = {}
    for col in feature_cols:
        stat_name = col.replace("diff_", "")
        row[col] = stats1.get(stat_name, 0) - stats2.get(stat_name, 0)

    X = pd.DataFrame([row])[feature_cols]
    prob_f1_wins = model.predict_proba(X)[0][1]

    return {
        "fighter_1": f1,
        "fighter_2": f2,
        "fighter_1_win_probability": round(float(prob_f1_wins), 3),
        "fighter_2_win_probability": round(float(1 - prob_f1_wins), 3),
        "stat_comparison": {
            stat: {"fighter_1": float(stats1.get(stat, 0)), "fighter_2": float(stats2.get(stat, 0))}
            for stat in ["height", "weight_lbs", "reach_inches", "age", "slpm", "td_avg"]
            if stat in stats1.index
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}