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

# allow the React dev server (and later your deployed frontend) to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend URL before shipping
    allow_methods=["*"],
    allow_headers=["*"],
)

# loaded once at startup
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
    _fighter_stats = pd.read_csv(FIGHTER_STATS_PATH).set_index("fighter_name")


@app.get("/fighters")
def list_fighters():
    return {"fighters": sorted(_fighter_stats.index.tolist())}


@app.get("/predict")
def predict(
    f1: str = Query(..., description="Fighter 1 name (red corner)"),
    f2: str = Query(..., description="Fighter 2 name (blue corner)"),
):
    if f1 not in _fighter_stats.index:
        raise HTTPException(404, f"Fighter not found: {f1}")
    if f2 not in _fighter_stats.index:
        raise HTTPException(404, f"Fighter not found: {f2}")

    stats1 = _fighter_stats.loc[f1]
    stats2 = _fighter_stats.loc[f2]

    model = _model_bundle["model"]
    feature_cols = _model_bundle["feature_cols"]

    # rebuild the same diff features used in training: diff_<stat> = f1_stat - f2_stat
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
            for stat in ["wins", "losses", "height", "reach", "age", "sig_str_landed_pm", "takedown_avg"]
            if stat in stats1.index
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}
