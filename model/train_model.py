"""
UFC Fight Winner Prediction - Model Training

Uses master.csv, which has one row per fight with both fighters'
PRE-FIGHT attributes (r_/b_ prefixed) plus the fight outcome.

IMPORTANT: we only use pre-fight known attributes (height, reach,
career striking/grappling averages, age) as features. We deliberately
exclude columns like r_total_sig_landed, r_total_td_success, etc. —
those are stats generated DURING the fight itself, so using them to
predict the winner would be data leakage (the model would essentially
already know the outcome).
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import re

def parse_height(height_str):
    """Converts "5' 10\"" style strings to total inches as a float."""
    if pd.isna(height_str):
        return None
    match = re.match(r"(\d+)'\s*(\d+)", str(height_str))
    if not match:
        return None
    feet, inches = int(match.group(1)), int(match.group(2))
    return feet * 12 + inches

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "master.csv"
MODEL_OUT = Path(__file__).parent / "trained_model.joblib"

# Pre-fight attributes only (no data leakage from in-fight stats)
STAT_NAMES = [
    "height", "weight_lbs", "reach_inches",
    "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"master.csv not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # age at time of fight = event_date - dob
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["r_dob"] = pd.to_datetime(df["r_dob"], errors="coerce")
    df["b_dob"] = pd.to_datetime(df["b_dob"], errors="coerce")
    df["r_age"] = (df["event_date"] - df["r_dob"]).dt.days / 365.25
    df["b_age"] = (df["event_date"] - df["b_dob"]).dt.days / 365.25

    feature_cols = []
    df["r_height"] = df["r_height"].apply(parse_height)
    df["b_height"] = df["b_height"].apply(parse_height)
    all_stats = STAT_NAMES + ["age"]
    for stat in all_stats:
        r_col, b_col = f"r_{stat}", f"b_{stat}"
        if r_col in df.columns and b_col in df.columns:
            diff_col = f"diff_{stat}"
            df[diff_col] = df[r_col] - df[b_col]
            feature_cols.append(diff_col)

    # target: 1 if red corner (r_fighter_id) won, 0 if blue corner won.
    # drop fights with no clear winner (draws/no-contests)
    valid = df["winner_id"].notna() & (
        (df["winner_id"] == df["r_fighter_id"]) | (df["winner_id"] == df["b_fighter_id"])
    )
    df = df[valid].copy()
    df["target"] = (df["winner_id"] == df["r_fighter_id"]).astype(int)

    df = df.dropna(subset=feature_cols + ["target"])
    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_model, best_auc, best_name = None, -1, None
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.3f}  |  AUC: {auc:.3f}")
        print(classification_report(y_test, preds))
        if auc > best_auc:
            best_auc, best_model, best_name = auc, model, name

    print(f"\nBest model: {best_name} (AUC={best_auc:.3f})")
    return best_model


if __name__ == "__main__":
    df = load_data()
    X, y, feature_cols = engineer_features(df)
    print(f"\nTraining on {len(X)} fights with features: {feature_cols}")

    model = train_and_evaluate(X, y)

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_OUT)
    print(f"\nSaved trained model to {MODEL_OUT}")