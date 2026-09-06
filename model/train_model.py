"""
UFC Fight Winner Prediction - Model Training

This script:
1. Loads the raw UFC fight dataset
2. Engineers features (stat DIFFERENCES between the two fighters,
   since "who wins" depends on relative advantage, not raw stats)
3. Trains a couple of simple classifiers and compares them
4. Saves the best model + the list of feature columns it expects

NOTE: The column names below (RAW_COLUMNS section) are placeholders.
Once you download the real Kaggle dataset, run:

    import pandas as pd
    df = pd.read_csv("data/raw/YOUR_FILE.csv")
    print(df.columns.tolist())
    print(df.head())

...and send Claude the output so we can match this script to the
actual column names in your file.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DATA_PATH = Path(__file__).parent.parent / "data" / "raw"
MODEL_OUT = Path(__file__).parent / "trained_model.joblib"

# ---------------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    csv_files = list(DATA_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {DATA_PATH}. Download the Kaggle dataset "
            f"and place it there first (see PUT_DATASET_HERE.md)."
        )
    print(f"Loading: {csv_files[0].name}")
    return pd.read_csv(csv_files[0])


# ---------------------------------------------------------------------
# STEP 2: Feature engineering
# ---------------------------------------------------------------------
# PLACEHOLDER column names — update these once you know the real
# dataset's columns. The idea stays the same regardless: for every
# raw stat, compute (fighter_A_stat - fighter_B_stat) so the model
# learns from *relative* advantage, which is what actually predicts
# a winner.
RAW_STAT_PAIRS = [
    ("r_wins", "b_wins"),
    ("r_losses", "b_losses"),
    ("r_height", "b_height"),
    ("r_reach", "b_reach"),
    ("r_age", "b_age"),
    ("r_sig_str_landed_pm", "b_sig_str_landed_pm"),
    ("r_takedown_avg", "b_takedown_avg"),
]
TARGET_COLUMN = "winner"  # placeholder — e.g. 1 if red corner won, 0 if blue


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = []
    for red_col, blue_col in RAW_STAT_PAIRS:
        if red_col in df.columns and blue_col in df.columns:
            diff_col = f"diff_{red_col.replace('r_', '')}"
            df[diff_col] = df[red_col] - df[blue_col]
            feature_cols.append(diff_col)
        else:
            print(f"WARNING: columns '{red_col}'/'{blue_col}' not found — skipping. "
                  f"Update RAW_STAT_PAIRS to match your real dataset.")

    if not feature_cols:
        raise ValueError(
            "No feature columns were built. Update RAW_STAT_PAIRS in this "
            "script to match your dataset's actual column names."
        )

    df = df.dropna(subset=feature_cols + [TARGET_COLUMN])
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    return X, y, feature_cols


# ---------------------------------------------------------------------
# STEP 3: Train + evaluate
# ---------------------------------------------------------------------
def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_model = None
    best_auc = -1
    best_name = None

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
            best_auc = auc
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} (AUC={best_auc:.3f})")
    return best_model


if __name__ == "__main__":
    df = load_data()
    X, y, feature_cols = engineer_features(df)
    print(f"\nTraining on {len(X)} fights with features: {feature_cols}")

    model = train_and_evaluate(X, y)

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_OUT)
    print(f"\nSaved trained model to {MODEL_OUT}")
