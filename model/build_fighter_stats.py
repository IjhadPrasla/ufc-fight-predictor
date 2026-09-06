"""
Builds a per-fighter stats lookup table.

Why this exists: train_model.py learns from fight-vs-fight rows (each
row already has both fighters' stats side by side). But at prediction
time, the user just picks two fighter NAMES from a dropdown — we need
somewhere to look up "what are Fighter X's average stats" so we can
compute the same diff features the model expects.

This script aggregates each fighter's average stats across their
fight history and saves one row per fighter to data/fighter_stats.csv.

PLACEHOLDER column names — update RAW_STAT_PAIRS-equivalent columns
below once you know your dataset's real schema (same columns as
train_model.py's RAW_STAT_PAIRS, just not diffed yet).
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "fighter_stats.csv"

# placeholder — update to match your real dataset's fighter name + stat columns
RED_NAME_COL = "r_fighter_name"
BLUE_NAME_COL = "b_fighter_name"
RED_STAT_COLS = ["r_wins", "r_losses", "r_height", "r_reach", "r_age",
                  "r_sig_str_landed_pm", "r_takedown_avg"]
BLUE_STAT_COLS = ["b_wins", "b_losses", "b_height", "b_reach", "b_age",
                   "b_sig_str_landed_pm", "b_takedown_avg"]
STAT_NAMES = ["wins", "losses", "height", "reach", "age",
              "sig_str_landed_pm", "takedown_avg"]


def build():
    csv_files = list(DATA_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {DATA_PATH}")
    df = pd.read_csv(csv_files[0])

    red = df[[RED_NAME_COL] + RED_STAT_COLS].rename(
        columns=dict(zip([RED_NAME_COL] + RED_STAT_COLS, ["fighter_name"] + STAT_NAMES))
    )
    blue = df[[BLUE_NAME_COL] + BLUE_STAT_COLS].rename(
        columns=dict(zip([BLUE_NAME_COL] + BLUE_STAT_COLS, ["fighter_name"] + STAT_NAMES))
    )

    all_fighters = pd.concat([red, blue], ignore_index=True)
    fighter_stats = all_fighters.groupby("fighter_name", as_index=False).mean(numeric_only=True)

    fighter_stats.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(fighter_stats)} fighters to {OUT_PATH}")


if __name__ == "__main__":
    build()
