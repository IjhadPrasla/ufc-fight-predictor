"""
Builds a per-fighter stats lookup table used at prediction time.

fighter.csv already has one row per fighter with career stats, so
this mostly just selects the relevant columns and computes current
age from date of birth.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
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

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "fighter.csv"
OUT_PATH = Path(__file__).parent.parent / "data" / "fighter_stats.csv"

STAT_NAMES = [
    "height", "weight_lbs", "reach_inches",
    "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
]


def build():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"fighter.csv not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["height"] = df["height"].apply(parse_height)

    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    today = pd.Timestamp(datetime.now())
    df["age"] = (today - df["dob"]).dt.days / 365.25

    keep_cols = ["fighter_name"] + STAT_NAMES + ["age"]
    fighter_stats = df[keep_cols].dropna(subset=["fighter_name"])

    # Impute missing stats with the column median rather than dropping fighters
    # entirely — a fighter with no UFC striking data yet still deserves a
    # reasonable estimate rather than crashing the prediction.
    for col in STAT_NAMES + ["age"]:
        if col in fighter_stats.columns:
            median_val = fighter_stats[col].median()
            fighter_stats[col] = fighter_stats[col].fillna(median_val)

    # if there are duplicate names, keep the first occurrence
    fighter_stats = fighter_stats.drop_duplicates(subset="fighter_name")

    fighter_stats.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(fighter_stats)} fighters to {OUT_PATH}")


if __name__ == "__main__":
    build()