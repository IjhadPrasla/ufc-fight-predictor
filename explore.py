import pandas as pd

df = pd.read_csv("data/raw/fighter.csv")

cols = ["height", "weight_lbs", "reach_inches", "stance", "slpm",
        "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]

for c in cols:
    print(c, "->", df[c].dtype)
    print(df[c].dropna().head(5).tolist())
    print()