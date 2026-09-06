# Dataset goes here

1. Go to Kaggle and search "UFC fight data" — look for a dataset with
   historical fighter-vs-fighter stats (common one is by user `rajeevw`,
   sometimes titled "UFC-Fight historical data and fight stats").
2. Download the CSV(s).
3. Place them in this folder (`data/raw/`).
4. Once it's here, send Claude the column names (`df.columns.tolist()`
   after loading it with pandas) so the feature engineering in
   `model/train_model.py` can be matched to the real columns — the
   script currently has placeholder column names as a starting guess.
