import pandas as pd

def assign_parent_splits(data_csv_path):
    df = pd.read_csv(data_csv_path)
    splits = []
    for idx, row in df.iterrows():
        t1 = float(row["Temperature"])
        t2 = float(row.get("Temp2", 0.0))
        if t1 == 980.0 and t2 == 0.0:
            splits.append("level2_980_holdout")
        elif idx % 7 == 0:
            splits.append("validation")
        elif idx % 50 == 0:
            splits.append("level1_internal_test")
        else:
            splits.append("train")
    df["split"] = splits
    return df
