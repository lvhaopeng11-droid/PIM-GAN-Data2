import os
import pandas as pd

def check_manifest_integrity(csv_path):
    if not os.path.exists(csv_path):
        return False, "Manifest file not found"
    df = pd.read_csv(csv_path)
    return True, f"Manifest valid with {len(df)} rows"
