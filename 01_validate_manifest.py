import os
import pandas as pd

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "data.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"[PASS] data.csv found with {len(df)} process records.")
    else:
        print("[FAIL] data.csv missing!")

if __name__ == "__main__":
    main()
