import pandas as pd
import numpy as np
import os

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"

def check_extremes(file_path, condition):
    try:
        xl = pd.ExcelFile(file_path)
        df = xl.parse(condition)
        
        print(f"\nChecking {os.path.basename(file_path)} - {condition}")
        for col in df.columns:
            vals = df[col].dropna().values
            if len(vals) > 0:
                mean_delta = np.mean(vals)
                print(f"  {col}: n_pillars={len(vals)}, Mean Delta={mean_delta:.3f}%")
    except Exception as e:
        print("Error:", e)

check_extremes(os.path.join(out_dir, "DNA_QC_v3_260825.xlsx"), "1e-15 M")
check_extremes(os.path.join(out_dir, "SAM_QC_v3_260826.xlsx"), "1e-14 M")
check_extremes(os.path.join(out_dir, "SAM_QC_v3_260826.xlsx"), "1e-15 M")
check_extremes(os.path.join(out_dir, "SAM_QC_v3_260826.xlsx"), "1e-13 M")
