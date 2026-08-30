import pandas as pd
import numpy as np
import os
import glob

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"
excel_files = glob.glob(os.path.join(out_dir, "*.xlsx"))

for f in excel_files:
    print(f"\n--- {os.path.basename(f)} ---")
    try:
        xl = pd.ExcelFile(f)
        blank_sheet = None
        for s in xl.sheet_names:
            if "Blank" in s or "0 M" in s:
                blank_sheet = s
                break
        
        thresh = 0
        if blank_sheet:
            df_blank = xl.parse(blank_sheet)
            all_blanks = []
            for col in df_blank.columns:
                all_blanks.extend(df_blank[col].dropna().values)
            if len(all_blanks) > 0:
                thresh = np.mean(all_blanks) + 3 * np.std(all_blanks)
                
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            grid_pcts = []
            for col in df.columns:
                vals = df[col].dropna().values
                if len(vals) > 0:
                    grid_pcts.append(np.sum(vals > thresh) / len(vals) * 100.0)
            
            overall_grid = np.mean(grid_pcts) if grid_pcts else 0
            print(f"Condition: {sheet:10s} | Grid > Thresh: {overall_grid:6.2f}% (from {len(grid_pcts)} FOVs)")
            
    except Exception as e:
        print("Error reading:", e)
