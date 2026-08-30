import pandas as pd
import numpy as np
import os

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"

def check_mask_pass_rate(qc_file, raw_file):
    if not os.path.exists(qc_file) or not os.path.exists(raw_file):
        print(f"File missing for {qc_file}")
        return
        
    xl_qc = pd.ExcelFile(qc_file)
    xl_raw = pd.ExcelFile(raw_file)
    
    total_valid = 0
    total_raw = 0
    
    for sheet in xl_qc.sheet_names:
        df_qc = xl_qc.parse(sheet)
        for col in df_qc.columns:
            valid_vals = df_qc[col].dropna().values
            total_valid += len(valid_vals)
            
    for sheet in xl_raw.sheet_names:
        df_raw = xl_raw.parse(sheet)
        for col in df_raw.columns:
            raw_vals = df_raw[col].dropna().values
            total_raw += len(raw_vals)
            
    rate = (total_valid / total_raw) * 100.0 if total_raw > 0 else 0
    print(f"[{os.path.basename(qc_file)}] Valid: {total_valid:7d} / Raw: {total_raw:7d} ({rate:5.2f}%)")

print("--- Mask Pass Rate Comparison ---")
check_mask_pass_rate(os.path.join(out_dir, "SAM_QC_v3_260706_p200_negative.xlsx"), 
                     os.path.join(out_dir, "SAM_QC_v3_260706_p200.xlsx")) # Raw Excel didn't have "_negative" but they are identical structurally.
                     
check_mask_pass_rate(os.path.join(out_dir, "SAM_QC_v3_260826_negative.xlsx"),
                     os.path.join(out_dir, "SAM_QC_v3_260826.xlsx"))

check_mask_pass_rate(os.path.join(out_dir, "DNA_QC_v3_260828_negative.xlsx"),
                     os.path.join(out_dir, "DNA_QC_v3_260828.xlsx"))

check_mask_pass_rate(os.path.join(out_dir, "SAM_QC_v3_260824_negative.xlsx"),
                     os.path.join(out_dir, "SAM_QC_v3_260824.xlsx"))
