import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572"

# -------------------------------------------------------------
# Priority 2: Plot Histogram for SAM_260706_p200 (1nM vs Blank)
# -------------------------------------------------------------
def plot_histogram():
    f = os.path.join(out_dir, "SAM_QC_v3_260706_p200_negative.xlsx")
    if not os.path.exists(f): return
    
    xl = pd.ExcelFile(f)
    
    # Extract Blank pillars
    df_blank = xl.parse("0 M(ブランク)") if "0 M(ブランク)" in xl.sheet_names else xl.parse("Blank")
    blank_vals = []
    for col in df_blank.columns:
        blank_vals.extend(df_blank[col].dropna().values)
    blank_vals = np.array(blank_vals)
    
    # Extract 1nM pillars
    df_1nm = xl.parse("1e-09 M")
    nm_vals = []
    for col in df_1nm.columns:
        nm_vals.extend(df_1nm[col].dropna().values)
    nm_vals = np.array(nm_vals)
    
    thresh = np.mean(blank_vals) - 3 * np.std(blank_vals)
    
    plt.figure(figsize=(10, 6))
    plt.hist(blank_vals, bins=100, density=True, color='blue', alpha=0.5, label='Blank')
    plt.hist(nm_vals, bins=100, density=True, color='red', alpha=0.5, label='1 nM')
    plt.axvline(thresh, color='black', linestyle='--', label=f'Threshold (-3σ): {thresh:.3f}%')
    plt.axvline(np.mean(blank_vals), color='blue', linestyle=':')
    plt.axvline(np.mean(nm_vals), color='red', linestyle=':')
    
    plt.title("Pillar Delta Intensity Distribution: 260706 p200 (1nM vs Blank)")
    plt.xlabel("Delta Intensity (%)")
    plt.ylabel("Density")
    plt.xlim(-1.5, 1.5)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(out_dir, "Histogram_260706_p200_1nM_vs_Blank.png")
    plt.savefig(out_png, dpi=300)
    print(f"Saved histogram to {out_png}")

# -------------------------------------------------------------
# Priority 1: Diagnose SAM_260826 Blank SD
# -------------------------------------------------------------
def diagnose_260826():
    f = os.path.join(out_dir, "SAM_QC_v3_260826_negative.xlsx")
    if not os.path.exists(f): return
    
    xl = pd.ExcelFile(f)
    df_blank = xl.parse("0 M(ブランク)") if "0 M(ブランク)" in xl.sheet_names else xl.parse("Blank")
    
    print("\n--- Diagnosing 260826 Blank FOVs ---")
    all_vals = []
    for col in df_blank.columns:
        vals = df_blank[col].dropna().values
        if len(vals) > 0:
            print(f"FOV {col:15s} | n={len(vals):4d} | mean={np.mean(vals):6.3f}% | std={np.std(vals):6.3f}%")
            all_vals.extend(vals)
            
    all_vals = np.array(all_vals)
    print(f"\nOverall Blank -> n={len(all_vals)}, Mean={np.mean(all_vals):.3f}%, SD={np.std(all_vals):.3f}%")
    
    # Also diagnose another day for comparison, e.g. 260828 SAM
    f2 = os.path.join(out_dir, "SAM_QC_v3_260828_negative.xlsx")
    if os.path.exists(f2):
        xl2 = pd.ExcelFile(f2)
        df_blank2 = xl2.parse("Blank") if "Blank" in xl2.sheet_names else xl2.parse("0 M(ブランク)")
        all_vals2 = []
        for col in df_blank2.columns:
            all_vals2.extend(df_blank2[col].dropna().values)
        all_vals2 = np.array(all_vals2)
        print(f"Comparison 260828 Blank -> n={len(all_vals2)}, Mean={np.mean(all_vals2):.3f}%, SD={np.std(all_vals2):.3f}%")

if __name__ == "__main__":
    plot_histogram()
    diagnose_260826()
