import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\enriched_correlation.csv")
# Sample 1 = 1nM (10^-9), Sample 2 = 10^-10 ... Sample 7 = 10^-15, Sample 8 = Blank (0)
def get_log_conc(s):
    if s == 8:
        return np.nan # Blank
    else:
        return -9 - (s - 1)

df['log_conc'] = df['sample'].apply(get_log_conc)

print("--- 1. 濃度との統計的関係 ---")
# ANOVA across all groups
groups = [df[df['sample'] == s]['corr'].values for s in range(1, 9)]
f_stat, p_val = stats.f_oneway(*groups)
print("ANOVA p-value (All Samples):", p_val)

# Regression against log_conc (excluding blank)
df_valid = df.dropna(subset=['log_conc'])
slope, intercept, r_value, p_value, std_err = stats.linregress(df_valid['log_conc'], df_valid['corr'])
print(f"Log-Concentration Regression (S1-S7): p-value = {p_value:.4f}, R-squared = {r_value**2:.4f}, slope = {slope:.4f}")

print("\n--- 2. 各種指標との相関 (交絡要因の検証) ---")
# Correlation of 'corr' with 'snr', 'ecc_score', 'defect_area'
for col in ['snr', 'ecc_score', 'defect_area']:
    r, p = stats.pearsonr(df[col].fillna(0), df['corr']) # fillna just in case ecc_score is nan
    print(f"{col}: r = {r:.4f}, p-value = {p:.4f}")

# Generate a Summary Table
print("\n--- 更新版テーブル ---")
summary = df.groupby('sample').agg({'corr': ['mean', 'std', 'min', 'max']}).reset_index()
summary.columns = ['Sample', 'Mean', 'Std', 'Min', 'Max']
summary['Concentration'] = ["1 nM", "100 pM", "10 pM", "1 pM", "100 fM", "10 fM", "1 fM", "Blank"]
summary = summary[['Sample', 'Concentration', 'Mean', 'Std', 'Min', 'Max']]
print(summary.to_string(index=False))
