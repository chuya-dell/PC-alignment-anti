import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'
with open(os.path.join(out_dir, 'p50_data.pkl'), 'rb') as f: results = pickle.load(f)

df = pd.DataFrame(results)

# 1. Filter out low correlation FOVs
median_corr = df['Corr'].median()
std_corr = df['Corr'].std()
thresh_corr = max(0.60, median_corr - 3 * std_corr)

print('--- Dropped FOVs (Alignment Failure) ---')
dropped = df[df['Corr'] < thresh_corr]
if len(dropped) == 0: print('None')
for _, r in dropped.iterrows():
    print(f"Dropped: S{int(r['s'])} F{int(r['f'])} (Corr: {r['Corr']:.4f})")

filtered_df = df[df['Corr'] >= thresh_corr]

# 2. Blank Stats (Sample 8)
blank_deltas = np.concatenate(filtered_df[filtered_df['s'] == 8]['Deltas'].values)
blank_mean = np.mean(blank_deltas)
blank_std = np.std(blank_deltas)
thresh_val = blank_mean + 3 * blank_std
print('\n--- Blank (S8) Statistics ---')
print(f'Blank Mean: {blank_mean:.4f}%')
print(f'Blank Std (σ): {blank_std:.4f}% (Expected ~0.229%)')
print(f'Threshold (Mean + 3σ): {thresh_val:.4f}%')

# 3. Exceedance Rates
rates = []
for _, r in filtered_df.iterrows():
    deltas = r['Deltas']
    rate = np.sum(deltas > thresh_val) / len(deltas) * 100
    rates.append({'Sample': int(r['s']), 'FOV': int(r['f']), 'Rate': rate})

res_df = pd.DataFrame(rates)
conc_map = {1: -9, 2: -10, 3: -11, 4: -12, 5: -13, 6: -14, 7: -15, 8: None}
res_df['log_conc'] = res_df['Sample'].map(conc_map)

summary = res_df.groupby('Sample').agg({'Rate': ['mean', 'sem', 'std']})
summary.columns = ['mean', 'sem', 'std']
print('\n--- Exceedance Rate Summary ---')
print(summary.round(3))

# 4. Stats
valid_df = res_df.dropna(subset=['log_conc'])
slope, intercept, r_val, p_val, std_err = stats.linregress(valid_df['log_conc'], valid_df['Rate'])
print(f'\nTrend Analysis (Linear Regression, S1-S7):')
print(f'Slope = {slope:.4f}, p-value = {p_val:.4f}')

groups = [res_df[res_df['Sample'] == s]['Rate'].values for s in range(1, 8)]
f_stat, anova_p = stats.f_oneway(*groups)
print(f'ANOVA (S1-S7): p-value = {anova_p:.4f}')

# 5. Plotting
fig, ax = plt.subplots(figsize=(10, 6))
sam_summary = summary.loc[1:7]
ax.errorbar(sam_summary.index, sam_summary['mean'], yerr=sam_summary['sem'], fmt='o-', color='blue', label='S1-S7 (SAM)', capsize=5)

ax.scatter(valid_df['Sample'], valid_df['Rate'], color='lightblue', alpha=0.6, zorder=2)

blank_mean_rate = summary.loc[8, 'mean']
blank_sem = summary.loc[8, 'sem']
ax.axhline(blank_mean_rate, color='red', linestyle='--', label=f'Blank (S8) Mean: {blank_mean_rate:.2f}%')
ax.axhspan(blank_mean_rate - blank_sem, blank_mean_rate + blank_sem, color='red', alpha=0.2)

ax.set_xticks(range(1, 9))
ax.set_xticklabels(['1nM(S1)', '100pM(S2)', '10pM(S3)', '1pM(S4)', '100fM(S5)', '10fM(S6)', '1fM(S7)', 'Blank(S8)'])
ax.set_ylabel('Threshold Exceedance Rate (%)')
ax.set_xlabel('Concentration')
ax.set_title('Concentration Dependence (p50 dataset)\nThreshold = Blank Mean + 3σ')
ax.legend()
plt.grid(True)
out_path = os.path.join(out_dir, 'p50_dose_response_v3.png')
plt.savefig(out_path)
print(f'\nSaved plot to {out_path}')
