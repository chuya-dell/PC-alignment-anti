import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

with open(os.path.join(out_dir, 'p50_data.pkl'), 'rb') as f:
    results = pickle.load(f)
df = pd.DataFrame(results)

# Calculate mean delta per FOV
df['mean_delta'] = df['Deltas'].apply(np.mean)

print("--- 1. 視野単位のばらつき (FOV-level variance) ---")
for s in range(1, 9):
    s_data = df[df['s'] == s].copy()
    valid_data = s_data[s_data['Corr'] >= 0.60] # Use filtered for the stats to match previous reports
    fov_means = valid_data['mean_delta'].values
    mean_val = np.mean(fov_means)
    sd_val = np.std(fov_means, ddof=1)
    sem_val = sd_val / np.sqrt(len(fov_means))
    print(f"S{s}: Mean={mean_val:.3f}%, SD={sd_val:.3f}%, SEM(n={len(fov_means)})={sem_val:.3f}%")
    for _, r in s_data.iterrows():
        status = " (Dropped)" if r['Corr'] < 0.60 else ""
        print(f"  F{int(r['f'])}: {r['mean_delta']:.3f}% [Corr: {r['Corr']:.4f}]{status}")

print("\n--- 2. 外れ視野の抽出 (Mean ± 2SD over all 64 FOVs) ---")
all_means = df['mean_delta'].values
global_mean = np.mean(all_means)
global_sd = np.std(all_means, ddof=1)
lower_bound = global_mean - 2 * global_sd
upper_bound = global_mean + 2 * global_sd
print(f"Global Mean = {global_mean:.3f}%, Global SD = {global_sd:.3f}%")
print(f"Normal Range (±2SD) = {lower_bound:.3f}% to {upper_bound:.3f}%")
outliers = df[(df['mean_delta'] < lower_bound) | (df['mean_delta'] > upper_bound)]
if len(outliers) == 0:
    print("  No FOVs outside ±2SD.")
for _, r in outliers.iterrows():
    print(f"  Outlier: S{int(r['s'])} F{int(r['f'])} = {r['mean_delta']:.3f}% (Corr: {r['Corr']:.4f})")

print("\n--- 3. S6の精査 (Detailed look at S6) ---")
s6_data = df[df['s'] == 6]
for _, r in s6_data.iterrows():
    print(f"  S6 F{int(r['f'])}: Delta={r['mean_delta']:.3f}%, Corr={r['Corr']:.4f}")

print("\n--- 4. 相関値と差分平均の関係 (Corr vs Delta) ---")
slope, intercept, r_val, p_val, std_err = stats.linregress(df['Corr'], df['mean_delta'])
print(f"Linear Regression (All 64 FOVs): slope={slope:.3f}, R^2={r_val**2:.3f}, p-value={p_val:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df['Corr'], df['mean_delta'], c=df['s'], cmap='jet', alpha=0.7)
cbar = plt.colorbar(scatter)
cbar.set_label('Sample (S1 to S8)')
ax.plot(df['Corr'], intercept + slope * df['Corr'], 'r--', label=f'Trend (p={p_val:.4f})')
ax.set_xlabel('Alignment Correlation (r)')
ax.set_ylabel('Mean Delta (%)')
ax.set_title('Delta vs Alignment Correlation (All 64 FOVs)')
ax.legend()
plt.grid(True)
out_plot = os.path.join(out_dir, 'p50_corr_vs_delta.png')
plt.savefig(out_plot)
print(f"\nSaved scatter plot to {out_plot}")
