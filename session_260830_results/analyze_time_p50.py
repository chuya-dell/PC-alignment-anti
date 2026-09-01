import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

with open(os.path.join(out_dir, 'p50_data.pkl'), 'rb') as f: results = pickle.load(f)
df = pd.DataFrame(results)

# Extract timestamps
for idx, row in df.iterrows():
    s = int(row['s'])
    f = int(row['f'])
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if os.path.exists(p_post):
        df.at[idx, 'timestamp'] = os.path.getmtime(p_post)
    else:
        df.at[idx, 'timestamp'] = np.nan

filtered_df = df[df['Corr'] >= 0.60].copy()
filtered_df['time_rel'] = filtered_df['timestamp'] - filtered_df['timestamp'].min()
filtered_df['mean_delta'] = filtered_df['Deltas'].apply(np.mean)

# Plot Time Drift
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(filtered_df['time_rel'] / 60, filtered_df['mean_delta'], c=filtered_df['s'], cmap='jet', label='FOVs')
cbar = plt.colorbar(scatter)
cbar.set_label('Sample Number (1=1nM -> 8=Blank)')
ax.set_xlabel('Time since first image (minutes)')
ax.set_ylabel('Mean Delta (%)')
ax.set_title('260828-p50: Delta vs Imaging Time')

# Regression on time
slope, intercept, r_val, p_val, std_err = stats.linregress(filtered_df['time_rel'], filtered_df['mean_delta'])
ax.plot(filtered_df['time_rel'] / 60, intercept + slope * filtered_df['time_rel'], 'r--', label=f'Trend (p={p_val:.4f})')

plt.grid(True)
plt.legend()
out_path = os.path.join(out_dir, 'p50_time_drift.png')
plt.savefig(out_path)
print(f"Saved p50 time drift plot to {out_path}")
print(f"Time regression: slope={slope*60:.4f} %/min, p={p_val:.4f}")
