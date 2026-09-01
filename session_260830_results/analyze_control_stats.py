import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260830_P50_条件a~d'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

with open(os.path.join(out_dir, 'control_data.pkl'), 'rb') as f: results = pickle.load(f)
df = pd.DataFrame(results)

# Extract timestamps for each file
for idx, row in df.iterrows():
    s = int(row['s'])
    f = int(row['f'])
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if os.path.exists(p_post):
        df.at[idx, 'timestamp'] = os.path.getmtime(p_post)
    else:
        df.at[idx, 'timestamp'] = np.nan

filtered_df = df[df['corr'] >= 0.60].copy()
filtered_df['time_rel'] = filtered_df['timestamp'] - filtered_df['timestamp'].min()

# Substrate level stats
sub_stats = []
for s in range(1, 6):
    s_data = filtered_df[filtered_df['s'] == s]
    deltas = np.concatenate(s_data['delta_array'].values) if len(s_data) > 0 else []
    fov_means = s_data['mean_delta'].values
    corrs = s_data['corr'].values
    times = s_data['time_rel'].values
    sub_stats.append({
        'Substrate': s,
        'Mean_%': np.mean(deltas) if len(deltas)>0 else np.nan,
        'SEM_%': np.std(deltas)/np.sqrt(len(deltas)) if len(deltas)>0 else np.nan,
        'FOV_Means': fov_means,
        'N_FOVs': len(fov_means),
        'Mean_Corr': np.mean(corrs) if len(corrs)>0 else np.nan,
        'Mean_Time': np.mean(times) if len(times)>0 else np.nan
    })

sub_df = pd.DataFrame(sub_stats)
print('\n--- Substrate Level Stats ---')
for _, r in sub_df.iterrows():
    print(f"Substrate {int(r['Substrate'])}: Mean = {r['Mean_%']:.3f}% (SEM: {r['SEM_%']:.4f}%), N_FOVs = {r['N_FOVs']}")
    print(f"  FOV Means: {np.round(r['FOV_Means'], 3)}")

# Conditions
cond_map = {1: '3 (Wash+Blower)', 2: '1 (Incubator)', 3: '2 (Blower)', 4: '3 (Wash+Blower)', 5: '1 (Incubator)'}
filtered_df['Condition'] = filtered_df['s'].map(cond_map)
cond_fovs = {}
print("\n--- Condition Summaries ---")
for cond in ['1 (Incubator)', '2 (Blower)', '3 (Wash+Blower)']:
    c_data = filtered_df[filtered_df['Condition'] == cond]
    cond_fovs[cond] = c_data['mean_delta'].values
    print(f"Condition {cond}: Mean = {np.mean(cond_fovs[cond]):.3f}% (n={len(cond_fovs[cond])} FOVs)")

# Comparisons
print('\n--- Comparisons (Welch t-test on FOV means) ---')
_, p_wash = stats.ttest_ind(cond_fovs['3 (Wash+Blower)'], cond_fovs['2 (Blower)'], equal_var=False)
print(f"Effect of Wash (3 vs 2): diff = {np.mean(cond_fovs['3 (Wash+Blower)']) - np.mean(cond_fovs['2 (Blower)']):.3f}%, p = {p_wash:.4f}")
_, p_blow = stats.ttest_ind(cond_fovs['2 (Blower)'], cond_fovs['1 (Incubator)'], equal_var=False)
print(f"Effect of Blower (2 vs 1): diff = {np.mean(cond_fovs['2 (Blower)']) - np.mean(cond_fovs['1 (Incubator)']):.3f}%, p = {p_blow:.4f}")
_, p_tot = stats.ttest_ind(cond_fovs['3 (Wash+Blower)'], cond_fovs['1 (Incubator)'], equal_var=False)
print(f"Total Effect (3 vs 1): diff = {np.mean(cond_fovs['3 (Wash+Blower)']) - np.mean(cond_fovs['1 (Incubator)']):.3f}%, p = {p_tot:.4f}")

# Replicates
print("\n--- Replicate Consistency ---")
_, p_rep3 = stats.ttest_ind(cond_fovs['3 (Wash+Blower)'][:8], cond_fovs['3 (Wash+Blower)'][8:], equal_var=False)
print(f"Substrate 1 vs 4 (Cond 3): p = {p_rep3:.4f}")
_, p_rep1 = stats.ttest_ind(cond_fovs['1 (Incubator)'][:8], cond_fovs['1 (Incubator)'][8:], equal_var=False)
print(f"Substrate 2 vs 5 (Cond 1): p = {p_rep1:.4f}")

# Time Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(filtered_df['time_rel'] / 60, filtered_df['mean_delta'], c=filtered_df['s'], cmap='Set1', label='FOVs')
ax.set_xlabel('Time since first image (minutes)')
ax.set_ylabel('Mean Delta (%)')
ax.set_title('Control Experiment: Delta vs Time')
plt.grid(True)
out_path = os.path.join(out_dir, 'control_time_drift.png')
plt.savefig(out_path)
print(f"\nSaved time drift plot to {out_path}")
