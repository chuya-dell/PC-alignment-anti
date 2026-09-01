import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'
with open(os.path.join(out_dir, 'p50_data.pkl'), 'rb') as f: results = pickle.load(f)
df = pd.DataFrame(results)
filtered_df = df[df['Corr'] >= 0.60]

stats_list = []
for s in range(1, 9):
    s_data = filtered_df[filtered_df['s'] == s]
    if len(s_data) == 0: continue
    all_deltas = np.concatenate(s_data['Deltas'].values)
    n_pillars = len(all_deltas)
    mean_pillar = np.mean(all_deltas)
    std_pillar = np.std(all_deltas)
    sem_pillar = std_pillar / np.sqrt(n_pillars)
    
    fov_means = [np.mean(d) for d in s_data['Deltas'].values]
    n_fovs = len(fov_means)
    std_fov = np.std(fov_means)
    sem_fov = std_fov / np.sqrt(n_fovs)
    
    stats_list.append({
        'Sample': s, 'N_Pillars': n_pillars, 'Mean_%': mean_pillar,
        'Pillar_Std_%': std_pillar, 'Pillar_SEM_%': sem_pillar,
        'N_FOVs': n_fovs, 'FOV_Std_%': std_fov, 'FOV_SEM_%': sem_fov,
        'All_Deltas': all_deltas, 'FOV_Means': fov_means
    })

res_df = pd.DataFrame(stats_list)
blank_row = res_df[res_df['Sample'] == 8].iloc[0]
blank_deltas = blank_row['All_Deltas']
blank_fovs = blank_row['FOV_Means']

for idx, row in res_df.iterrows():
    s = row['Sample']
    if s == 8:
        res_df.at[idx, 'p_val_pillars'] = 1.0
        res_df.at[idx, 'p_val_fovs'] = 1.0
        continue
    _, p_val_p = stats.ttest_ind(row['All_Deltas'], blank_deltas, equal_var=False)
    res_df.at[idx, 'p_val_pillars'] = p_val_p
    if len(row['FOV_Means']) > 1 and len(blank_fovs) > 1:
        _, p_val_f = stats.ttest_ind(row['FOV_Means'], blank_fovs, equal_var=False)
        res_df.at[idx, 'p_val_fovs'] = p_val_f
    else:
        res_df.at[idx, 'p_val_fovs'] = np.nan

print('\n--- A & B. Detailed Stats (Normalization: (Pre - Post) / Pre * 100) ---')
display_df = res_df.drop(columns=['All_Deltas', 'FOV_Means'])
print(display_df.to_string(index=False))

conc_map = {1: -9, 2: -10, 3: -11, 4: -12, 5: -13, 6: -14, 7: -15, 8: None}
res_df['log_conc'] = res_df['Sample'].map(conc_map)
valid_df = res_df.dropna(subset=['log_conc'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(valid_df['log_conc'], valid_df['Mean_%'], yerr=valid_df['Pillar_SEM_%'], fmt='o-', capsize=5, label='Mean ± SEM (n=pillars)', color='blue')
ax.errorbar(valid_df['log_conc'] + 0.1, valid_df['Mean_%'], yerr=valid_df['FOV_SEM_%'], fmt='s', capsize=5, label='Mean ± SEM (n=FOVs)', color='orange', alpha=0.5)

blank_mean = blank_row['Mean_%']
ax.axhline(blank_mean, color='red', linestyle='--', label='Blank (S8) Mean')
ax.axhspan(blank_mean - blank_row['Pillar_SEM_%'], blank_mean + blank_row['Pillar_SEM_%'], color='red', alpha=0.2)

ax.set_xticks(valid_df['log_conc'])
ax.set_xticklabels(['1nM', '100pM', '10pM', '1pM', '100fM', '10fM', '1fM'])
ax.set_ylabel('Mean Delta (%)')
ax.set_xlabel('Concentration')
ax.set_title('Dose Response: Mean (Pre-Post)/Pre * 100')
ax.legend()
plt.grid(True)
out_plot = os.path.join(out_dir, 'p50_mean_dose_response.png')
plt.savefig(out_plot)
print(f'\nSaved plot to {out_plot}')
