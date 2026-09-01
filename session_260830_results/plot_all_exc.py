import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

ds_configs = [
    ('260826-p50-sam', 'aug', 'aug_arrays.pkl'),
    ('260828-p50-SAM', 'p50', 'p50_data.pkl'),
    ('260829-p50-sam', 'aug', 'aug_arrays.pkl'),
    ('p200', 'july', 'p200_results.pkl'),
    ('p100_1', 'july', 'p100_1_results.pkl'),
    ('p100_2', 'july', 'p100_2_results.pkl')
]

conc_map = {1:-9, 2:-10, 3:-11, 4:-12, 5:-13, 6:-14, 7:-15, 8:-16}
labels = ['1nM','100pM','10pM','1pM','100fM','10fM','1fM','Blank']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Preload
data_cache = {}

for idx, (ds_name, ds_type, pkl_file) in enumerate(ds_configs):
    ax = axes[idx]
    
    if pkl_file not in data_cache:
        path = os.path.join(out_dir, pkl_file)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data_cache[pkl_file] = pickle.load(f)
        else:
            data_cache[pkl_file] = None
            
    raw_data = data_cache[pkl_file]
    if raw_data is None:
        ax.set_title(f"{ds_name} (Not Found)")
        continue
        
    df = pd.DataFrame(raw_data)
    
    # Unify schema
    if ds_type == 'aug':
        df = df[df['dataset'] == ds_name]
        df = df[df['corr'] >= 0.70]
        array_col = 'deltas'
    elif ds_type == 'p50':
        df = df[df['Corr'] >= 0.70]
        array_col = 'Deltas'
    elif ds_type == 'july':
        df = df[df['corr'] >= 0.60]
        array_col = 'delta_array'
        
    s8_data = df[df['s']==8]
    if len(s8_data) == 0:
        ax.set_title(f"{ds_name} (No S8 Data)")
        continue
        
    s8_deltas = np.concatenate(s8_data[array_col].values)
    b_mean, b_std = np.mean(s8_deltas), np.std(s8_deltas)
    th_pos = b_mean + 3*b_std
    th_neg = b_mean - 3*b_std
    
    results = []
    for s in range(1, 9):
        s_data = df[df['s']==s]
        if len(s_data) == 0: continue
        deltas = np.concatenate(s_data[array_col].values)
        exc_pos = np.sum(deltas > th_pos) / len(deltas) * 100
        exc_neg = np.sum(deltas < th_neg) / len(deltas) * 100
        results.append({'s':s, 'log_c':conc_map[s], 'pos':exc_pos, 'neg':exc_neg})
        
    if not results: continue
    res_df = pd.DataFrame(results)
    
    ax.plot(res_df['log_c'], res_df['pos'], marker='o', color='red', label='> +3 SD (Darker)')
    ax.plot(res_df['log_c'], res_df['neg'], marker='o', color='blue', label='< -3 SD (Brighter)')
    
    ax.set_xticks([-9,-10,-11,-12,-13,-14,-15,-16])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(f'{ds_name}')
    ax.grid(True)
    if idx == 0:
        ax.legend()
        ax.set_ylabel('Exceedance Rate (%)')
    if idx == 3:
        ax.set_ylabel('Exceedance Rate (%)')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'exceedance_all.png'))
print("Saved to exceedance_all.png")
