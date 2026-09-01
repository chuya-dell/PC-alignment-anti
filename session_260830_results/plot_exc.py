import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'
with open(os.path.join(out_dir, 'p50_data.pkl'), 'rb') as f:
    df = pd.DataFrame(pickle.load(f))

# Filter bad FOVs
df = df[df['Corr'] >= 0.70]

# Calculate S8 stats
s8_data = df[df['s']==8]
s8_deltas = np.concatenate(s8_data['Deltas'].values)
b_mean, b_std = np.mean(s8_deltas), np.std(s8_deltas)
th_pos = b_mean + 3*b_std
th_neg = b_mean - 3*b_std

conc_map = {1:-9, 2:-10, 3:-11, 4:-12, 5:-13, 6:-14, 7:-15, 8:-16}

results = []
for s in range(1, 9):
    s_data = df[df['s']==s]
    if len(s_data) == 0: continue
    deltas = np.concatenate(s_data['Deltas'].values)
    exc_pos = np.sum(deltas > th_pos) / len(deltas) * 100
    exc_neg = np.sum(deltas < th_neg) / len(deltas) * 100
    results.append({'s':s, 'log_c':conc_map[s], 'pos':exc_pos, 'neg':exc_neg})

res_df = pd.DataFrame(results)

plt.figure(figsize=(10,6))
plt.plot(res_df['log_c'], res_df['pos'], marker='o', color='red', label='> +3 SD (Darker)')
plt.plot(res_df['log_c'], res_df['neg'], marker='o', color='blue', label='< -3 SD (Brighter)')

labels = ['1nM(S1)','100pM(S2)','10pM(S3)','1pM(S4)','100fM(S5)','10fM(S6)','1fM(S7)','Blank(S8)']
plt.xticks(res_df['log_c'], labels, rotation=45)
plt.xlabel('Concentration')
plt.ylabel('Threshold Exceedance Rate (%)')
plt.title('Threshold Exceedance Rate vs Concentration (260828-p50, Corr>=0.70)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'exceedance_plot_p50.png'))
print('Plot saved as exceedance_plot_p50.png')
