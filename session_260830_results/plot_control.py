import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'
with open(os.path.join(out_dir, 'drift_results_control.pkl'), 'rb') as f:
    df = pickle.load(f)

cond_map = {1: 3, 2: 1, 3: 2, 4: 3, 5: 1} # 3=Wash, 1=Inc, 2=Blow
df['cond'] = df['s'].map(cond_map)
df['time_rel'] = (df['timestamp'] - df['timestamp'].min()) / 60

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = {1: 'blue', 2: 'green', 3: 'red'}
labels = {1: 'Cond 1 (Inc)', 2: 'Cond 2 (Blow)', 3: 'Cond 3 (Wash)'}

for c in [1, 2, 3]:
    c_df = df[df['cond'] == c]
    if len(c_df) == 0: continue
    ax1.scatter(c_df['time_rel'], c_df['pillar_delta'], c=colors[c], marker='o', label=f'Pillar {labels[c]}')
    ax1.scatter(c_df['time_rel'], c_df['bg_delta'], c=colors[c], marker='x', alpha=0.5, label=f'BG {labels[c]}')
    ax2.scatter(c_df['time_rel'], c_df['fwhm'], c=colors[c], marker='o', label=labels[c])

ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Mean Delta (%)')
ax1.set_title('Control: Delta vs Time (Color = Condition)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('FWHM (px)')
ax2.set_title('Control: FWHM vs Time')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'control_drift_by_time.png'))

with open(os.path.join(out_dir, 'p100_1_results.pkl'), 'rb') as f:
    p100 = pd.DataFrame(pickle.load(f))
for s in [1, 8]:
    sdf = p100[p100['s']==s]
    print(f"S{s}: total FOVs={len(sdf)}, valid(Corr>=0.6)={len(sdf[sdf['corr']>=0.60])}, mean corr={sdf['corr'].mean():.3f}")

print("\nPairwise Time Trends:")
for label, s_pair in [("Cond 3 (Sub 1->4)", [1, 4]), ("Cond 1 (Sub 2->5)", [2, 5])]:
    pdf = df[df['s'].isin(s_pair)]
    sp, _, _, p_p, _ = stats.linregress(pdf['time_rel'], pdf['pillar_delta'])
    sb, _, _, p_b, _ = stats.linregress(pdf['time_rel'], pdf['bg_delta'])
    sf, _, _, p_f, _ = stats.linregress(pdf['time_rel'], pdf['fwhm'])
    print(f"{label}:")
    print(f"  Pillar Trend: {sp:.4f}/min (p={p_p:.4f})")
    print(f"  BG Trend:     {sb:.4f}/min (p={p_b:.4f})")
    print(f"  FWHM Trend:   {sf:.4f}/min (p={p_f:.4f})")
