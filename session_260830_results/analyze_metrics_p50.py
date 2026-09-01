import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import cv2

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

with open(os.path.join(out_dir, 'p50_data.pkl'), 'rb') as f:
    results = pickle.load(f)

for res in results:
    s, f = res['s'], res['f']
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    d0 = np.nan_to_num(img0, nan=np.nanmean(img0))
    diff0 = d0 - cv2.GaussianBlur(d0, (51, 51), 0)
    std0 = np.std(diff0[~np.isnan(img0)])
    m0 = cv2.dilate((np.abs(diff0) > 3*std0).astype(np.uint8), np.ones((7,7), np.uint8)).astype(bool)
    res['defect_area'] = np.sum(m0) / m0.size * 100
    res['snr'] = np.mean(img0) / np.std(img0) # Inverse CV as SNR
    res['mean_delta'] = np.mean(res['Deltas'])

df = pd.DataFrame(results)

print('--- 1. Regression vs Mean Delta ---')
for m in ['Corr', 'snr', 'defect_area']:
    s_val, i_val, r_val, p_val, _ = stats.linregress(df[m], df['mean_delta'])
    print(f'{m}: r={r_val:.3f}, p={p_val:.4f}')

print('\n--- 2. Full FOV Table ---')
print('Sample,FOV,Delta(%),Corr,SNR,DefectArea(%)')
for _, r in df.sort_values(['s', 'f']).iterrows():
    print(f"S{int(r['s'])},F{int(r['f'])},{r['mean_delta']:.3f},{r['Corr']:.4f},{r['snr']:.2f},{r['defect_area']:.3f}")

print('\n--- 3. Sensitivity Analysis ---')
df['log_conc'] = df['s'].map({1: -9, 2: -10, 3: -11, 4: -12, 5: -13, 6: -14, 7: -15, 8: None})
for th in [0.60, 0.70, 0.80, 0.85]:
    v_df = df[df['Corr'] >= th]
    dropped = len(df) - len(v_df)
    print(f'\nThreshold Corr >= {th} (Dropped {dropped})')
    for s in range(1, 9):
        s_data = v_df[v_df['s'] == s]
        if len(s_data) == 0: print(f'  S{s}: All dropped'); continue
        mean_val = np.mean(s_data['mean_delta'])
        sem_val = np.std(s_data['mean_delta'], ddof=1)/np.sqrt(len(s_data)) if len(s_data)>1 else 0
        print(f'  S{s}: {mean_val:.3f}% ± {sem_val:.3f}% (n={len(s_data)})')
    v_conc = v_df.dropna(subset=['log_conc'])
    if len(v_conc) > 2:
        s_c, i_c, r_c, p_c, _ = stats.linregress(v_conc['log_conc'], v_conc['mean_delta'])
        print(f'  Regression S1-S7: slope={s_c:.4f}, p={p_c:.4f}')
