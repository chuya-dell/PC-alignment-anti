import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

# --- Part A: Threshold Sensitivity (p50) ---
print("=== A. Threshold Sensitivity Analysis (p50) ===")
with open(os.path.join(out_dir, 'p50_data.pkl'), 'rb') as f:
    p50 = pd.DataFrame(pickle.load(f))
p50['mean_delta'] = p50['Deltas'].apply(np.mean)
p50['log_conc'] = p50['s'].map({1: -9, 2: -10, 3: -11, 4: -12, 5: -13, 6: -14, 7: -15, 8: None})

print("| Threshold | Dropped FOVs | S1 Mean (n) | S8 Mean (n) | p-value |")
print("| :--- | :--- | :--- | :--- | :--- |")
for th in np.arange(0.60, 0.95, 0.05):
    v = p50[p50['Corr'] >= th]
    dropped = len(p50) - len(v)
    s1 = v[v['s']==1]['mean_delta']
    s8 = v[v['s']==8]['mean_delta']
    v_conc = v.dropna(subset=['log_conc'])
    if len(v_conc) > 2:
        _, _, _, p_val, _ = stats.linregress(v_conc['log_conc'], v_conc['mean_delta'])
        p_str = f"{p_val:.4f}"
    else:
        p_str = "N/A"
    s1_str = f"{s1.mean():.3f}% ({len(s1)})" if len(s1)>0 else "N/A"
    s8_str = f"{s8.mean():.3f}% ({len(s8)})" if len(s8)>0 else "N/A"
    print(f"| Corr >= {th:.2f} | {dropped} | {s1_str} | {s8_str} | {p_str} |")

# --- Part B & C: July Data Neg Grid % & Timestamps ---
print("\n=== B & C. July Data Exceedance Rate & Timestamps ===")
for name in ['p200', 'p100_1', 'p100_2']:
    path = os.path.join(out_dir, f'{name}_results.pkl')
    if not os.path.exists(path): continue
    with open(path, 'rb') as f:
        df = pd.DataFrame(pickle.load(f))
    df = df[df['corr'] >= 0.60]
    
    # Check timestamps
    times = df.groupby('s')['timestamp'].min().sort_index()
    is_monotonic = times.is_monotonic_increasing or times.is_monotonic_decreasing
    print(f"\n[{name}]")
    print(f"  Sequential Processing (S1-S8): {is_monotonic} (Times: {[pd.to_datetime(t, unit='s').strftime('%H:%M:%S') for t in times]})")
    
    # Calculate Exceedance Rate
    s8_data = df[df['s']==8]
    if len(s8_data) > 0:
        s8_deltas = np.concatenate(s8_data['delta_array'].values)
        b_mean, b_std = np.mean(s8_deltas), np.std(s8_deltas)
        th_pos = b_mean + 3*b_std
        th_neg = b_mean - 3*b_std
        print(f"  Blank Stats: Mean={b_mean:.3f}%, SD={b_std:.3f}%. Thresholds: >{th_pos:.3f}% or <{th_neg:.3f}%")
        
        for s in range(1, 9):
            s_data = df[df['s']==s]
            if len(s_data) == 0: continue
            deltas = np.concatenate(s_data['delta_array'].values)
            exc_pos = np.sum(deltas > th_pos) / len(deltas) * 100
            exc_neg = np.sum(deltas < th_neg) / len(deltas) * 100
            if s == 1 or s == 8:
                print(f"  S{s}: >+3SD = {exc_pos:.3f}%, <-3SD = {exc_neg:.3f}%")

# --- Part D: p50 Background, Pillar, FWHM ---
print("\n=== D. p50 Drift Analysis ===")
with open(os.path.join(out_dir, 'drift_results.pkl'), 'rb') as f:
    drift = pickle.load(f)

print("Sample Averages (Filtered Corr >= 0.70):")
print("Sample | Pillar Delta (%) | BG Delta (%) | FWHM (px)")
for s in range(1, 9):
    s_df = drift[drift['s'] == s]
    if len(s_df) > 0:
        print(f"S{s} | {s_df['pillar_delta'].mean():.3f} | {s_df['bg_delta'].mean():.3f} | {s_df['fwhm'].mean():.3f}")

print("\nIntra-Sample (FOV 1->8) Averages:")
for f_idx in range(1, 9):
    f_df = drift[drift['f'] == f_idx]
    if len(f_df) > 0:
        print(f"FOV {f_idx} | {f_df['pillar_delta'].mean():.3f} | {f_df['bg_delta'].mean():.3f} | {f_df['fwhm'].mean():.3f}")
