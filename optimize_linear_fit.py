import os
import re
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Dataset-specific concentration mappings
MAPPING_P100_1 = {
    '10': ('0 M (Blank)', 0.0), '11': ('0 M (Blank 2)', 0.0),
    '7':  ('1 fM', 1e-15), '9':  ('1 fM (Dup)', 1e-15),
    '12': ('10 fM', 1e-14), '5':  ('100 fM', 1e-13),
    '4':  ('1 pM', 1e-12),  '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10), '1':  ('1 nM', 1e-9)
}
MAPPING_P100_2 = {
    '8':  ('0 M (Blank)', 0.0), '9':  ('1 fM', 1e-15),
    '6':  ('10 fM', 1e-14), '12': ('10 fM (Dup)', 1e-14),
    '5':  ('100 fM', 1e-13), '11': ('100 fM (Dup)', 1e-13),
    '4':  ('1 pM', 1e-12),  '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10), '1':  ('1 nM', 1e-9)
}
MAPPING_P200 = {
    '13': ('0 M (Blank)', 0.0), '6':  ('1 fM', 1e-15),
    '10': ('10 fM', 1e-14), '9':  ('100 fM', 1e-13),
    '11': ('1 pM', 1e-12),  '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10), '1':  ('1 nM', 1e-9)
}

EXCLUDE_SERIES = {
    '260706_sam_p200': ['8'],
    '260707_sam_p100_1': ['8', '6'],
    '260707_sam_p100_2': ['7']
}

def hill_equation(x, vmax, kd, n):
    return vmax * (x**n) / (kd**n + x**n)

def run_advanced_optimization(input_dir, dataset_name, mapping):
    aligned_csvs = sorted(glob.glob(os.path.join(input_dir, "*_aligned.csv")))
    estimated_csvs = sorted(glob.glob(os.path.join(input_dir, "*_estimated_grid.csv")))
    
    if not aligned_csvs:
        return None
        
    print(f"\n=======================================================")
    print(f" [開始] {dataset_name} 高度物理数学モデル自動探索")
    print(f"=======================================================\n")
    
    ex_list = EXCLUDE_SERIES.get(dataset_name, [])
    
    bg_vals = []
    bg_files = [f for f in aligned_csvs if os.path.basename(f).startswith(("10-", "11-", "8-", "13-"))]
    for bgf in bg_files:
        series = re.split(r'[_\-\.]', os.path.basename(bgf))[0]
        if series in ex_list: continue
        df_bg = pd.read_csv(bgf)
        bg_vals.extend(df_bg['mean_intensity'].dropna().values)
        
    bg_arr = np.array(bg_vals)
    bg_mean = np.mean(bg_arr)
    bg_std = np.std(bg_arr)
    
    data_cache = []
    all_csvs = sorted(aligned_csvs + estimated_csvs)
    
    for cpath in all_csvs:
        fname = os.path.basename(cpath)
        series = re.split(r'[_\-\.]', fname)[0]
        if series in ex_list: continue
            
        pattern = 'Pattern A' if 'aligned' in fname else 'Pattern B'
        df = pd.read_csv(cpath)
        vals = df['mean_intensity'].dropna().values
        
        if len(vals) > 0:
            conc_info = mapping.get(str(series), ('Unknown', None))
            data_cache.append({
                'fname': fname, 'series': series, 'pattern': pattern,
                'vals': vals, 'conc_M': conc_info[1]
            })
            
    # Advanced Grid Space
    sigmas = [3.0, 5.0, 8.0]
    z_cutoffs = [2.8, 3.5, None]
    patterns = ['Pattern A', 'Pattern B']
    # 4 New metrics added!
    metrics = ['ratio', 'delta_mean', 'delta_median', 'trimmed_mean_5pct', 'snr_norm']
    relu_filters = [False, True] # New physical constraint filter
    
    results = []
    
    for pat in patterns:
        for sig in sigmas:
            thresh = bg_mean + sig * bg_std
            for z_cut in z_cutoffs:
                for met in metrics:
                    for use_relu in relu_filters:
                        sample_list = []
                        for item in data_cache:
                            if item['pattern'] != pat or item['conc_M'] is None: continue
                            
                            vals = item['vals']
                            if use_relu:
                                # 物理的非負フィルター (Negative delta means noise)
                                vals = vals[vals >= bg_mean]
                                
                            n_total = len(vals)
                            if n_total == 0: continue
                                
                            if met == 'ratio':
                                val = (np.sum(vals > thresh) / n_total) * 100
                            elif met == 'delta_mean':
                                val = np.mean(vals) - bg_mean
                            elif met == 'delta_median':
                                val = np.median(vals) - bg_mean
                            elif met == 'trimmed_mean_5pct':
                                # 上下5%トリム平均 (Robust metric)
                                p5 = np.percentile(vals, 5)
                                p95 = np.percentile(vals, 95)
                                trim_vals = vals[(vals >= p5) & (vals <= p95)]
                                val = np.mean(trim_vals) - bg_mean if len(trim_vals)>0 else 0
                            elif met == 'snr_norm':
                                # SNR規格化 (Signal to Noise Ratio)
                                val = (np.mean(vals) - bg_mean) / bg_std
                                
                            sample_list.append({
                                'fname': item['fname'], 'series': item['series'],
                                'conc_M': item['conc_M'], 'val': val
                            })
                            
                        df_run = pd.DataFrame(sample_list)
                        if len(df_run) == 0: continue
                            
                        if z_cut is not None:
                            clean_rows = []
                            for s, grp in df_run.groupby('series'):
                                if len(grp) >= 3:
                                    mu_g = grp['val'].mean()
                                    std_g = grp['val'].std()
                                    for _, r in grp.iterrows():
                                        if abs(r['val'] - mu_g) / (std_g + 1e-9) <= z_cut:
                                            clean_rows.append(r)
                                else:
                                    clean_rows.extend([r for _, r in grp.iterrows()])
                            df_run = pd.DataFrame(clean_rows)
                            
                        g_conc = df_run.groupby('conc_M')['val'].mean().reset_index()
                        g_log = g_conc[g_conc['conc_M'] > 0].copy()
                        
                        if len(g_log) >= 4:
                            log_x = np.log10(g_log['conc_M'].values)
                            y_vals = g_log['val'].values
                            
                            slope, intercept, r_val, p_val, std_err = stats.linregress(log_x, y_vals)
                            r2 = r_val ** 2
                            
                            # Fit Hill Equation (Sigmoidal)
                            hill_r2 = 0
                            try:
                                popt, _ = curve_fit(hill_equation, g_log['conc_M'].values, y_vals, p0=[max(y_vals), 1e-12, 1], bounds=(0, [np.inf, 1e-6, 5]), maxfev=2000)
                                y_pred_hill = hill_equation(g_log['conc_M'].values, *popt)
                                ss_res = np.sum((y_vals - y_pred_hill)**2)
                                ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
                                hill_r2 = 1 - (ss_res / (ss_tot + 1e-9))
                            except:
                                hill_r2 = 0
                            
                            results.append({
                                'pattern': pat, 'sigma': sig, 'z_cut': z_cut if z_cut else 'None',
                                'metric': met, 'relu': use_relu,
                                'Linear_R2': r2, 'Hill_R2': hill_r2,
                                'slope': slope, 'intercept': intercept, 'df_conc': g_conc
                            })
                            
    df_res = pd.DataFrame(results).sort_values(by='Linear_R2', ascending=False)
    
    print("=======================================================")
    print(f" [結果] {dataset_name} 線形性 (Linear R^2) 上位 5 パラメータ")
    print("=======================================================")
    top5 = df_res.head(5)
    print(top5[['pattern', 'metric', 'relu', 'sigma', 'z_cut', 'Linear_R2', 'Hill_R2']].to_string(index=False))
    
    best = df_res.iloc[0]
    
    g_best = best['df_conc']
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_plot = [x if x > 0 else 1e-16 for x in g_best['conc_M'].values]
    y_plot = g_best['val'].values
    
    label_str = f"Fit ({best['pattern']}, {best['metric']}, ReLU={best['relu']})"
    ax.plot(x_plot, y_plot, 'o-', color='#1f77b4', linewidth=2.5, markersize=9, label=label_str)
    
    x_fit = np.logspace(-15, -9, 100)
    y_fit = best['slope'] * np.log10(x_fit) + best['intercept']
    ax.plot(x_fit, y_fit, 'r--', linewidth=2.0, label=f"Linear Fit (R2 = {best['Linear_R2']:.4f})")
    
    ticks = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
    tick_labels = ['0 M\n(Blank)', '1 fM', '10 fM', '100 fM', '1 pM', '10 pM', '100 pM', '1 nM']
    
    ax.set_xscale('log')
    ax.set_xlabel('Concentration [M]', fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Signal ({best['metric']})", fontsize=11, fontweight='bold')
    ax.set_title(f"{dataset_name}: Advanced Optimal Fit (R2 = {best['Linear_R2']:.4f})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=35, fontsize=9)
    plt.tight_layout()
    
    out_png = os.path.join(input_dir, f"{dataset_name}_advanced_optimal_curve.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    return best

if __name__ == "__main__":
    base_p200 = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200"
    base_p100_1 = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1"
    base_p100_2 = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2"
    
    run_advanced_optimization(base_p200, '260706_sam_p200', MAPPING_P200)
    run_advanced_optimization(base_p100_1, '260707_sam_p100_1', MAPPING_P100_1)
    run_advanced_optimization(base_p100_2, '260707_sam_p100_2', MAPPING_P100_2)
