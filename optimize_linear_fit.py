import os
import re
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

CONC_MAP = {
    '0':  ('0 M (Blank)', 0.0),
    '14': ('0 M (Blank 2)', 0.0),
    '10': ('1 fM', 1e-15),
    '8':  ('10 fM', 1e-14),
    '13': ('10 fM (Dup)', 1e-14),
    '5':  ('100 fM', 1e-13),
    '4':  ('1 pM', 1e-12),
    '12': ('1 pM (Dup)', 1e-12),
    '3':  ('10 pM', 1e-11),
    '11': ('10 pM (Leak?)', 1e-11),
    '2':  ('100 pM (Leak?)', 1e-10),
    '1':  ('1 nM', 1e-9)
}

DEFECT_NOTES = {
    '0-1': '染みとごみ', '0-4': 'ごみ', '0-5': 'ごみ', '0-6': 'ごみ', '0-7': '染み', '0-8': '染みとごみ',
    '3-4': 'ごみ', '3-6': '染み（大）', '3-7': '染み', '3-8': '染み',
    '4-1': '染み', '4-2': 'ごみ', '4-5': 'ごみ', '4-6': '染み',
    '8-1': '染みとごみ', '8-5': '染み', '8-7': 'ごみ', '8-8': 'ごみと染み',
    '10-2': 'ごみと染み', '10-4': 'ごみ', '10-6': '染み', '10-8': '染みとごみ',
    '11-1': 'ごみ', '11-2': 'ごみ', '11-3': '染み', '11-4': '傷',
    '12-8': '染み', '13-4': 'ごみ', '14-1': '染み', '14-6': '染み'
}

def is_defect_file(filename):
    m = re.match(r'^(\d+)\-(\d+)', os.path.basename(filename))
    if m:
        key = f"{m.group(1)}-{m.group(2)}"
        if key in DEFECT_NOTES:
            return DEFECT_NOTES[key]
    return None

def run_grid_search_optimization(input_dir):
    aligned_csvs = sorted(glob.glob(os.path.join(input_dir, "*_aligned.csv")))
    estimated_csvs = sorted(glob.glob(os.path.join(input_dir, "*_estimated_grid.csv")))
    
    if not aligned_csvs:
        print("CSVファイルが準備できていません。アライメント処理を待機してください。")
        return
        
    print("\n=======================================================")
    print(" 🚀 線形検量線 (R^2 最大化) 自動試行錯誤・グリッドサーチ")
    print("=======================================================\n")
    
    # Pre-load pillar raw arrays for super fast parameter testing
    data_cache = []
    
    # Pure clean BG files
    bg_vals = []
    bg_files = [f for f in aligned_csvs if os.path.basename(f).startswith(("0-", "14-")) and not is_defect_file(f)]
    for bgf in bg_files:
        df_bg = pd.read_csv(bgf)
        bg_vals.extend(df_bg['mean_intensity'].dropna().values)
    bg_arr = np.array(bg_vals)
    bg_mean = np.mean(bg_arr)
    bg_std = np.std(bg_arr)
    
    print(f"■ 純粋ブランク基準: μ={bg_mean:.4f}, σ={bg_std:.4f} (総母数: {len(bg_arr):,} ピラー)")
    
    all_csvs = sorted(aligned_csvs + estimated_csvs)
    for cpath in all_csvs:
        fname = os.path.basename(cpath)
        series = re.split(r'[_\-\.]', fname)[0]
        pattern = 'Pattern A' if 'aligned' in fname else 'Pattern B'
        
        df = pd.read_csv(cpath)
        vals = df['mean_intensity'].dropna().values
        if len(vals) > 0:
            data_cache.append({
                'fname': fname,
                'series': series,
                'pattern': pattern,
                'vals': vals,
                'is_defect': is_defect_file(fname) is not None,
                'conc_M': CONC_MAP.get(str(series), ('Unknown', None))[1]
            })
            
    # Search grid space
    sigmas = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    z_cutoffs = [2.5, 2.8, 3.0, 3.5, None]
    patterns = ['Pattern A', 'Pattern B']
    metrics = ['ratio', 'delta_mean', 'delta_median']
    
    results = []
    
    for pat in patterns:
        for sig in sigmas:
            thresh = bg_mean + sig * bg_std
            for z_cut in z_cutoffs:
                for met in metrics:
                    sample_list = []
                    for item in data_cache:
                        if item['pattern'] != pat or item['is_defect'] or item['conc_M'] is None:
                            continue
                        vals = item['vals']
                        n_total = len(vals)
                        if n_total == 0:
                            continue
                            
                        if met == 'ratio':
                            val = (np.sum(vals > thresh) / n_total) * 100
                        elif met == 'delta_mean':
                            val = np.mean(vals) - bg_mean
                        elif met == 'delta_median':
                            val = np.median(vals) - bg_mean
                            
                        sample_list.append({
                            'fname': item['fname'],
                            'series': item['series'],
                            'conc_M': item['conc_M'],
                            'val': val
                        })
                        
                    df_run = pd.DataFrame(sample_list)
                    if len(df_run) == 0:
                        continue
                        
                    # Z-score filter within condition
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
                        
                    # Aggregate by concentration
                    g_conc = df_run.groupby('conc_M')['val'].mean().reset_index()
                    # Filter for non-zero concentrations for log-linear regression
                    g_log = g_conc[g_conc['conc_M'] > 0].copy()
                    
                    if len(g_log) >= 4:
                        log_x = np.log10(g_log['conc_M'].values)
                        y_vals = g_log['val'].values
                        
                        slope, intercept, r_val, p_val, std_err = stats.linregress(log_x, y_vals)
                        r2 = r_val ** 2
                        rho, spearm_p = stats.spearmanr(log_x, y_vals)
                        
                        results.append({
                            'pattern': pat,
                            'sigma': sig,
                            'z_cut': z_cut if z_cut else 'None',
                            'metric': met,
                            'R2': r2,
                            'R': r_val,
                            'p_val': p_val,
                            'spearman_rho': rho,
                            'slope': slope,
                            'intercept': intercept,
                            'df_conc': g_conc,
                            'df_raw': df_run
                        })
                        
    df_res = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    
    print("=======================================================")
    print(" 🏆 線形性 (R^2) 上位 5 つの最優パラメータ結果")
    print("=======================================================")
    top5 = df_res.head(5)
    print(top5[['pattern', 'sigma', 'z_cut', 'metric', 'R2', 'R', 'p_val', 'spearman_rho']].to_string(index=False))
    
    best = df_res.iloc[0]
    print("\n-------------------------------------------------------")
    print(f" ★ 最優組み合わせ選出:")
    print(f"   - パターン  : {best['pattern']}")
    print(f"   - 判定閾値  : μ + {best['sigma']}σ ({bg_mean + best['sigma']*bg_std:.2f})")
    print(f"   - Z除外閾値 : {best['z_cut']}")
    print(f"   - 評価指標  : {best['metric']}")
    print(f"   - 決定係数 R^2 : {best['R2']:.4f}")
    print(f"   - 相関係数 R   : {best['R']:.4f} (p = {best['p_val']:.4e})")
    print(f"   - 単調増加 ρ   : {best['spearman_rho']:.4f}")
    print("-------------------------------------------------------")

    # Plot optimal linear curve
    g_best = best['df_conc']
    log_x = np.log10(g_best[g_best['conc_M'] > 0]['conc_M'].values)
    y_best = g_best[g_best['conc_M'] > 0]['val'].values
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_plot = [x if x > 0 else 1e-16 for x in g_best['conc_M'].values]
    y_plot = g_best['val'].values
    
    ax.plot(x_plot, y_plot, 'o-', color='#1f77b4', linewidth=2.5, markersize=9, label=f"Optimal Fit ({best['pattern']}, μ+{best['sigma']}σ)")
    
    # Plot linear trendline
    x_fit = np.logspace(-15, -9, 100)
    y_fit = best['slope'] * np.log10(x_fit) + best['intercept']
    ax.plot(x_fit, y_fit, 'r--', linewidth=2.0, label=f"Linear Fit (R² = {best['R2']:.4f}, R = {best['R']:.4f})")
    
    ticks = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
    tick_labels = ['0 M\n(Blank)', '1 fM', '10 fM', '100 fM', '1 pM', '10 pM', '100 pM', '1 nM']
    
    ax.set_xscale('log')
    ax.set_xlabel('Concentration [M]', fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Signal ({best['metric']})", fontsize=11, fontweight='bold')
    ax.set_title(f"Optimal Linear Calibration Curve (R² = {best['R2']:.4f})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=35, fontsize=9)
    
    plt.tight_layout()
    
    out_png = os.path.join(input_dir, "optimal_linear_dose_response_curve.png")
    scratch_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer", "optimal_linear_dose_response_curve.png")
    brain_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572", "optimal_linear_dose_response_curve.png")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(scratch_out, dpi=300)
    plt.savefig(brain_out, dpi=300)
    plt.close()
    print(f"\n[完了] 最適線形検量線プロット出力: {out_png}")

if __name__ == "__main__":
    run_grid_search_optimization(r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\2060824_p50_SHC6OH")
