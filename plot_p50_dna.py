import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

base_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna"

for sub in ['a', 'b']:
    excel_path = os.path.join(base_dir, sub, "intensity_summary_3s_5s.xlsx")
    if not os.path.exists(excel_path):
        continue
        
    df = pd.read_excel(excel_path, sheet_name='サンプル系列別規格化平均')
    df_valid = df[df['サンプル系列'].astype(str).str.isdigit()].copy()
    df_valid['series_num'] = df_valid['サンプル系列'].astype(int)
    df_sorted = df_valid.sort_values(by='series_num')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    patterns = [
        ('Pattern A', 'Pattern A (Matched Pair)', 'o-', '#1f77b4'),
        ('Pattern B', 'Pattern B (All Grid Projected)', 's--', '#ff7f0e')
    ]
    
    for p_label, p_legend, fmt, color in patterns:
        sub_df = df_sorted[df_sorted['評価パターン'].str.contains(p_label)]
        if len(sub_df) == 0:
            continue
            
        x_vals = sub_df['series_num'].values
        y_5s = sub_df['5σ超過規格化割合 [% = (超過数/N_total)*100]'].values
        y_delta = sub_df['BG比輝度変化 (ΔMean)'].values
        
        ax1.plot(x_vals, y_5s, fmt, label=p_legend, color=color, linewidth=2.5, markersize=8)
        ax2.plot(x_vals, y_delta, fmt, label=p_legend, color=color, linewidth=2.5, markersize=8)
        
    ax1.set_xlabel('Condition Series No. (0 = Blank)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('5-Sigma Exceeding Pillar Ratio [%]', fontsize=11, fontweight='bold')
    ax1.set_title(f'260822_p50_dna ({sub}): 5-Sigma Ratio [%] by Condition', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, ls="--", alpha=0.5)
    
    ax2.set_xlabel('Condition Series No. (0 = Blank)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Intensity Change (Delta Mean)', fontsize=11, fontweight='bold')
    ax2.set_title(f'260822_p50_dna ({sub}): Delta Mean Intensity by Condition', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, ls="--", alpha=0.5)
    
    plt.tight_layout()
    
    out_png = os.path.join(base_dir, sub, f"260822_p50_dna_{sub}_summary_plot.png")
    scratch_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer", f"260822_p50_dna_{sub}_summary_plot.png")
    brain_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572", f"260822_p50_dna_{sub}_summary_plot.png")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(scratch_out, dpi=300)
    plt.savefig(brain_out, dpi=300)
    plt.close()
    print(f"Plot saved: {out_png}")
