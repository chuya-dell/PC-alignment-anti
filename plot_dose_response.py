import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Explicit exclusion list
EXCLUDE_MAP = {
    '260706_sam_p200': ['8'],            # No.8: 1 pM (失敗・ゴミ)
    '260707_sam_p100_1': ['8', '6'],     # No.8: 0 M(疑問), No.6: 10 fM (異常低下)
    '260707_sam_p100_2': ['7']           # No.7: 1 fM (post測定時に剥離)
}

mapping_p200 = {
    '13': ('0 M (Blank)', 0.0),
    '6':  ('1 fM', 1e-15),
    '10': ('10 fM', 1e-14),
    '9':  ('100 fM', 1e-13),
    '11': ('1 pM', 1e-12),
    '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10),
    '1':  ('1 nM', 1e-9)
}

mapping_p100_1 = {
    '10': ('0 M (Blank)', 0.0),
    '11': ('0 M (Blank)', 0.0),
    '7':  ('1 fM', 1e-15),
    '9':  ('1 fM (Dup)', 1e-15),
    '12': ('10 fM', 1e-14),
    '5':  ('100 fM', 1e-13),
    '4':  ('1 pM', 1e-12),
    '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10),
    '1':  ('1 nM', 1e-9)
}

mapping_p100_2 = {
    '8':  ('0 M (Blank)', 0.0),
    '9':  ('1 fM', 1e-15),
    '6':  ('10 fM', 1e-14),
    '12': ('10 fM (Dup)', 1e-14),
    '5':  ('100 fM', 1e-13),
    '11': ('100 fM (Dup)', 1e-13),
    '4':  ('1 pM', 1e-12),
    '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10),
    '1':  ('1 nM', 1e-9)
}

datasets = [
    ('260706_sam_p200', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200\intensity_summary_3s_5s.xlsx', mapping_p200),
    ('260707_sam_p100_1', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1\intensity_summary_3s_5s.xlsx', mapping_p100_1),
    ('260707_sam_p100_2', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2\intensity_summary_3s_5s.xlsx', mapping_p100_2)
]

for name, excel_path, mapping in datasets:
    if not os.path.exists(excel_path):
        continue
        
    dir_path = os.path.dirname(excel_path)
    df = pd.read_excel(excel_path, sheet_name=0)
    df['series'] = df['ファイル名'].apply(lambda x: re.split(r'[_\-\.]', str(x))[0])
    
    # Filter out abnormal / excluded samples
    ex_list = EXCLUDE_MAP.get(name, [])
    df_filtered = df[~df['series'].isin(ex_list)].copy()
    
    df_filtered['濃度表記'] = df_filtered['series'].apply(lambda s: mapping.get(str(s), ('Unknown', None))[0])
    df_filtered['濃度(M)'] = df_filtered['series'].apply(lambda s: mapping.get(str(s), ('Unknown', None))[1])
    
    df_valid = df_filtered[df_filtered['濃度(M)'].notnull()].copy()
    col_5s = [c for c in df.columns if '5σ超過規格化割合' in c or '> μ+5σ (%)' in c][0]
    col_3s = [c for c in df.columns if '3σ超過規格化割合' in c or '> μ+3σ (%)' in c][0]
    
    g = df_valid.groupby(['濃度(M)', '濃度表記', '評価パターン']).agg({
        col_5s: lambda x: np.mean(x)*100 if np.mean(x)<1.0 else np.mean(x),
        col_3s: lambda x: np.mean(x)*100 if np.mean(x)<1.0 else np.mean(x),
        'BG比輝度変化 (ΔMean)': 'mean'
    }).reset_index().sort_values(by='濃度(M)')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    patterns = [('Pattern A', 'Pattern A (Matched Pairs)', 'o-', '#1f77b4'), ('Pattern B', 'Pattern B (All Grid Projected)', 's--', '#ff7f0e')]
    
    for p_label, p_legend, fmt, color in patterns:
        sub = g[g['評価パターン'].str.contains(p_label)]
        if len(sub) == 0:
            continue
            
        x_vals = [x if x > 0 else 1e-16 for x in sub['濃度(M)']]
        y_5s = sub[col_5s].values
        y_delta = sub['BG比輝度変化 (ΔMean)'].values
        
        # Subplot 1: Normalized 5sigma Exceeding Percentage (%)
        ax1.plot(x_vals, y_5s, fmt, label=p_legend, color=color, linewidth=2.5, markersize=8)
        
        # Subplot 2: Delta Mean Intensity
        ax2.plot(x_vals, y_delta, fmt, label=p_legend, color=color, linewidth=2.5, markersize=8)

    # Format Subplot 1
    ax1.set_xscale('log')
    ax1.set_xlabel('Concentration [M]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('5-Sigma Exceeding Pillar Ratio [%]', fontsize=11, fontweight='bold')
    ax1.set_title(f'{name} (Outliers Filtered): 5-Sigma Ratio [%]', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    ticks = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9]
    tick_labels = ['0 M\n(Blank)', '1 fM', '10 fM', '100 fM', '1 pM', '10 pM', '100 pM', '1 nM']
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels, rotation=35, fontsize=9)

    # Format Subplot 2
    ax2.set_xscale('log')
    ax2.set_xlabel('Concentration [M]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Intensity Change (Delta Mean)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{name} (Outliers Filtered): Delta Mean Intensity', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels, rotation=35, fontsize=9)

    plt.tight_layout()
    
    out_png = os.path.join(dir_path, f"{name}_dose_response_curve.png")
    scratch_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer", f"{name}_dose_response_curve.png")
    brain_out = os.path.join(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572", f"{name}_dose_response_curve.png")
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(scratch_out, dpi=300)
    plt.savefig(brain_out, dpi=300)
    plt.close()
    print(f"Filter graph saved: {name}")
