import pandas as pd
import numpy as np
import os
import re
import scipy.stats as stats

mapping_p200 = {
    '13': ('0 M (Blank)', 0.0),
    '6':  ('1 fM', 1e-15),
    '10': ('10 fM', 1e-14),
    '9':  ('100 fM', 1e-13),
    '11': ('1 pM', 1e-12),
    '8':  ('1 pM (Fail)', None),
    '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10),
    '1':  ('1 nM', 1e-9)
}

mapping_p100_1 = {
    '10': ('0 M (Blank)', 0.0),
    '11': ('0 M (Blank)', 0.0),
    '8':  ('0 M (Blank?)', 0.0),
    '7':  ('1 fM', 1e-15),
    '9':  ('1 fM (Dup)', 1e-15),
    '6':  ('10 fM', 1e-14),
    '12': ('10 fM (Dup)', 1e-14),
    '5':  ('100 fM', 1e-13),
    '4':  ('1 pM', 1e-12),
    '3':  ('10 pM', 1e-11),
    '2':  ('100 pM', 1e-10),
    '1':  ('1 nM', 1e-9)
}

mapping_p100_2 = {
    '8':  ('0 M (Blank)', 0.0),
    '9':  ('1 fM', 1e-15),
    '7':  ('1 fM (Peeled)', None),
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
        
    print(f"\n=======================================================")
    print(f" ■ {name} 濃度(Log C) vs 輝度応答・相関解析結果")
    print(f"=======================================================")
    
    df = pd.read_excel(excel_path, sheet_name=0)
    df['series'] = df['ファイル名'].apply(lambda x: re.split(r'[_\-\.]', str(x))[0])
    df['濃度表記'] = df['series'].apply(lambda s: mapping.get(str(s), ('Unknown', None))[0])
    df['濃度(M)'] = df['series'].apply(lambda s: mapping.get(str(s), ('Unknown', None))[1])
    
    # Valid non-zero concentration samples
    df_valid = df[df['濃度(M)'].notnull()].copy()
    
    # 1. 260706_sam_p200 specific concentration response table
    dose_table = df_valid.groupby(['濃度(M)', '濃度表記']).agg({
        '平均輝度 (μ)': 'mean',
        'BG比輝度変化 (ΔMean)': 'mean',
        '3σ超過規格化割合 [% = (超過数/N_total)*100]': 'mean',
        '5σ超過規格化割合 [% = (超過数/N_total)*100]': 'mean'
    }).reset_index().sort_values(by='濃度(M)')
    
    print("\n--- 濃度順平均データ ---")
    print(dose_table.to_string(index=False))
    
    # Correlation with Log10(Concentration) for non-zero concentrations
    df_nonzero = df_valid[df_valid['濃度(M)'] > 0].copy()
    df_nonzero['log_c'] = np.log10(df_nonzero['濃度(M)'])
    
    if len(df_nonzero) > 0:
        r_p, p_p = stats.pearsonr(df_nonzero['log_c'], df_nonzero['平均輝度 (μ)'])
        r_s, p_s = stats.spearmanr(df_nonzero['log_c'], df_nonzero['平均輝度 (μ)'])
        
        print(f"\n--- 統計相関係数 (Log10[Conc] vs 平均輝度) ---")
        print(f"  ・ピアソン積率相関係数 R  : {r_p:+.4f} (p = {p_p:.4e})")
        print(f"  ・スピアマン順位相関係数 R: {r_s:+.4f} (p = {p_s:.4e})")
        if p_p < 0.05 or p_s < 0.05:
            print("  ★ 判定: 有意な対数濃度依存性（系統的相関）が確認されました！ (p < 0.05)")
        else:
            print("  判定: 統計的な直線相関は低め（非線形または飽和領域あり）")
