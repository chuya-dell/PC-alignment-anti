import pandas as pd
import numpy as np
import os
import re

datasets = [
    ('260706_sam_p200', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200\intensity_summary_3s_5s.xlsx'),
    ('260707_sam_p100_1', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1\intensity_summary_3s_5s.xlsx'),
    ('260707_sam_p100_2', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2\intensity_summary_3s_5s.xlsx')
]

print("=======================================================")
print(" 全実験データに対する統計的異常値 (Outlier) 自動検診")
print("=======================================================\n")

for name, p in datasets:
    if not os.path.exists(p):
        continue
        
    df = pd.read_excel(p, sheet_name=0)
    df['series'] = df['ファイル名'].apply(lambda x: re.split(r'[_\-\.]', str(x))[0])
    
    col_5s = [c for c in df.columns if '5σ超過規格化割合' in c or '> μ+5σ (%)' in c][0]
    
    print(f"--- ■ {name} (総データ数: {len(df)} 行) ---")
    
    outliers = []
    
    for (series, pattern), group in df.groupby(['series', '評価パターン']):
        mean_int = group['平均輝度 (μ)'].values
        ratio_5s = group[col_5s].values
        files = group['ファイル名'].values
        
        # Calculate Z-scores within same series & pattern
        if len(group) > 2:
            mu_int, std_int = np.mean(mean_int), np.std(mean_int)
            mu_r, std_r = np.mean(ratio_5s), np.std(ratio_5s)
            
            for i in range(len(group)):
                z_int = abs(mean_int[i] - mu_int) / (std_int + 1e-9)
                z_r = abs(ratio_5s[i] - mu_r) / (std_r + 1e-9)
                
                # If Z-score > 2.5 (strong statistical outlier within same condition)
                if (z_int > 2.5 and std_int > 1.0) or (z_r > 2.5 and std_r > 0.01):
                    outliers.append({
                        '系列': series,
                        'パターン': pattern,
                        'ファイル名': files[i],
                        '輝度': mean_int[i],
                        '系列内平均輝度': mu_int,
                        'Z-Score (輝度)': z_int,
                        '5σ割合 (%)': ratio_5s[i] * 100 if ratio_5s[i] < 1.0 else ratio_5s[i],
                        '理由': f'同一条件内での大きな乖離 (Z={z_int:.2f})'
                    })
        
        # Check global extreme abnormal drops (e.g. Delta Mean < -5.0)
        for i in range(len(group)):
            delta = group['BG比輝度変化 (ΔMean)'].values[i]
            if delta < -5.0:
                outliers.append({
                    '系列': series,
                    'パターン': pattern,
                    'ファイル名': files[i],
                    '輝度': mean_int[i],
                    '系列内平均輝度': np.mean(mean_int),
                    'Z-Score (輝度)': 9.9,
                    '5σ割合 (%)': ratio_5s[i] * 100 if ratio_5s[i] < 1.0 else ratio_5s[i],
                    '理由': f'大幅な輝度低下・画像欠損 (ΔMean={delta:.2f})'
                })

    df_out = pd.DataFrame(outliers)
    if len(df_out) > 0:
        df_out = df_out.drop_duplicates(subset=['ファイル名', 'パターン'])
        print(f"  [注意] 検出された統計的異常値 ({len(df_out)} 件):")
        print(df_out[['系列', 'パターン', 'ファイル名', '輝度', '5σ割合 (%)', '理由']].to_string(index=False))
    else:
        print("  [OK] 異常値なし（データはすべて同一条件内で極めて安定しています）")
    print()
