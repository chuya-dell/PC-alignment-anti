import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import re

from dose_response_analyzer import datasets

for name, p, m in datasets:
    if not os.path.exists(p):
        continue
    df = pd.read_excel(p)
    df['series'] = df['ファイル名'].apply(lambda x: re.split(r'[_\-\.]', str(x))[0])
    df['c'] = df['series'].apply(lambda s: m.get(str(s), (None, None))[1])
    df['label'] = df['series'].apply(lambda s: m.get(str(s), ('Unknown', None))[0])
    
    # Exclude invalid / peeled
    df_valid = df[df['c'].notnull()].copy()
    
    # 5σ percentage column name
    col_5s = [c for c in df.columns if '5σ超過規格化割合' in c or '> μ+5σ (%)' in c][0]
    col_3s = [c for c in df.columns if '3σ超過規格化割合' in c or '> μ+3σ (%)' in c][0]
    
    print(f"\n=======================================================")
    print(f" ■ {name} 濃度 vs 『規格化割合 (%)』 の評価結果")
    print(f"=======================================================")
    
    table = df_valid.groupby(['c', 'label']).agg({
        col_3s: lambda x: np.mean(x) * 100 if np.mean(x) < 1.0 else np.mean(x),
        col_5s: lambda x: np.mean(x) * 100 if np.mean(x) < 1.0 else np.mean(x)
    }).reset_index().sort_values(by='c')
    
    print(table.to_string(index=False))
    
    # Calculate correlation for non-zero concentrations
    df_nz = df_valid[df_valid['c'] > 0].copy()
    df_nz['log_c'] = np.log10(df_nz['c'])
    r3, p3 = stats.pearsonr(df_nz['log_c'], df_nz[col_3s])
    r5, p5 = stats.pearsonr(df_nz['log_c'], df_nz[col_5s])
    
    print(f"\n--- 【規格化割合 (%)】の濃度相関係数 ---")
    print(f"  ・3σ超過割合(%) 相関 R: {r3:+.4f} (p = {p3:.4e})")
    print(f"  ・5σ超過割合(%) 相関 R: {r5:+.4f} (p = {p5:.4e})")
