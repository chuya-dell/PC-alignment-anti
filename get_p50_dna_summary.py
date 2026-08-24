import os
import re
import pandas as pd
import numpy as np

base_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna"

for sub in ['a', 'b']:
    excel_path = os.path.join(base_dir, sub, "intensity_summary_3s_5s.xlsx")
    if not os.path.exists(excel_path):
        print(f"[{sub}] エクセル準備中: {excel_path}")
        continue
    df = pd.read_excel(excel_path, sheet_name=0)
    df['series'] = df['ファイル名'].apply(lambda x: re.split(r'[_\-\.]', str(x))[0])
    
    print(f"\n=======================================================")
    print(f" ■ 260822_p50_dna / サブフォルダ 『{sub}』 条件別集計")
    print(f"=======================================================")
    col_5s = [c for c in df.columns if '5σ超過規格化割合' in c or '> μ+5σ (%)' in c][0]
    
    g = df.groupby(['series', '評価パターン']).agg({
        '規格化母数 [総ピラー数 N_total]': 'mean',
        '平均輝度 (μ)': 'mean',
        'BG比輝度変化 (ΔMean)': 'mean',
        col_5s: lambda x: np.mean(x)*100 if np.mean(x)<1.0 else np.mean(x)
    }).reset_index()
    
    print(g.to_string(index=False))
