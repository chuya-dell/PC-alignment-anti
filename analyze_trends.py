import pandas as pd
import os
import re

paths = [
    ('260706_sam_p200 (基準 No.13 / 全約30,000ピラー)', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200\intensity_summary_3s_5s.xlsx'),
    ('260707_sam_p100_1 (基準 No.10&11 / 全約65,000ピラー)', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1\intensity_summary_3s_5s.xlsx'),
    ('260707_sam_p100_2 (基準 No.8 / 全約54,000ピラー)', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2\intensity_summary_3s_5s.xlsx')
]

for name, p in paths:
    if not os.path.exists(p):
        continue
    print(f"\n=======================================================")
    print(f" ■ {name}")
    print(f"   Pattern A (実測同定成功マッチピラー数) の集計結果")
    print(f"=======================================================")
    df = pd.read_excel(p)
    df_patA = df[df['評価パターン'].str.contains('Pattern A')].copy()
    df_patA['series'] = df_patA['ファイル名'].apply(lambda x: re.split(r'[_\-\.]', str(x))[0])
    
    summary = df_patA.groupby('series')['規格化母数 [総ピラー数 N_total]'].agg(
        枚数='count',
        平均マッチ数='mean',
        最小マッチ数='min',
        最大マッチ数='max'
    )
    print(summary.to_string())
