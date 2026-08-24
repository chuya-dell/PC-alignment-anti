import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import re

EXCLUDE_REASONS = []

def run_scientific_exclusion(excel_path, dataset_name, user_notes):
    if not os.path.exists(excel_path):
        return None, None
        
    df = pd.read_excel(excel_path, sheet_name=0)
    df['series'] = df['ファイル名'].apply(lambda x: re.split(r'[_\-\.]', str(x))[0])
    
    col_5s = [c for c in df.columns if '5σ超過規格化割合' in c or '> μ+5σ (%)' in c][0]
    
    excluded_rows = []
    clean_indices = []
    
    for idx, row in df.iterrows():
        filename = str(row['ファイル名'])
        series = str(row['series'])
        mean_i = row['平均輝度 (μ)']
        delta_m = row['BG比輝度変化 (ΔMean)']
        n_total = row['規格化母数 [総ピラー数 N_total]']
        pattern = row['評価パターン']
        
        # Rule 1: User / Experimental Note Justification
        if series in user_notes:
            excluded_rows.append({
                'ファイル名': filename,
                '系列': series,
                '評価パターン': pattern,
                '平均輝度': mean_i,
                '除外基準カテゴリー': '【基準1】実験ノート記録（物理的破壊・ゴミ・調製失敗）',
                '定量的根拠・数理証明': f'実験記録に明記: {user_notes[series]}'
            })
            continue
            
        # Rule 2: Alignment Failure (Matching Count < 500 in Pattern A)
        if 'Pattern A' in pattern and n_total < 500:
            excluded_rows.append({
                'ファイル名': filename,
                '系列': series,
                '評価パターン': pattern,
                '平均輝度': mean_i,
                '除外基準カテゴリー': '【基準2】アライメント幾何学的不全（同定数極小）',
                '定量的根拠・数理証明': f'マッチ同定数 N={n_total} < 500 (追跡失敗)'
            })
            continue

        # Rule 3: Extreme Signal Drop / SNR Failure (Delta Mean < -5.0)
        if delta_m < -5.0:
            excluded_rows.append({
                'ファイル名': filename,
                '系列': series,
                '評価パターン': pattern,
                '平均輝度': mean_i,
                '除外基準カテゴリー': '【基準3】撮影不全・コントラスト極度低下（焦点ズレ）',
                '定量的根拠・数理証明': f'ネガティブコントロール比 ΔMean={delta_m:.2f} < -5.0'
            })
            continue
            
        clean_indices.append(idx)
        
    # Rule 4: Grubbs' Test / Z-Score > 3.0 within same condition
    df_temp = df.loc[clean_indices].copy()
    final_clean_indices = []
    
    for (series, pattern), group in df_temp.groupby(['series', '評価パターン']):
        intensities = group['平均輝度 (μ)'].values
        g_indices = group.index.values
        
        if len(group) >= 4:
            mu = np.mean(intensities)
            sigma = np.std(intensities, ddof=1)
            
            for i in range(len(group)):
                z = abs(intensities[i] - mu) / (sigma + 1e-9)
                if z > 3.0 and sigma > 1.5:
                    excluded_rows.append({
                        'ファイル名': df.loc[g_indices[i], 'ファイル名'],
                        '系列': series,
                        '評価パターン': pattern,
                        '平均輝度': intensities[i],
                        '除外基準カテゴリー': '【基準4】グラブス検定統計的外れ値 (Z > 3.0)',
                        '定量的根拠・数理証明': f'同一条件内平均 μ={mu:.2f}, σ={sigma:.2f} に対し Z={z:.2f} > 3.0 (信頼区間99.7%外)'
                    })
                else:
                    final_clean_indices.append(g_indices[i])
        else:
            final_clean_indices.extend(g_indices)
            
    df_excluded = pd.DataFrame(excluded_rows)
    df_clean_final = df.loc[list(set(final_clean_indices))].copy()
    
    return df_clean_final, df_excluded

notes_p200 = {'8': '1 pM (ゴミ・調製失敗)'}
notes_p100_1 = {'8': '0 M (疑問・非標準ブランク)'}
notes_p100_2 = {'7': '1 fM (Post測定時基板剥離)'}

datasets = [
    ('260706_sam_p200', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200\intensity_summary_3s_5s.xlsx', notes_p200),
    ('260707_sam_p100_1', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1\intensity_summary_3s_5s.xlsx', notes_p100_1),
    ('260707_sam_p100_2', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2\intensity_summary_3s_5s.xlsx', notes_p100_2)
]

for name, excel_path, notes in datasets:
    df_clean, df_ex = run_scientific_exclusion(excel_path, name, notes)
    if df_clean is None:
        continue
        
    print(f"\n=======================================================")
    print(f" ■ {name} 科学的根拠に基づくデータ除外結果")
    print(f"=======================================================")
    print(f"  ・総分析データ数  : {len(pd.read_excel(excel_path)):,} 行")
    print(f"  ・除外データ数    : {len(df_ex):,} 行")
    print(f"  ・採用クリーン数  : {len(df_clean):,} 行")
    
    if len(df_ex) > 0:
        print("\n--- 【科学的除外根拠一覧】 ---")
        print(df_ex[['ファイル名', '除外基準カテゴリー', '定量的根拠・数理証明']].head(15).to_string(index=False))
