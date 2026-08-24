import os
import glob
import re
import pandas as pd
import numpy as np

def generate_clean_p50_dna_excel(input_dir):
    input_dir = os.path.abspath(input_dir)
    sub = os.path.basename(input_dir)
    
    # 1. Collect CSV files
    aligned_csvs = sorted(glob.glob(os.path.join(input_dir, "*_aligned.csv")))
    estimated_csvs = sorted(glob.glob(os.path.join(input_dir, "*_estimated_grid.csv")))
    
    if not aligned_csvs:
        aligned_csvs = sorted(glob.glob(os.path.join(input_dir, "*_pillars.csv")))
        
    # 2. Identify BG files (starting with 0-)
    bg_aligned = [f for f in aligned_csvs if os.path.basename(f).startswith("0-")]
    
    bg_vals = []
    for bg_file in bg_aligned:
        df_bg = pd.read_csv(bg_file)
        bg_vals.extend(df_bg['mean_intensity'].dropna().values)
        
    if len(bg_vals) == 0:
        print(f"[{sub}] ブランクデータの読み込みに失敗しました。")
        return
        
    bg_arr = np.array(bg_vals)
    bg_mean = np.mean(bg_arr)
    bg_std = np.std(bg_arr)
    thresh_3s = bg_mean + 3 * bg_std
    thresh_5s = bg_mean + 5 * bg_std
    
    print(f"\n=======================================================")
    print(f" ■ 260822_p50_dna / サブフォルダ 『{sub}』 解析結果")
    print(f"=======================================================")
    print(f"  ・ブランク平均 (μ_BG)  : {bg_mean:.4f}")
    print(f"  ・ブランク標準偏差(σ_BG): {bg_std:.4f}")
    print(f"  ・3σ 判定閾値          : {thresh_3s:.4f}")
    print(f"  ・5σ 判定閾値          : {thresh_5s:.4f}")
    
    # Process all CSVs
    all_csvs = sorted(aligned_csvs + estimated_csvs)
    summary_rows = []
    
    for cpath in all_csvs:
        fname = os.path.basename(cpath)
        base_name = os.path.splitext(fname)[0]
        
        # Extract series number (e.g. '0', '1', '2', '6', '8'...)
        series = re.split(r'[_\-\.]', fname)[0]
        
        if 'aligned' in fname:
            pattern = 'Pattern A (実測同定ペア)'
        elif 'estimated_grid' in fname:
            pattern = 'Pattern B (全基準格子推定)'
        else:
            pattern = '単一画像全検出'
            
        df = pd.read_csv(cpath)
        vals = df['mean_intensity'].dropna().values
        n_total = len(vals)
        
        if n_total == 0:
            continue
            
        mu = np.mean(vals)
        std = np.std(vals)
        delta_m = mu - bg_mean
        
        c3s = np.sum(vals > thresh_3s)
        r3s = (c3s / n_total) * 100
        
        c5s = np.sum(vals > thresh_5s)
        r5s = (c5s / n_total) * 100
        
        summary_rows.append({
            'ファイル名': fname,
            'サンプル系列': series,
            '評価パターン': pattern,
            '規格化母数 [総ピラー数 N_total]': n_total,
            '平均輝度 (μ)': mu,
            '輝度標準偏差 (σ)': std,
            'BG比輝度変化 (ΔMean)': delta_m,
            '3σ超過数 [個]': c3s,
            '3σ超過規格化割合 [% = (超過数/N_total)*100]': r3s,
            '5σ超過数 [個]': c5s,
            '5σ超過規格化割合 [% = (超過数/N_total)*100]': r5s
        })
        
    df_summary = pd.DataFrame(summary_rows)
    
    # Aggregate series summary
    df_series = df_summary.groupby(['サンプル系列', '評価パターン']).agg({
        '規格化母数 [総ピラー数 N_total]': 'mean',
        '平均輝度 (μ)': 'mean',
        '輝度標準偏差 (σ)': 'mean',
        'BG比輝度変化 (ΔMean)': 'mean',
        '3σ超過数 [個]': 'mean',
        '3σ超過規格化割合 [% = (超過数/N_total)*100]': 'mean',
        '5σ超過数 [個]': 'mean',
        '5σ超過規格化割合 [% = (超過数/N_total)*100]': 'mean'
    }).reset_index()
    
    # Save Excel file cleanly
    excel_out = os.path.join(input_dir, "intensity_summary_3s_5s.xlsx")
    with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
        df_summary.drop(columns=['サンプル系列']).to_excel(writer, sheet_name='全データ規格化サマリー', index=False)
        df_series.to_excel(writer, sheet_name='サンプル系列別規格化平均', index=False)
        
    print(f"  [完了] エクセル保存完了: {excel_out}")
    print("\n--- サンプル条件別平均集計結果 ---")
    print(df_series.to_string(index=False))

if __name__ == "__main__":
    base = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna"
    for sub in ['a', 'b']:
        generate_clean_p50_dna_excel(os.path.join(base, sub))
