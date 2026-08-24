import os
import glob
import re
import numpy as np
import pandas as pd
from analyzer import analyze_image

datasets = [
    ('260706_sam_p200', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200', ['13']),
    ('260707_sam_p100_1', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_1', ['10', '11']),
    ('260707_sam_p100_2', r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260707_sam_p100_2', ['8'])
]

print("=========================================================")
print(" 3つの実験データ 基準(バックグラウンド)統計量 速報結果")
print("=========================================================\n")

for name, exp_path, bg_prefixes in datasets:
    if not os.path.exists(exp_path):
        print(f"[{name}] フォルダが見つかりません: {exp_path}")
        continue
        
    all_files = sorted(os.listdir(exp_path))
    str_prefixes = [str(b).strip() for b in bg_prefixes]
    
    bg_files = []
    for f in all_files:
        if not f.endswith('.tif'):
            continue
        for bg in str_prefixes:
            if f.startswith(bg + '-') or f.startswith(bg + '_') or f.startswith(bg + '.'):
                bg_files.append(f)
                break

    all_vals = []
    processed_count = 0
    
    # Process up to 6 background TIF files for high accuracy baseline calculation
    for bg_file in bg_files[:6]:
        img_path = os.path.join(exp_path, bg_file)
        base_name, _ = os.path.splitext(bg_file)
        csv_path = os.path.join(exp_path, f"{base_name}_pillars.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = analyze_image(img_path, tile_size=2000, overlap=50, method='peak', min_dist=4, top_hat_size=15)
            df.to_csv(csv_path, index=False)
            
        vals = df['mean_intensity'].dropna().values
        all_vals.extend(vals)
        processed_count += 1
        
    if len(all_vals) > 0:
        vals_arr = np.array(all_vals)
        mu = np.mean(vals_arr)
        std = np.std(vals_arr)
        thresh_3s = mu + 3 * std
        thresh_5s = mu + 5 * std
        
        print(f"---------------------------------------------------------")
        print(f"■ {name}")
        print(f"  ・基準識別子        : No.{', No.'.join(str_prefixes)}")
        print(f"  ・解析対象BG画像数   : {processed_count} 枚 ({', '.join(bg_files[:processed_count])})")
        print(f"  ・集計ピラー総数     : {len(vals_arr):,} 個")
        print(f"  ・BG平均輝度 (μ)    : {mu:.4f}")
        print(f"  ・BG標準偏差 (σ)    : {std:.4f}")
        print(f"  ・判定閾値 (μ + 3σ)  : {thresh_3s:.4f}")
        print(f"  ・判定閾値 (μ + 5σ)  : {thresh_5s:.4f}")
        print(f"---------------------------------------------------------")
    else:
        print(f"■ {name}: 基準画像が見つかりませんでした (検索キー: {str_prefixes})")

print("\n=========================================================")
