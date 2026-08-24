import os
import glob
import numpy as np
import pandas as pd
from analyzer import analyze_image

base_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260822_p50_dna"

print("=========================================================")
print(" 260822_p50_dna (a / b) ブランク(0-系) 基準統計量 速報")
print("=========================================================\n")

for sub in ['a', 'b']:
    sub_dir = os.path.join(base_dir, sub)
    if not os.path.exists(sub_dir):
        continue
        
    bg_files = sorted([f for f in os.listdir(sub_dir) if f.startswith("0-") and f.endswith(".tif")])
    
    all_vals = []
    processed_files = []
    
    for f in bg_files:
        img_path = os.path.join(sub_dir, f)
        base_name, _ = os.path.splitext(f)
        csv_path = os.path.join(sub_dir, f"{base_name}_pillars.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = analyze_image(img_path, tile_size=2000, overlap=50, method="peak", min_dist=4, top_hat_size=15)
            df.to_csv(csv_path, index=False)
            
        vals = df['mean_intensity'].dropna().values
        all_vals.extend(vals)
        processed_files.append(f)
        
    if len(all_vals) > 0:
        vals_arr = np.array(all_vals)
        mu = np.mean(vals_arr)
        std = np.std(vals_arr)
        thresh_3s = mu + 3 * std
        thresh_5s = mu + 5 * std
        
        print(f"---------------------------------------------------------")
        print(f"■ サブフォルダ: {sub}")
        print(f"  ・解析対象ブランク画像数: {len(processed_files)} 枚 ({', '.join(processed_files[:5])} ...)")
        print(f"  ・集計基準ピラー総数    : {len(vals_arr):,} 個")
        print(f"  ・ブランク平均輝度 (μ) : {mu:.4f}")
        print(f"  ・ブランク標準偏差 (σ) : {std:.4f}")
        print(f"  ・判定閾値 (μ + 3σ)   : {thresh_3s:.4f}")
        print(f"  ・判定閾値 (μ + 5σ)   : {thresh_5s:.4f}")
        print(f"---------------------------------------------------------")
