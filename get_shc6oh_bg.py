import os
import re
import numpy as np
import pandas as pd
from analyzer import analyze_image

def save_csv_safe(df, filepath):
    filepath = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temp_path = filepath + ".tmp"
    try:
        df.to_csv(temp_path, index=False)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_path, filepath)
    except Exception:
        df.to_csv(filepath, index=False)

p = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\2060824_p50_SHC6OH"
bg_files = sorted([f for f in os.listdir(p) if (f.startswith("0-") or f.startswith("14-")) and f.endswith(".tif")])

all_vals = []
processed = []

print("=========================================================")
print(" 2060824_p50_SHC6OH ブランク (0- & 14-系) 基準統計量 速報")
print("=========================================================\n")

for f in bg_files:
    img_path = os.path.join(p, f)
    base_name, _ = os.path.splitext(f)
    csv_path = os.path.join(p, f"{base_name}_pillars.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = analyze_image(img_path, tile_size=2000, overlap=50, method="peak", min_dist=4, top_hat_size=15)
        save_csv_safe(df, csv_path)
        
    vals = df['mean_intensity'].dropna().values
    all_vals.extend(vals)
    processed.append(f)

if len(all_vals) > 0:
    vals_arr = np.array(all_vals)
    mu = np.mean(vals_arr)
    std = np.std(vals_arr)
    thresh_3s = mu + 3 * std
    thresh_5s = mu + 5 * std
    
    print(f"■ 解析対象ブランク画像数: {len(processed)} 枚")
    print(f"■ 集計基準ピラー総数    : {len(vals_arr):,} 個")
    print(f"■ ブランク平均輝度 (μ) : {mu:.4f}")
    print(f"■ ブランク標準偏差 (σ) : {std:.4f}")
    print(f"■ 判定閾値 (μ + 3σ)   : {thresh_3s:.4f}")
    print(f"■ 判定閾値 (μ + 5σ)   : {thresh_5s:.4f}")
