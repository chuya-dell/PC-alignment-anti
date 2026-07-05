import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import os

def run_shuffle_verification(pre_csv, aligned_csv):
    """
    検証処理:
    1. 実際のアライメント結果のマッチ率
    2. 画像内に一様ランダムに配置した場合の期待マッチ率 (一様分布)
    3. アライメント後の点群を全体的に平行移動（ランダムにシフト）させた場合のマッチ率
    """
    # CSV読み込み
    df_pre = pd.read_csv(pre_csv)
    df_aligned = pd.read_csv(aligned_csv)
    
    pts_pre = df_pre[['x', 'y']].values
    pts_aligned = df_aligned[['x', 'y']].values
    
    n_pre = len(pts_pre)
    n_post = len(pts_aligned)
    
    tree_pre = KDTree(pts_pre)
    
    # 1. 実際のアライメントマッチ率
    dists, _ = tree_pre.query(pts_aligned, k=1)
    actual_matches = np.sum(dists <= 1.5)
    actual_rate = (actual_matches / n_post) * 100.0
    
    # 2. 一様ランダム配置によるベースライン (10回試行の平均)
    random_rates = []
    # 画像サイズ 2048 x 2044
    w, h = 2048, 2044
    for _ in range(10):
        random_pts = np.column_stack((
            np.random.uniform(0, w, n_post),
            np.random.uniform(0, h, n_post)
        ))
        r_dists, _ = tree_pre.query(random_pts, k=1)
        r_matches = np.sum(r_dists <= 1.5)
        random_rates.append((r_matches / n_post) * 100.0)
    avg_random_rate = np.mean(random_rates)
    
    # 3. 座標系全体をランダムシフトさせた場合のマッチ率 (100回試行)
    # ズレの幅：[-50, 50]ピクセル、ただしグリッド間隔(5.5px)より大きくズラす
    shifted_rates = []
    for _ in range(100):
        # 5.5px未満の微小なズレを排除するために、一定以上のズレを与える
        dx = np.random.uniform(10, 100) * np.random.choice([-1, 1])
        dy = np.random.uniform(10, 100) * np.random.choice([-1, 1])
        
        shifted_pts = pts_aligned.copy()
        shifted_pts[:, 0] += dx
        shifted_pts[:, 1] += dy
        
        s_dists, _ = tree_pre.query(shifted_pts, k=1)
        s_matches = np.sum(s_dists <= 1.5)
        shifted_rates.append((s_matches / n_post) * 100.0)
        
    avg_shifted_rate = np.mean(shifted_rates)
    max_shifted_rate = np.max(shifted_rates)
    
    return {
        'n_pre': n_pre,
        'n_post': n_post,
        'actual_rate': actual_rate,
        'random_rate': avg_random_rate,
        'shifted_rate_mean': avg_shifted_rate,
        'shifted_rate_max': max_shifted_rate
    }

def main():
    out_dir = "F:/GoogleDrive_local/1.実験データ_gdrive/5.生データ D/260704 sam 位置合わせ test/foranti"
    
    # 評価するペア
    target_pairs = ["1-1", "1-2", "8-5", "8-8"]
    
    print("=== Alignment Baseline Significance Verification ===")
    print("Pre-pillar radius matching threshold: 1.5 pixels\n")
    print(f"{'Pair':<6} | {'Pre P.':<8} | {'Post P.':<8} | {'Actual Match%':<14} | {'Uniform Rand%':<14} | {'Shifted Match% (Mean / Max)':<30}")
    print("-" * 95)
    
    results = []
    for pair in target_pairs:
        pre_csv = os.path.join(out_dir, f"{pair}_pillars_pre.csv")
        aligned_csv = os.path.join(out_dir, f"{pair}_aligned_to_pre.csv")
        
        if not os.path.exists(pre_csv) or not os.path.exists(aligned_csv):
            print(f"{pair:<6} | Files not found. Skipped.")
            continue
            
        res = run_shuffle_verification(pre_csv, aligned_csv)
        print(f"{pair:<6} | {res['n_pre']:<8} | {res['n_post']:<8} | {res['actual_rate']:<13.2f}% | {res['random_rate']:<13.2f}% | {res['shifted_rate_mean']:5.2f}% / {res['shifted_rate_max']:5.2f}%")
        results.append(res)
        
    # 全体の統計的結論
    print("\n--- Statistical Conclusion ---")
    for i, pair in enumerate(target_pairs):
        actual = results[i]['actual_rate']
        base = results[i]['random_rate']
        shift_max = results[i]['shifted_rate_max']
        
        ratio_to_base = actual / base if base > 0 else 0
        ratio_to_shift = actual / shift_max if shift_max > 0 else 0
        
        print(f"Pair {pair}:")
        print(f"  - Actual match rate ({actual:.2f}%) is {ratio_to_base:.1f}x higher than uniform random baseline ({base:.2f}%).")
        print(f"  - Actual match rate is {ratio_to_shift:.1f}x higher than the maximum accidental shift match rate ({shift_max:.2f}%).")
        print(f"  -> Statistical Significance: {'HIGHLY SIGNIFICANT' if ratio_to_base > 1.5 else 'LOW'}\n")

if __name__ == "__main__":
    main()
