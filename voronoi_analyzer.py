import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, KDTree

def polygon_area_vectorized(vertices_list, regions, point_region, roi_mask):
    """
    Fast calculation of Voronoi polygon areas for interior points.
    """
    areas = np.full(len(point_region), np.nan)
    
    for i in np.where(roi_mask)[0]:
        reg_idx = point_region[i]
        reg = regions[reg_idx]
        if not reg or any(v < 0 for v in reg):
            continue
        
        pts = vertices_list[reg]
        x = pts[:, 0]
        y = pts[:, 1]
        # Shoelace formula
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas[i] = area
        
    return areas

def analyze_voronoi(csv_path, output_dir=None, margin=50):
    """
    High-performance Voronoi Partitioning & Spatial Order Analysis for Pillar Arrays.
    """
    t0 = time.time()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV must contain 'x' and 'y' columns.")

    num_pts = len(df)
    print(f"Loaded {num_pts:,} pillars from {csv_path}")
    points = df[['x', 'y']].values.astype(np.float64)

    if num_pts < 4:
        print("Error: Need at least 4 points for Voronoi tessellation.")
        return None

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    # Calculate ROI boundary to remove edge artifacts
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    roi_mask = (
        (points[:, 0] >= x_min + margin) & (points[:, 0] <= x_max - margin) &
        (points[:, 1] >= y_min + margin) & (points[:, 1] <= y_max - margin)
    )
    df['is_interior'] = roi_mask

    # 1. Voronoi Tessellation via SciPy Qhull
    print("Computing Voronoi tessellation...")
    vor = Voronoi(points)
    
    # 2. Extract Coordination Number (number of vertices in Voronoi region)
    print("Extracting coordination numbers...")
    coord_numbers = np.fromiter(
        (len(vor.regions[r]) if r >= 0 else 0 for r in vor.point_region),
        dtype=np.int32,
        count=num_pts
    )
    df['coordination_number'] = coord_numbers

    # 3. Calculate Voronoi Cell Areas (Shoelace formula)
    print("Computing cell areas...")
    vertices_arr = vor.vertices
    areas = polygon_area_vectorized(vertices_arr, vor.regions, vor.point_region, roi_mask)
    df['voronoi_area'] = areas

    # 4. Compute Vectorized Hexagonal Order Parameter psi_6
    print("Computing Hexagonal Order Parameter (psi_6)...")
    interior_indices = np.where(roi_mask)[0]
    psi_6_arr = np.full(num_pts, np.nan)
    
    if len(interior_indices) > 0:
        tree = KDTree(points)
        # Query nearest 6 neighbors (k=7 including self)
        dists, nbr_indices = tree.query(points[interior_indices], k=7)
        nbr_pts = points[nbr_indices[:, 1:]] # shape: (N_roi, 6, 2)
        ref_pts = points[interior_indices][:, None, :] # shape: (N_roi, 1, 2)
        
        dx = nbr_pts[:, :, 0] - ref_pts[:, :, 0]
        dy = nbr_pts[:, :, 1] - ref_pts[:, :, 1]
        angles = np.arctan2(dy, dx)
        
        # psi_6 = |1/6 sum(exp(6i * theta_j))|
        psi_6_vals = np.abs(np.mean(np.exp(6j * angles), axis=1))
        psi_6_arr[interior_indices] = psi_6_vals

    df['psi_6'] = psi_6_arr

    df_int = df[df['is_interior'] & df['voronoi_area'].notna()]
    
    t_elapsed = time.time() - t0
    print("\n================ ボロノイ解析サマリー ================")
    print(f"対象ピラー総数 (全体)     : {num_pts:,} 個")
    print(f"内部領域ピラー数 (ROI内)   : {len(df_int):,} 個")
    print(f"平均ボロノイ領域面積      : {df_int['voronoi_area'].mean():.2f} px^2 (標準偏差: {df_int['voronoi_area'].std():.2f})")
    print(f"中央値ボロノイ面積        : {df_int['voronoi_area'].median():.2f} px^2")
    print(f"平均配位数 (隣接ピラー数) : {df_int['coordination_number'].mean():.2f}")
    print(f"配位数 6 (理想HCP) の割合 : {(df_int['coordination_number'] == 6).mean()*100:.2f}%")
    print(f"平均配向秩序度 (ψ6)       : {df_int['psi_6'].mean():.4f}")
    print(f"総処理時間               : {t_elapsed:.2f} 秒")
    print("====================================================\n")

    # Save CSV
    out_csv = os.path.join(output_dir, "voronoi_analysis_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"結果CSVを保存しました: {out_csv}")

    # Plot 1: Area Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df_int['voronoi_area'], bins=80, color='crimson', edgecolor='black', alpha=0.7)
    plt.title('Voronoi Cell Area Distribution (Interior ROI)', fontsize=13)
    plt.xlabel('Cell Area (px²)', fontsize=11)
    plt.ylabel('Pillar Count', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "voronoi_area_histogram.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"ボロノイ面積ヒストグラムを出力: {hist_path}")

    # Plot 2: Coordination Distribution
    plt.figure(figsize=(7, 5))
    counts = df_int['coordination_number'].value_counts().sort_index()
    plt.bar(counts.index, counts.values, color='teal', edgecolor='black', alpha=0.8)
    plt.title('Coordination Number (Nearest Neighbors) Distribution', fontsize=13)
    plt.xlabel('Coordination Number (Ideal HCP = 6)', fontsize=11)
    plt.ylabel('Pillar Count', fontsize=11)
    plt.xticks(counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    coord_path = os.path.join(output_dir, "voronoi_coordination_distribution.png")
    plt.savefig(coord_path, dpi=150)
    plt.close()
    print(f"配位数分布グラフを出力: {coord_path}")

    # Plot 3: Spatial Heatmap of Voronoi Cell Area
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        df_int['x'], df_int['y'],
        c=df_int['voronoi_area'],
        cmap='viridis',
        s=0.5,
        alpha=0.9
    )
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label='Voronoi Cell Area (px²)')
    plt.title('Spatial Map of Voronoi Cell Area (Defect / Density Mapping)', fontsize=13)
    plt.xlabel('X (px)', fontsize=11)
    plt.ylabel('Y (px)', fontsize=11)
    plt.axis('equal')
    plt.tight_layout()
    map_path = os.path.join(output_dir, "voronoi_area_heatmap.png")
    plt.savefig(map_path, dpi=200)
    plt.close()
    print(f"空間面積ヒートマップを出力: {map_path}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voronoi Partitioning & Spatial Order Analysis for Pillar Arrays")
    parser.add_argument("--csv", type=str, required=True, help="Path to input pillar coordinates CSV")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--margin", type=int, default=50, help="ROI boundary margin in pixels to remove edge artifacts")

    args = parser.parse_args()
    analyze_voronoi(args.csv, args.outdir, margin=args.margin)
