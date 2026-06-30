import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def generate_summary(csv_path, histogram_path, heatmap_path, marker_size=None):
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("No pillars found in the CSV!")
        return
        
    num_pillars = len(df)
    mean_int = df['mean_intensity'].mean()
    std_int = df['mean_intensity'].std()
    min_int = df['mean_intensity'].min()
    max_int = df['mean_intensity'].max()
    
    print("\n================ 解析結果サマリー ================")
    print(f"検出されたピラー数 : {num_pillars:,} 個")
    print(f"平均輝度 (Mean)    : {mean_int:.2f}")
    print(f"輝度の標準偏差 (Std): {std_int:.2f}")
    print(f"最小輝度 (Min)     : {min_int:.2f}")
    print(f"最大輝度 (Max)     : {max_int:.2f}")
    print("=================================================\n")
    
    # 1. Generate histogram
    print("ヒストグラム画像を生成中...")
    plt.figure(figsize=(8, 6))
    plt.hist(df['mean_intensity'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Pillar Intensity Distribution', fontsize=14)
    plt.xlabel('Average Intensity (0-255)', fontsize=12)
    plt.ylabel('Pillar Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=150)
    plt.close()
    print(f"-> 輝度分布のヒストグラムを {histogram_path} に保存しました。")
    
    # 2. Generate spatial heatmap
    print("空間ヒートマップ画像を生成中...")
    plt.figure(figsize=(10, 8))
    
    # Adjust point size 's' based on user parameter or total pillars
    if marker_size is not None:
        point_size = marker_size
    elif num_pillars > 500000:
        point_size = 0.05
    elif num_pillars > 100000:
        point_size = 0.1
    else:
        point_size = 0.3 # Default is now very small for clean visualization
        
    # Scatter plot representing coordinates colored by intensity
    sc = plt.scatter(
        df['x'], 
        df['y'], 
        c=df['mean_intensity'], 
        cmap='inferno', # 'inferno' or 'viridis' are excellent scientific colormaps
        s=point_size, 
        alpha=1.0,
        edgecolors='none'
    )
    
    # Invert Y-axis to match image/microscope orientation (origin at top-left)
    plt.gca().invert_yaxis()
    
    plt.title('Spatial Intensity Heatmap of Plasmonic Pillars', fontsize=14)
    plt.xlabel('X Position (pixels)', fontsize=12)
    plt.ylabel('Y Position (pixels)', fontsize=12)
    plt.colorbar(sc, label='Average Intensity')
    plt.axis('equal') # Keep aspect ratio square
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300) # High-resolution output for crisp points
    plt.close()
    print(f"-> 空間ヒートマップを {heatmap_path} に保存しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize analysis results and plot intensity distribution + heatmap")
    parser.add_argument("--csv", type=str, default="results.csv", help="Path to results CSV")
    parser.add_argument("--hist", type=str, default="intensity_histogram.png", help="Path to save the histogram image")
    parser.add_argument("--map", type=str, default="intensity_heatmap.png", help="Path to save the heatmap image")
    parser.add_argument("--size", type=float, default=None, help="Marker size for the heatmap (default: auto)")
    
    args = parser.parse_args()
    
    generate_summary(args.csv, args.hist, args.map, args.size)
