import os
import argparse
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def load_image_16bit(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def generate_superposition_plot(pre_img_path, post_img_path, pre_csv_path, aligned_csv_path, output_path):
    print(f"Loading data for Superposition Plot...")
    
    pre_img = load_image_16bit(pre_img_path)
    post_img = load_image_16bit(post_img_path)
    if pre_img is None or post_img is None:
        return False
        
    def to_8bit(img):
        img_norm = (img - np.min(img)) / (np.percentile(img, 99) - np.min(img) + 1e-5)
        return np.clip(img_norm * 255, 0, 255).astype(np.uint8)
        
    pre_8 = to_8bit(pre_img)
    post_8 = to_8bit(post_img)
    
    composite = np.zeros((*pre_8.shape, 3), dtype=np.uint8)
    composite[:, :, 0] = pre_8  # Blue
    composite[:, :, 2] = post_8 # Red
    
    df_pre = pd.read_csv(pre_csv_path)
    df_post = pd.read_csv(aligned_csv_path)
    
    if 'matched_ref_id' not in df_post.columns:
        print("Invalid aligned CSV format.")
        return False
        
    df_matched = df_post[df_post['matched_ref_id'] >= 0].copy()
    print(f"Plotting {len(df_matched)} matched pairs...")
    
    # Precompute pre_id to coords dict
    pre_coords = {row['pillar_id']: (row['x'], row['y']) for _, row in df_pre.iterrows()}
    
    for _, row in df_matched.iterrows():
        ref_id = row['matched_ref_id']
        if ref_id in pre_coords:
            x1, y1 = pre_coords[ref_id]
            x2, y2 = row['x'], row['y']
            
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            
            cv2.circle(composite, (x1, y1), 2, (255, 0, 0), -1)
            cv2.circle(composite, (x2, y2), 2, (0, 0, 255), -1)
            cv2.line(composite, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
    h, w = composite.shape[:2]
    cy, cx = h//2, w//2
    size = 500
    cropped = composite[cy-size:cy+size, cx-size:cx+size]
    
    cv2.imwrite(output_path, cropped)
    print(f"Superposition Plot saved to {output_path}")
    return True

def generate_null_distribution(pre_csv_path, aligned_csv_path, output_path, iterations=1000):
    print(f"Generating Null Distribution (Random Shuffle) for {aligned_csv_path}...")
    df_pre = pd.read_csv(pre_csv_path)
    df_post = pd.read_csv(aligned_csv_path)
    
    df_matched = df_post[df_post['matched_ref_id'] >= 0].copy()
    pre_ints = {row['pillar_id']: row['mean_intensity'] for _, row in df_pre.iterrows()}
    
    ref_vals = []
    tgt_vals = []
    for _, row in df_matched.iterrows():
        ref_id = row['matched_ref_id']
        if ref_id in pre_ints:
            ref_vals.append(pre_ints[ref_id])
            tgt_vals.append(row['mean_intensity'])
            
    ref_vals = np.array(ref_vals)
    tgt_vals = np.array(tgt_vals)
    
    # 1. True delta I distribution (Real alignment)
    true_delta = tgt_vals - ref_vals
    
    # 2. Null delta I distribution (Random alignment)
    null_deltas = []
    for _ in range(iterations):
        np.random.shuffle(tgt_vals)
        null_deltas.extend((tgt_vals - ref_vals).tolist())
        
    null_deltas = np.array(null_deltas)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true distribution
    ax.hist(true_delta, bins=50, density=True, alpha=0.6, color='blue', label=f'True Alignment (N={len(true_delta)})')
    
    # Plot null distribution
    ax.hist(null_deltas, bins=100, density=True, alpha=0.5, color='gray', label=f'Null Distribution (Shuffled N={len(null_deltas)})')
    
    # Statistics
    true_mean = np.mean(true_delta)
    null_mean = np.mean(null_deltas)
    true_std = np.std(true_delta)
    
    ax.axvline(true_mean, color='blue', linestyle='dashed', linewidth=2, label=f'True Mean = {true_mean:.2f}')
    ax.axvline(null_mean, color='red', linestyle='dashed', linewidth=2, label=f'Null Mean = {null_mean:.2f}')
    
    ax.set_title("Physical Verification: True Alignment vs Random Null Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Intensity Change (Delta I)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Null Distribution Plot saved to {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=str, required=True, help="Path to Pre TIF image")
    parser.add_argument("--post", type=str, required=True, help="Path to Post TIF image")
    parser.add_argument("--pre-csv", type=str, required=True, help="Path to Pre CSV (raw detections)")
    parser.add_argument("--csv", type=str, required=True, help="Path to aligned Post CSV")
    parser.add_argument("--out-super", type=str, required=True, help="Output path for superposition plot")
    parser.add_argument("--out-null", type=str, required=True, help="Output path for null distribution plot")
    
    args = parser.parse_args()
    
    generate_superposition_plot(args.pre, args.post, args.pre_csv, args.csv, args.out_super)
    generate_null_distribution(args.pre_csv, args.csv, args.out_null)
