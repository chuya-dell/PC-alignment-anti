import cv2
import numpy as np
import pandas as pd
import argparse
import time

def warp_and_sample(pre_img_path, post_img_path, pre_csv_path, output_csv_path, min_match=10, ransac_thresh=3.0):
    t0 = time.time()
    print(f"Loading Pre image: {pre_img_path}")
    pre_img = cv2.imdecode(np.fromfile(pre_img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    print(f"Loading Post image: {post_img_path}")
    post_img = cv2.imdecode(np.fromfile(post_img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    # 1. Compute robust alignment matrix (we can use ECC or Phase Correlation + RANSAC)
    # For now, let's use a robust feature-based or FFT phase correlation approach.
    # To keep this script self-contained and highly robust for 16-bit shift:
    
    # Simple Phase Correlation for Translation
    h, w = pre_img.shape
    # Apply Hanning window to reduce edge effects in FFT
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    pre_w = pre_img * window
    post_w = post_img * window
    
    shift, response = cv2.phaseCorrelate(pre_w, post_w)
    dx, dy = shift
    print(f"Phase Correlation Shift (Post -> Pre): dx={dx:.2f}, dy={dy:.2f}")
    
    # Create affine warp matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Warp Post image to Pre coordinates
    post_warped = cv2.warpAffine(post_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 2. Load Pre detection coordinates
    df_pre = pd.read_csv(pre_csv_path)
    print(f"Loaded {len(df_pre)} pillars from Pre CSV.")
    
    # 3. Sample intensities at exact identical coordinates
    # Using 3x3 local average around integer coordinates to be robust to sub-pixel noise
    pre_blur = cv2.blur(pre_img, (3, 3))
    post_blur = cv2.blur(post_warped, (3, 3))
    
    xi = np.round(df_pre['x'].values).astype(np.int32)
    yi = np.round(df_pre['y'].values).astype(np.int32)
    
    # Filter bounds
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    
    df_valid = df_pre[valid].copy()
    xi = xi[valid]
    yi = yi[valid]
    
    # Sample
    df_valid['pre_intensity'] = pre_blur[yi, xi]
    df_valid['post_intensity'] = post_blur[yi, xi]
    
    # Calculate paired differences
    df_valid['delta_I'] = df_valid['post_intensity'] - df_valid['pre_intensity']
    # Add epsilon to prevent div/0
    eps = 1e-6
    df_valid['ratio_I'] = df_valid['post_intensity'] / (df_valid['pre_intensity'] + eps)
    df_valid['log_ratio_I'] = np.log((df_valid['post_intensity'] + eps) / (df_valid['pre_intensity'] + eps))
    
    # Remove border zeros from warp
    df_valid = df_valid[df_valid['post_intensity'] > 0]
    
    df_valid.to_csv(output_csv_path, index=False)
    
    print(f"Sampled {len(df_valid)} valid pillars.")
    print(f"Median Delta: {df_valid['delta_I'].median():.6f}")
    print(f"10% Trimmed Mean Delta: {scipy.stats.trim_mean(df_valid['delta_I'], 0.1):.6f}")
    
    print(f"Saved paired results to {output_csv_path}")
    print(f"Completed in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    import scipy.stats
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=str, required=True)
    parser.add_argument("--post", type=str, required=True)
    parser.add_argument("--pre-csv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    
    args = parser.parse_args()
    warp_and_sample(args.pre, args.post, args.pre_csv, args.out)
