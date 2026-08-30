import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def extract_fft_grid(img, pitch_px=6.29):
    f = np.fft.fft2(img - np.mean(img))
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / pitch_px
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    r_target = rows * freq
    mask = (r >= r_target - 15) & (r <= r_target + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    margin = 15
    valid_mask = np.zeros_like(img, dtype=bool)
    valid_mask[margin:-margin, margin:-margin] = True
    return np.where(local_max & valid_mask)

def analyze_drift_and_delta(sample_dir):
    results = []
    for sample_id in [1, 8]: # 1nM and Blank
        for pos_id in range(1, 9):
            pre_path = os.path.join(sample_dir, f"{sample_id}-{pos_id}-0.tif")
            post_path = os.path.join(sample_dir, f"{sample_id}-{pos_id}-1.tif")
            
            if not os.path.exists(pre_path) or not os.path.exists(post_path):
                continue
                
            pre = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            post = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            
            # Find BG (Valleys) in Pre
            local_min = ndi.minimum_filter(pre, size=5) == pre
            margin = 15
            mask = np.zeros_like(pre, dtype=bool)
            mask[margin:-margin, margin:-margin] = True
            valleys_pre = local_min & mask
            
            bg_pre_mean = np.mean(pre[valleys_pre])
            bg_post_mean = np.mean(post[valleys_pre]) # read exact same coords in post
            drift_baseline = (bg_post_mean - bg_pre_mean) / bg_pre_mean * 100.0 if bg_pre_mean > 0 else 0
            
            # Extract Pillars (FFT)
            gy, gx = extract_fft_grid(pre)
            
            # Read single pixel vs 3x3 average to see if variance drops
            # Original probably used a region. Let's do 3x3 sum/mean.
            delta_1px = (post[gy, gx] - pre[gy, gx]) / pre[gy, gx] * 100.0
            
            # 3x3 patch logic
            pre_padded = np.pad(pre, 1, mode='reflect')
            post_padded = np.pad(post, 1, mode='reflect')
            
            # +1 because of pad
            gy_p = gy + 1
            gx_p = gx + 1
            
            pre_3x3 = np.zeros_like(gy_p, dtype=np.float32)
            post_3x3 = np.zeros_like(gy_p, dtype=np.float32)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    pre_3x3 += pre_padded[gy_p + dy, gx_p + dx]
                    post_3x3 += post_padded[gy_p + dy, gx_p + dx]
            
            delta_3x3 = (post_3x3 - pre_3x3) / pre_3x3 * 100.0
            
            # Corrected delta
            corrected_1px = delta_1px - drift_baseline
            corrected_3x3 = delta_3x3 - drift_baseline
            
            results.append({
                "Sample": sample_id,
                "FOV": pos_id,
                "BG Drift (%)": drift_baseline,
                "Delta 1px Mean": np.mean(delta_1px),
                "Delta 1px Std": np.std(delta_1px),
                "Corr 1px Mean": np.mean(corrected_1px),
                "Delta 3x3 Mean": np.mean(delta_3x3),
                "Delta 3x3 Std": np.std(delta_3x3),
                "Corr 3x3 Mean": np.mean(corrected_3x3)
            })
            
    df = pd.DataFrame(results)
    
    print("--- Drift & Smoothing Analysis ---")
    print(df.to_string(index=False))
    
    # Calculate Z-score
    print("\n--- Z-Score Calculation (3x3 Patch, Corrected) ---")
    df_1 = df[df["Sample"] == 1]["Corr 3x3 Mean"]
    df_8 = df[df["Sample"] == 8]["Corr 3x3 Mean"]
    
    mean_1 = df_1.mean()
    mean_8 = df_8.mean()
    std_8 = df_8.std()
    
    print(f"Blank (Sample 8) Corrected Mean = {mean_8:.4f}%, Std = {std_8:.4f}%")
    print(f"1nM (Sample 1) Corrected Mean = {mean_1:.4f}%")
    if std_8 > 0:
        z = (mean_1 - mean_8) / std_8
        print(f"Corrected Z-Score = {z:.4f}")

analyze_drift_and_delta(p50_dir)
