import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_valley_bg(img):
    local_min = ndi.minimum_filter(img, size=5) == img
    margin = 15
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    valleys = img[local_min & mask]
    return np.mean(valleys) if len(valleys) > 0 else np.nan

def extract_fft_grid(img, pitch_px=6.29):
    f = np.fft.fft2(img - np.mean(img))
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / pitch_px
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    margin = 15
    valid = np.zeros_like(img, dtype=bool)
    valid[margin:-margin, margin:-margin] = True
    return np.where(local_max & valid)

def get_3x3_pillar_intensities(img, gy, gx):
    padded = np.pad(img, 1, mode='reflect')
    gy_p, gx_p = gy + 1, gx + 1
    res = np.zeros_like(gy_p, dtype=np.float32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            res += padded[gy_p + dy, gx_p + dx]
    return res / 9.0

def process_file(p):
    base = os.path.basename(p)
    a, b, c = map(int, base.replace('.tif', '').split('-'))
    post_p = p.replace('-0.tif', '-1.tif')
    if not os.path.exists(post_p): return None
    
    img0 = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(post_p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    # Background
    bg0 = get_valley_bg(img0)
    bg1 = get_valley_bg(img1)
    
    # Pillars
    gy, gx = extract_fft_grid(img0)
    p0 = get_3x3_pillar_intensities(img0, gy, gx)
    p1 = get_3x3_pillar_intensities(img1, gy, gx)
    
    # Ratios
    bg_ratio = bg1 / bg0
    p_ratio = np.mean(p1) / np.mean(p0)
    
    # Deltas
    delta_raw = (p1 - p0) / p0 * 100.0
    
    p0_norm = p0 / bg0
    p1_norm = p1 / bg1
    delta_corr = (p1_norm - p0_norm) / p0_norm * 100.0
    
    return {
        "Sample": a,
        "Position": b,
        "BG_Ratio": bg_ratio,
        "Pillar_Ratio": p_ratio,
        "Delta_Raw_Mean": np.mean(delta_raw),
        "Delta_Corr_Mean": np.mean(delta_corr),
        "Delta_Corr_Arr": delta_corr
    }

if __name__ == '__main__':
    pre_files = glob.glob(os.path.join(p50_dir, "**", "*-0.tif"), recursive=True)
    results = []
    
    print(f"Processing {len(pre_files)} files...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(process_file, pre_files):
            if res:
                results.append(res)

    df = pd.DataFrame(results)
    
    # 1. Ratios Comparison
    print("=== Intensity Change Ratios ===")
    ratios = df.groupby('Sample')[['BG_Ratio', 'Pillar_Ratio']].mean()
    print(ratios.to_string())
    
    # 2. Z-Score and Neg Grid % Calculation (Sample 1 vs 8)
    # Filter Blank (Sample 8) array
    blank_fovs = df[df['Sample'] == 8]
    if len(blank_fovs) > 0:
        all_blank_deltas = np.concatenate(blank_fovs['Delta_Corr_Arr'].values)
        blank_mean = np.mean(all_blank_deltas)
        blank_std = np.std(all_blank_deltas)
        qc_thresh = blank_mean - 3 * blank_std
        print(f"\nBlank Threshold (Mean - 3σ): {qc_thresh:.4f}% (Mean={blank_mean:.4f}, Std={blank_std:.4f})")
        
        for sample_id in [1, 8]:
            sample_fovs = df[df['Sample'] == sample_id]
            if len(sample_fovs) == 0: continue
            all_deltas = np.concatenate(sample_fovs['Delta_Corr_Arr'].values)
            s_mean = np.mean(all_deltas)
            z_score = (s_mean - blank_mean) / blank_std
            neg_grid = np.sum(all_deltas < qc_thresh) / len(all_deltas) * 100.0
            print(f"Sample {sample_id} -> Corrected Mean: {s_mean:.4f}%, Z-Score: {z_score:.4f}, Neg Grid: {neg_grid:.2f}%")

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    groups = df.groupby('Sample')
    
    data_raw = [group['Delta_Raw_Mean'].values for name, group in groups]
    data_corr = [group['Delta_Corr_Mean'].values for name, group in groups]
    positions = [name for name, group in groups]
    
    plt.subplot(1, 2, 1)
    plt.boxplot(data_raw, positions=positions, patch_artist=True, boxprops=dict(facecolor='pink'))
    plt.title('RAW Delta (%)')
    plt.xlabel('Sample Number')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(data_corr, positions=positions, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title('CORRECTED Delta (%)')
    plt.xlabel('Sample Number')
    plt.grid(True)
    
    plt.tight_layout()
    out_path = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\drift_plot_corrected.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved corrected plot to {out_path}")
