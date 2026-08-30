import os
import glob
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_valley_snr(img):
    local_max = ndi.maximum_filter(img, size=5) == img
    local_min = ndi.minimum_filter(img, size=5) == img
    margin = 10
    mask = np.zeros_like(img, dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    peaks = img[local_max & mask]
    valleys = img[local_min & mask]
    if len(peaks) < 100 or len(valleys) < 100: return 0
    return (np.mean(peaks) - np.mean(valleys)) / np.std(valleys) if np.std(valleys) > 0 else 0

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

def extract_otsu(img):
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
    # p50 might need different clipping to catch faint pillars, but let's use the v3 standard
    blurred_8u = (np.clip(blurred * 20, 0, 1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(blurred_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    valid = (stats[:, cv2.CC_STAT_AREA] >= 4) & (stats[:, cv2.CC_STAT_AREA] <= 100)
    c = centroids[valid]
    cx, cy = np.round(c[:, 0]).astype(np.int32), np.round(c[:, 1]).astype(np.int32)
    valid_mask = (cx >= 15) & (cx < img.shape[1]-15) & (cy >= 15) & (cy < img.shape[0]-15)
    return cy[valid_mask], cx[valid_mask]

# 1. 8-5 Raw Intensity Comparison
sample_id, pos_id = 8, 5
pre_path = os.path.join(p50_dir, f"{sample_id}-{pos_id}-0.tif")
post_path = os.path.join(p50_dir, f"{sample_id}-{pos_id}-1.tif")
pre = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
post = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

print("=== Raw Intensity Check for 8-5 ===")
# FFT
gy, gx = extract_fft_grid(pre)
pre_fft = pre[gy, gx]
post_fft = post[gy, gx]
delta_fft = (post_fft - pre_fft) / pre_fft * 100
print(f"FFT Grid (N={len(pre_fft)}):")
print(f"  Pre Mean  = {np.mean(pre_fft):.5f}")
print(f"  Post Mean = {np.mean(post_fft):.5f}")
print(f"  Delta Mean= {np.mean(delta_fft):.5f}%")

# Otsu
oy, ox = extract_otsu(pre)
pre_otsu = pre[oy, ox]
post_otsu = post[oy, ox]
delta_otsu = (post_otsu - pre_otsu) / pre_otsu * 100
print(f"Otsu (N={len(pre_otsu)}):")
print(f"  Pre Mean  = {np.mean(pre_otsu):.5f}")
print(f"  Post Mean = {np.mean(post_otsu):.5f}")
print(f"  Delta Mean= {np.mean(delta_otsu):.5f}%")

# 2. SNR Grouping and Drift Plot for all FOVs
results = []
pre_files = glob.glob(os.path.join(p50_dir, "**", "*-0.tif"), recursive=True)
for p in pre_files:
    base = os.path.basename(p)
    a, b, c = map(int, base.replace('.tif', '').split('-'))
    post_p = p.replace('-0.tif', '-1.tif')
    if not os.path.exists(post_p): continue
    
    img0 = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(post_p, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    
    snr = get_valley_snr(img0)
    gy, gx = extract_fft_grid(img0)
    delta = np.mean((img1[gy, gx] - img0[gy, gx]) / img0[gy, gx] * 100)
    
    group = "Cross 1 (1-4)" if b <= 4 else "Cross 2 (5-8)"
    results.append({"Sample": a, "Position": b, "Group": group, "SNR": snr, "Delta": delta})

df = pd.DataFrame(results)

print("\n=== SNR by Position (b) ===")
print(df.groupby('Position')['SNR'].mean().round(2))
print("\n=== SNR by Group ===")
print(df.groupby('Group')['SNR'].mean().round(2))

# Plot Sample vs Delta
plt.figure(figsize=(8, 5))
import seaborn as sns
sns.boxplot(x='Sample', y='Delta', data=df, color='lightblue')
sns.swarmplot(x='Sample', y='Delta', data=df, color='black', alpha=0.7)
plt.title('Sample Number (Imaging Order) vs Delta (%)')
plt.xlabel('Sample Number (1=1nM, 8=Blank)')
plt.ylabel('Mean Delta (%) [FFT Grid]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\drift_plot.png", dpi=300)
print("\nSaved drift plot to drift_plot.png")
