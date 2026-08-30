import os
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi

p50_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260706_sam_p200"
sample_id = 8
pos_id = 1

pre_path = os.path.join(p50_dir, f"{sample_id}-{pos_id}-0.tif")
post_path = os.path.join(p50_dir, f"{sample_id}-{pos_id}-1.tif")

pre = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
post = cv2.imdecode(np.fromfile(post_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0

# 1. FFT Grid Method
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

gy, gx = extract_fft_grid(pre)
delta_fft = (post[gy, gx] - pre[gy, gx]) / pre[gy, gx] * 100.0
fft_mean = np.mean(delta_fft)
fft_std = np.std(delta_fft)

# 2. Original Method (Otsu + Connected Components)
tophat = cv2.morphologyEx(pre, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
blurred_8u = (np.clip(blurred * 20, 0, 1) * 255).astype(np.uint8)
_, thresh = cv2.threshold(blurred_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

valid_indices = (stats[:, cv2.CC_STAT_AREA] >= 4) & (stats[:, cv2.CC_STAT_AREA] <= 100)
centroids_valid = centroids[valid_indices]
cx = np.round(centroids_valid[:, 0]).astype(np.int32)
cy = np.round(centroids_valid[:, 1]).astype(np.int32)

valid_mask = (cx >= 15) & (cx < pre.shape[1]-15) & (cy >= 15) & (cy < pre.shape[0]-15)
cx = cx[valid_mask]
cy = cy[valid_mask]

# Delta for Original
delta_otsu = (post[cy, cx] - pre[cy, cx]) / pre[cy, cx] * 100.0
otsu_mean = np.mean(delta_otsu)
otsu_std = np.std(delta_otsu)

# 3. Original Method + 99.5th Percentile Cap (Like V3)
p995 = np.percentile(pre[cy, cx], 99.5)
cap_mask = pre[cy, cx] >= p995
delta_cap = delta_otsu[cap_mask]
cap_mean = np.mean(delta_cap)
cap_std = np.std(delta_cap)

print("--- Extraction Comparison on 8-5 ---")
print(f"1. FFT Grid (N={len(delta_fft)}): Mean = {fft_mean:.4f}%, Std = {fft_std:.4f}%")
print(f"2. Otsu Only (N={len(delta_otsu)}): Mean = {otsu_mean:.4f}%, Std = {otsu_std:.4f}%")
print(f"3. Otsu + 99.5% Cap (N={len(delta_cap)}): Mean = {cap_mean:.4f}%, Std = {cap_std:.4f}%")
