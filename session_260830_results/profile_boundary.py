import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
from scipy import stats

img_path = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\同一視野\1.tif'
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
rows, cols = img.shape

# 1. Data-driven Boundary Estimation
img_blur = cv2.GaussianBlur(img, (201, 201), 0)
img_norm = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, mask = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(mask, 100, 200)
ey, ex = np.where(edges > 0)
points = np.column_stack((ex, ey)).astype(np.float32)
mean, eigenvectors = cv2.PCACompute(points, mean=None)
dir_vec = eigenvectors[0]
a = -dir_vec[1]
b = dir_vec[0]
c = -(a * mean[0,0] + b * mean[0,1])
norm = np.sqrt(a**2 + b**2)

# 2. Extract Pillars and Valleys
f = np.fft.fft2(img - np.nanmean(img))
fshift = np.fft.fftshift(f)
crow, ccol = rows // 2, cols // 2
freq = 1.0 / 6.29
y_grid, x_grid = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
r = np.sqrt(x_grid**2 + y_grid**2)
fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
fshift_filtered = fshift * fft_mask
img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered)).real
local_max = ndi.maximum_filter(img_filtered, size=5) == img_filtered
local_min = ndi.minimum_filter(img_filtered, size=5) == img_filtered
margin = 50
valid = np.zeros_like(img, dtype=bool)
valid[margin:-margin, margin:-margin] = True
py, px = np.where(local_max & valid)
vy, vx = np.where(local_min & valid)

# 3. 3x3 Intensities
def get_3x3_ints(y_coords, x_coords):
    pad_img = np.pad(img, 1, mode='constant', constant_values=np.nan)
    y_p, x_p = y_coords + 1, x_coords + 1
    ints = np.zeros(len(y_coords), dtype=np.float32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ints += pad_img[y_p + dy, x_p + dx]
    return ints

p_ints = get_3x3_ints(py, px)
v_ints = get_3x3_ints(vy, vx)

# 4. Signed Distance
p_dist = (a * px + b * py + c) / norm
v_dist = (a * vx + b * vy + c) / norm

# Align sign so SAM (Top-Right, dark region) is positive
if np.nanmean(p_ints[p_dist > 100]) > np.nanmean(p_ints[p_dist < -100]):
    p_dist = -p_dist
    v_dist = -v_dist

# 5. Profile Binning
bins = np.arange(np.min(p_dist), np.max(p_dist), 20)
def calc_profile(dist, ints, bins):
    b_mean, _, _ = stats.binned_statistic(dist, ints, statistic='mean', bins=bins)
    b_std, _, _ = stats.binned_statistic(dist, ints, statistic='std', bins=bins)
    b_count, _, _ = stats.binned_statistic(dist, ints, statistic='count', bins=bins)
    b_err = b_std / np.sqrt(b_count)
    return b_mean, b_err

p_mean, p_err = calc_profile(p_dist, p_ints, bins)
v_mean, v_err = calc_profile(v_dist, v_ints, bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Normalization
p_nosam = np.nanmean(p_ints[p_dist < -200])
v_nosam = np.nanmean(v_ints[v_dist < -200])
p_pct = p_mean / p_nosam * 100
p_err_pct = p_err / p_nosam * 100
v_pct = v_mean / v_nosam * 100
v_err_pct = v_err / v_nosam * 100

# 6. Plotting
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()

# Plot on % axis (ax2)
ax2.errorbar(bin_centers, p_pct, yerr=p_err_pct, fmt='o-', color='red', label='Pillars', markersize=4)
ax2.errorbar(bin_centers, v_pct, yerr=v_err_pct, fmt='s-', color='blue', label='Valleys', markersize=4)
ax2.set_ylabel('Normalized Intensity (% of NoSAM Average)', color='black')

# Plot on ADU axis (ax1) for scaling only, hide the lines
ax1.errorbar(bin_centers, p_mean, yerr=p_err, alpha=0) 
ax1.errorbar(bin_centers, v_mean, yerr=v_err, alpha=0)
ax1.set_xlabel('Signed Perpendicular Distance from Boundary (pixels)\nNegative = NoSAM, Positive = SAM')
ax1.set_ylabel('Absolute Intensity (ADU)', color='black')

ax2.axvline(x=0, color='k', linestyle='--', label='Estimated Boundary (PCA on Otsu Edges)')
ax2.legend(loc='lower left')
plt.title('Boundary Profile: Pillar vs Valley Intensity (1.tif)')
ax1.grid(True)
out_path = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch\profile_1.png'
plt.savefig(out_path)
print('Saved plot to:', out_path)
print(f'Pillar NoSAM Baseline ADU: {p_nosam:.2f}')
print(f'Valley NoSAM Baseline ADU: {v_nosam:.2f}')
