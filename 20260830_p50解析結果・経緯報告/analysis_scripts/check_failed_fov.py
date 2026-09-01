import os
import cv2
import numpy as np
import scipy.ndimage as ndi

p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\4-7-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\4-7-1.tif"

img0_raw = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
img1_raw = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)

img0 = img0_raw.astype(np.float32) / 65535.0
img1 = img1_raw.astype(np.float32) / 65535.0

# Coarse shift
rows, cols = img0.shape
patch_size = 1000
cy, cx = rows // 2, cols // 2
patch = img0[cy - patch_size//2:cy + patch_size//2, cx - patch_size//2:cx + patch_size//2]
res = cv2.matchTemplate(img1, patch, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
match_x, match_y = max_loc
dx_c = (match_x + patch_size//2) - cx
dy_c = (match_y + patch_size//2) - cy
print(f"Coarse shift: dx={dx_c}, dy={dy_c}, max_cc={max_val}")

warp_matrix = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
img0_8u = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
img1_8u = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
try:
    retval, warp_matrix = cv2.findTransformECC(img0_8u, img1_8u, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 1)
    print(f"ECC Transform:\n{warp_matrix}\nECC Score: {retval}")
except cv2.error as e:
    print("ECC failed:", e)

# 2D Sweep to find true peak
import pandas as pd
f = np.fft.fft2(img0 - np.mean(img0))
fshift = np.fft.fftshift(f)
crow, ccol = rows // 2, cols // 2
freq = 1.0 / 6.29
y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
r = np.sqrt(x**2 + y**2)
mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
fshift_filtered = fshift * mask
img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
margin = 30
valid = np.zeros_like(img0, dtype=bool)
valid[margin:-margin, margin:-margin] = True
gy, gx = np.where(local_max & valid)

def calc_corr_for_shift(dx, dy):
    img1_shifted = ndi.shift(img1, (dy, dx), order=1)
    pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
    pad1 = np.pad(img1_shifted, 1, mode='constant', constant_values=np.nan)
    gy_p, gx_p = gy + 1, gx + 1
    sum0 = np.zeros(len(gy), dtype=np.float32)
    sum1 = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            sum0 += pad0[gy_p + dy_, gx_p + dx_]
            sum1 += pad1[gy_p + dy_, gx_p + dx_]
    
    valid_mask = ~np.isnan(sum1) & ~np.isnan(sum0)
    if np.sum(valid_mask) > 1000:
        return np.corrcoef(sum0[valid_mask], sum1[valid_mask])[0, 1]
    return 0

# Sweep around the coarse shift
results = []
shifts = np.arange(-3.0, 3.5, 0.5)
for dy_sweep in shifts:
    for dx_sweep in shifts:
        c = calc_corr_for_shift(dx_c + dx_sweep, dy_c + dy_sweep)
        results.append((dx_c + dx_sweep, dy_c + dy_sweep, c))

df = pd.DataFrame(results, columns=["dx", "dy", "corr"])
max_row = df.loc[df['corr'].idxmax()]
print(f"\nSweep Max Correlation: {max_row['corr']:.4f} at dx={max_row['dx']:.1f}, dy={max_row['dy']:.1f}")

# Compare to ECC correlation
dx_ecc, dy_ecc = warp_matrix[0, 2], warp_matrix[1, 2]
c_ecc = calc_corr_for_shift(dx_ecc, dy_ecc)
print(f"ECC Correlation: {c_ecc:.4f} at dx={dx_ecc:.1f}, dy={dy_ecc:.1f}")
