import os
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def get_base_arrays(p_pre, p_post):
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    if img0 is None or img1 is None: return None
    
    img0 = img0.astype(np.float32) / 65535.0
    img1 = img1.astype(np.float32) / 65535.0
    
    # Extract grid on img0
    f = np.fft.fft2(img0 - np.mean(img0))
    fshift = np.fft.fftshift(f)
    rows, cols = img0.shape
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
    
    return img0, img1, gy, gx

def calc_corr_for_shift(img0, img1, gy, gx, dx, dy):
    img1_shifted = ndi.shift(img1, (dy, dx), order=1)
    
    pad0 = np.pad(img0, 1, mode='reflect')
    pad1 = np.pad(img1_shifted, 1, mode='reflect')
    gy_p, gx_p = gy + 1, gx + 1
    
    sum0 = np.zeros(len(gy), dtype=np.float32)
    sum1 = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            sum0 += pad0[gy_p + dy_, gx_p + dx_]
            sum1 += pad1[gy_p + dy_, gx_p + dx_]
            
    corr = np.corrcoef(sum0, sum1)[0, 1]
    return corr

print("Sweeping 260704 (SAM, Optical)... Range: -15 to +15")
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-1.tif"

img0, img1, gy, gx = get_base_arrays(p_pre, p_post)

# Sweep from -15.0 to 15.0 in 0.5 increments
shifts_dx = np.arange(-15.0, 15.5, 0.5)
shifts_dy = np.arange(-15.0, 15.5, 0.5)

corr_map = np.zeros((len(shifts_dy), len(shifts_dx)))

results = []
for i, dy in enumerate(shifts_dy):
    for j, dx in enumerate(shifts_dx):
        c = calc_corr_for_shift(img0, img1, gy, gx, dx, dy)
        corr_map[i, j] = c
        results.append((dx, dy, c))

df = pd.DataFrame(results, columns=["dx", "dy", "corr"])
max_row = df.loc[df['corr'].idxmax()]
print(f"\nMax Correlation: {max_row['corr']:.4f} at dx={max_row['dx']:.1f}, dy={max_row['dy']:.1f}")

# Plot 2D map
plt.figure(figsize=(10, 8))
plt.imshow(corr_map, extent=[-15, 15, 15, -15], cmap='viridis')
plt.colorbar(label='Correlation')
plt.scatter([max_row['dx']], [max_row['dy']], color='red', marker='x', s=100, label='Max')
plt.title(f"Correlation Map (Max: {max_row['corr']:.4f} at {max_row['dx']:.1f}, {max_row['dy']:.1f})")
plt.xlabel('dx (pixels)')
plt.ylabel('dy (pixels)')
plt.legend()
plt.savefig(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\sweep_corr_map.png")
print("Map saved to sweep_corr_map.png")
