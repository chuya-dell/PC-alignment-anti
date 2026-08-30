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

print("Sweeping p50 Full Exp...")
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-1.tif"

img0, img1, gy, gx = get_base_arrays(p_pre, p_post)

img1_00 = img1
img1_05 = ndi.shift(img1, (0.0, 0.5), order=1)
img1_50 = ndi.shift(img1, (0.5, 0.0), order=1)
img1_55 = ndi.shift(img1, (0.5, 0.5), order=1)

pad0 = np.pad(img0, 1, mode='reflect')
pad1_00 = np.pad(img1_00, 1, mode='reflect')
pad1_05 = np.pad(img1_05, 1, mode='reflect')
pad1_50 = np.pad(img1_50, 1, mode='reflect')
pad1_55 = np.pad(img1_55, 1, mode='reflect')

gy_p, gx_p = gy + 1, gx + 1
sum0 = np.zeros(len(gy), dtype=np.float32)
for dy_ in [-1, 0, 1]:
    for dx_ in [-1, 0, 1]:
        sum0 += pad0[gy_p + dy_, gx_p + dx_]

def calc_corr_fast(dx, dy):
    int_dx = int(np.floor(dx))
    int_dy = int(np.floor(dy))
    rem_dx = dx - int_dx
    rem_dy = dy - int_dy
    
    if rem_dx == 0.0 and rem_dy == 0.0:
        pad1 = pad1_00
    elif rem_dx == 0.5 and rem_dy == 0.0:
        pad1 = pad1_05
    elif rem_dx == 0.0 and rem_dy == 0.5:
        pad1 = pad1_50
    else:
        pad1 = pad1_55
        
    sum1 = np.zeros(len(gy), dtype=np.float32)
    gy_shift = gy_p - int_dy
    gx_shift = gx_p - int_dx
    
    # Bounds check
    valid_mask = (gy_shift - 1 >= 0) & (gy_shift + 1 < pad1.shape[0]) & (gx_shift - 1 >= 0) & (gx_shift + 1 < pad1.shape[1])
    
    if np.sum(valid_mask) < 100: return np.nan
        
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            sum1[valid_mask] += pad1[gy_shift[valid_mask] + dy_, gx_shift[valid_mask] + dx_]
            
    return np.corrcoef(sum0[valid_mask], sum1[valid_mask])[0, 1]

print("Sweeping around dx=148, dy=-4...")
shifts_dx = np.arange(143.0, 153.0, 0.5)
shifts_dy = np.arange(-9.0, 1.0, 0.5)

corr_map = np.zeros((len(shifts_dy), len(shifts_dx)))

results = []
for i, dy in enumerate(shifts_dy):
    for j, dx in enumerate(shifts_dx):
        c = calc_corr_fast(dx, dy)
        corr_map[i, j] = c
        results.append((dx, dy, c))

df = pd.DataFrame(results, columns=["dx", "dy", "corr"])
max_row = df.loc[df['corr'].idxmax()]
print(f"\nMax Correlation: {max_row['corr']:.4f} at dx={max_row['dx']:.1f}, dy={max_row['dy']:.1f}")

plt.figure(figsize=(10, 8))
plt.imshow(corr_map, extent=[-15, 15, 15, -15], cmap='viridis')
plt.colorbar(label='Correlation')
plt.scatter([max_row['dx']], [max_row['dy']], color='red', marker='x', s=100, label='Max')
plt.title(f"Correlation Map (Max: {max_row['corr']:.4f} at dx={max_row['dx']:.1f}, dy={max_row['dy']:.1f})")
plt.xlabel('dx (pixels)')
plt.ylabel('dy (pixels)')
plt.legend()
plt.savefig(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\sweep_corr_map.png")
print("Map saved to sweep_corr_map.png")
