import os
import numpy as np
import cv2
import pandas as pd
import scipy.ndimage as ndi

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
    margin = 20
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
            
    # Quick normalization without precise BG (correlation doesn't change with scalar scaling, 
    # but bg might vary slightly. Let's just use raw sums for correlation since bg is roughly constant per image)
    
    corr = np.corrcoef(sum0, sum1)[0, 1]
    return corr

print("Sweeping 260704 (SAM, Optical)...")
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-1.tif"

img0, img1, gy, gx = get_base_arrays(p_pre, p_post)

results = []
# Sweep from -6.0 to 6.0 in 0.5 increments
shifts = np.arange(-6.0, 6.5, 0.5)
for dy in shifts:
    for dx in shifts:
        c = calc_corr_for_shift(img0, img1, gy, gx, dx, dy)
        results.append((dx, dy, c))

df = pd.DataFrame(results, columns=["dx", "dy", "corr"])
max_row = df.loc[df['corr'].idxmax()]
print(f"\nMax Correlation: {max_row['corr']:.4f} at dx={max_row['dx']:.1f}, dy={max_row['dy']:.1f}")

# Also print the grid near the max
print("\nCorrelation Grid (dy around max, dx around max):")
max_dy = max_row['dy']
max_dx = max_row['dx']
grid_dy = np.arange(max_dy - 1.5, max_dy + 2.0, 0.5)
grid_dx = np.arange(max_dx - 1.5, max_dx + 2.0, 0.5)

print("dy \\ dx\t" + "\t".join([f"{x:.1f}" for x in grid_dx]))
for y in grid_dy:
    row_str = f"{y:.1f}\t"
    for x in grid_dx:
        match = df[(df['dx'] == x) & (df['dy'] == y)]
        if not match.empty:
            row_str += f"{match.iloc[0]['corr']:.4f}\t"
        else:
            row_str += "N/A\t"
    print(row_str)
