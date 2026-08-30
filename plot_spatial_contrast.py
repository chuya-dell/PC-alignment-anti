import cv2
import numpy as np
import scipy.ndimage as ndi
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

data_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\同一視野"
out_dir = r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\scratch"
os.makedirs(out_dir, exist_ok=True)

def generate_heatmap(p_img, out_path):
    img_raw = cv2.imdecode(np.fromfile(p_img, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    if img_raw is None: return
    img = img_raw.astype(np.float32) / 65535.0
    rows, cols = img.shape

    # Extract pillars
    f = np.fft.fft2(img - np.nanmean(img))
    fshift = np.fft.fftshift(f)
    crow, ccol = rows // 2, cols // 2
    freq = 1.0 / 6.29
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x**2 + y**2)
    mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    fshift_filtered = fshift * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real
    
    margin = 50
    valid = np.zeros_like(img, dtype=bool)
    valid[margin:-margin, margin:-margin] = True
    gy, gx = np.where(local_max & valid)

    pad_img = np.pad(img, 1, mode='constant', constant_values=np.nan)
    gy_p, gx_p = gy + 1, gx + 1
    
    ints = np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            ints += pad_img[gy_p + dy_, gx_p + dx_]

    # Local background
    blurred = cv2.GaussianBlur(np.nan_to_num(img, nan=np.nanmean(img)), (51, 51), 0)
    bg_ints = np.zeros(len(gy), dtype=np.float32)
    pad_bg = np.pad(blurred, 1, mode='constant', constant_values=np.nan)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            bg_ints += pad_bg[gy_p + dy_, gx_p + dx_]
            
    contrast = (ints - bg_ints) * 65535.0
    
    # Filter extreme outliers for better visualization
    p1, p99 = np.percentile(contrast, 1), np.percentile(contrast, 99)
    valid_contrast_mask = (contrast >= p1) & (contrast <= p99)
    gx_c = gx[valid_contrast_mask]
    gy_c = gy[valid_contrast_mask]
    c_c = contrast[valid_contrast_mask]

    # Create spatial heatmap
    grid_x, grid_y = np.mgrid[0:cols:10, 0:rows:10]
    grid_z = griddata((gx_c, gy_c), c_c, (grid_x, grid_y), method='linear')

    plt.figure(figsize=(10, 8))
    plt.imshow(grid_z.T, extent=(0, cols, rows, 0), origin='upper', cmap='viridis')
    plt.colorbar(label='Pillar Contrast (ADU)')
    plt.title(f"Contrast Spatial Distribution: {os.path.basename(p_img)}")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    
    # Add a diagonal line to test the user's theory
    plt.plot([0, cols], [rows, 0], 'r--', alpha=0.5, label='Diagonal (BL to TR)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {out_path}")

for i in range(1, 7):
    p = os.path.join(data_dir, f"{i}.tif")
    if os.path.exists(p):
        out = os.path.join(out_dir, f"heatmap_{i}.png")
        generate_heatmap(p, out)
