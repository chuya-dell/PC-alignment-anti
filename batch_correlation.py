import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from glob import glob

data_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_coarse_shift(img0, img1):
    rows, cols = img0.shape
    patch_size = 1000
    cy, cx = rows // 2, cols // 2
    patch = img0[cy - patch_size//2:cy + patch_size//2, cx - patch_size//2:cx + patch_size//2]
    
    # Search over entire img1 (this might take a second but is robust)
    res = cv2.matchTemplate(img1, patch, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # max_loc is top-left corner of the match in img1
    match_x, match_y = max_loc
    # The center of the match in img1 is match_x + patch_size//2, match_y + patch_size//2
    # The shift is (match_center) - (original_center)
    dx = (match_x + patch_size//2) - cx
    dy = (match_y + patch_size//2) - cy
    
    return dx, dy

results = []

for s in range(1, 9):
    for f in range(1, 9):
        p_pre = os.path.join(data_dir, f"{s}-{f}-0.tif")
        p_post = os.path.join(data_dir, f"{s}-{f}-1.tif")
        
        if not os.path.exists(p_pre) or not os.path.exists(p_post):
            continue
            
        print(f"Processing S{s} F{f}...")
        
        img0_raw = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
        img1_raw = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
        
        if img0_raw is None or img1_raw is None:
            continue
            
        img0 = img0_raw.astype(np.float32) / 65535.0
        img1 = img1_raw.astype(np.float32) / 65535.0
        rows, cols = img0.shape
        
        # 1. Coarse Shift
        dx_c, dy_c = get_coarse_shift(img0, img1)
        
        # 2. ECC Affine
        warp_matrix = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
        img0_8u = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
        img1_8u = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4) # faster criteria
        try:
            _, warp_matrix = cv2.findTransformECC(img0_8u, img1_8u, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 1)
        except cv2.error:
            print(f"  ECC failed for S{s} F{f}, using coarse {dx_c}, {dy_c}")
            
        # 3. Align with Cubic
        img1_aligned = cv2.warpAffine(img1, warp_matrix, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        
        # 4. Extract pillars
        fft_img = np.fft.fft2(img0 - np.nanmean(img0))
        fshift = np.fft.fftshift(fft_img)
        crow, ccol = rows // 2, cols // 2
        freq = 1.0 / 6.29
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        r = np.sqrt(x**2 + y**2)
        mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
        fshift_filtered = fshift * mask
        img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
        local_max = ndi.maximum_filter(img_filtered.real, size=5) == img_filtered.real

        margin = 30
        valid_border = np.zeros_like(img0, dtype=bool)
        valid_border[margin:-margin, margin:-margin] = True
        gy, gx = np.where(local_max & valid_border)

        # 5. Pillar intensities
        def get_pillar_intensities(img, gy, gx):
            pad = np.pad(img, 1, mode='constant', constant_values=np.nan)
            gy_p, gx_p = gy + 1, gx + 1
            sum_int = np.zeros(len(gy), dtype=np.float32)
            for dy_ in [-1, 0, 1]:
                for dx_ in [-1, 0, 1]:
                    sum_int += pad[gy_p + dy_, gx_p + dx_]
            return sum_int

        int0 = get_pillar_intensities(img0, gy, gx)
        int1 = get_pillar_intensities(img1_aligned, gy, gx)

        valid = ~np.isnan(int1)
        
        # 6. Dilated Defect Mask
        def get_defects(img):
            img_filled = np.nan_to_num(img, nan=np.nanmean(img))
            blurred = cv2.GaussianBlur(img_filled, (51, 51), 0)
            diff = img_filled - blurred
            std = np.std(diff[~np.isnan(img)])
            mask_bin = (np.abs(diff) > 3 * std).astype(np.uint8)
            mask_dilated = cv2.dilate(mask_bin, np.ones((7, 7), np.uint8))
            return mask_dilated.astype(bool)

        defects0 = get_defects(img0)
        defects1 = get_defects(img1_aligned)
        defect_mask_pillars = defects0[gy, gx] | defects1[gy, gx]

        clean_mask = valid & ~defect_mask_pillars
        
        if np.sum(clean_mask) > 1000:
            corr = np.corrcoef(int0[clean_mask], int1[clean_mask])[0, 1]
            print(f"  -> Corr: {corr:.4f} (Valid: {np.sum(clean_mask)})")
            results.append({'sample': s, 'fov': f, 'corr': corr, 'valid': np.sum(clean_mask)})
        else:
            print("  -> Too few valid pillars")

import pandas as pd
df = pd.DataFrame(results)
df.to_csv(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\batch_correlation.csv", index=False)

# Plotting
plt.figure(figsize=(10, 6))
import seaborn as sns
sns.boxplot(x='sample', y='corr', data=df, color='lightblue')
sns.stripplot(x='sample', y='corr', data=df, color='red', alpha=0.7, jitter=True)
plt.title('Pre/Post Clean Correlation per Sample (p50 Full Exp)')
plt.xlabel('Sample (Concentration)')
plt.ylabel('Correlation (Cubic + Dilated Mask)')
plt.ylim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\batch_correlation_plot.png")
print("Saved batch_correlation.csv and batch_correlation_plot.png")
