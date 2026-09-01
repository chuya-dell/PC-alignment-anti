import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import pandas as pd

data_dir = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM"

def get_coarse_shift(img0, img1):
    rows, cols = img0.shape
    patch_size = 1000
    cy, cx = rows // 2, cols // 2
    patch = img0[cy - patch_size//2:cy + patch_size//2, cx - patch_size//2:cx + patch_size//2]
    res = cv2.matchTemplate(img1, patch, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    match_x, match_y = max_loc
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
        
        # 1. SNR of img0
        # Simple SNR estimate: mean of brightest pixels vs std of darkest pixels
        sorted_img = np.sort(img0.flatten())
        bg_std = np.std(sorted_img[:int(len(sorted_img)*0.5)])
        signal_mean = np.mean(sorted_img[-int(len(sorted_img)*0.01):]) - np.mean(sorted_img[:int(len(sorted_img)*0.5)])
        snr = signal_mean / bg_std if bg_std > 0 else 0

        # 2. Coarse Shift & ECC
        dx_c, dy_c = get_coarse_shift(img0, img1)
        warp_matrix = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
        img0_8u = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
        img1_8u = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
        ecc_score = 0.0
        try:
            retval, warp_matrix = cv2.findTransformECC(img0_8u, img1_8u, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 1)
            ecc_score = retval
        except cv2.error:
            ecc_score = np.nan
            
        img1_aligned = cv2.warpAffine(img1, warp_matrix, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        
        # 3. Pillar extraction
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

        pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
        pad1 = np.pad(img1_aligned, 1, mode='constant', constant_values=np.nan)
        int0 = np.zeros(len(gy), dtype=np.float32)
        int1 = np.zeros(len(gy), dtype=np.float32)
        gy_p, gx_p = gy + 1, gx + 1
        for dy_ in [-1, 0, 1]:
            for dx_ in [-1, 0, 1]:
                int0 += pad0[gy_p + dy_, gx_p + dx_]
                int1 += pad1[gy_p + dy_, gx_p + dx_]

        valid = ~np.isnan(int1)
        
        # 4. Defect Masks
        def get_defects(img):
            img_filled = np.nan_to_num(img, nan=np.nanmean(img))
            blurred = cv2.GaussianBlur(img_filled, (51, 51), 0)
            diff = img_filled - blurred
            std = np.std(diff[~np.isnan(img)])
            mask_bin = (np.abs(diff) > 3 * std).astype(np.uint8)
            mask_dilated = cv2.dilate(mask_bin, np.ones((7, 7), np.uint8))
            return mask_bin, mask_dilated.astype(bool)

        def_bin0, def_dil0 = get_defects(img0)
        def_bin1, def_dil1 = get_defects(img1_aligned)
        
        defect_area = np.sum(def_bin0 | def_bin1) # Raw defect area in pixels
        defect_mask_pillars = def_dil0[gy, gx] | def_dil1[gy, gx]

        clean_mask = valid & ~defect_mask_pillars
        
        if np.sum(clean_mask) > 1000:
            corr = np.corrcoef(int0[clean_mask], int1[clean_mask])[0, 1]
            results.append({
                'sample': s, 
                'fov': f, 
                'corr': corr, 
                'valid_pillars': np.sum(clean_mask),
                'snr': snr,
                'ecc_score': ecc_score,
                'defect_area': defect_area
            })

df = pd.DataFrame(results)
df.to_csv(r"C:\Users\chuya\.gemini\antigravity\brain\60f5a68e-8281-4136-8936-9e4e95854572\enriched_correlation.csv", index=False)
print("Saved enriched_correlation.csv")
