import os
import cv2
import numpy as np
import scipy.ndimage as ndi

def get_fair_correlation(p_pre, p_post, warp_matrix, dataset_name):
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    rows, cols = img0.shape

    # Use INTER_CUBIC to preserve peak sharpness
    img1_aligned = cv2.warpAffine(img1, warp_matrix, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    # Extract pillars from img0
    f = np.fft.fft2(img0 - np.nanmean(img0))
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
    valid_border = np.zeros_like(img0, dtype=bool)
    valid_border[margin:-margin, margin:-margin] = True
    gy, gx = np.where(local_max & valid_border)

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
    
    def get_defects(img):
        img_filled = np.nan_to_num(img, nan=np.nanmean(img))
        blurred = cv2.GaussianBlur(img_filled, (51, 51), 0)
        diff = img_filled - blurred
        std = np.std(diff[~np.isnan(img)])
        mask = (np.abs(diff) > 3 * std).astype(np.uint8)
        # Dilate mask by 3 pixels to hide interpolation ringing artifacts
        mask_dilated = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        return mask_dilated.astype(bool)

    defects0 = get_defects(img0)
    defects1 = get_defects(img1_aligned)
    defect_mask_pillars = defects0[gy, gx] | defects1[gy, gx]

    clean_mask = valid & ~defect_mask_pillars

    corr = np.corrcoef(int0[clean_mask], int1[clean_mask])[0, 1]
    print(f"{dataset_name} | CLEAN Corr (Cubic + Dilated Mask): {corr:.4f} | Valid pillars: {np.sum(clean_mask)}")

# 1. 260704 Control
M_260704 = np.float32([[1, 0, 5.5], [0, 1, -20.5]]) # True optimal translation
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-1.tif"
get_fair_correlation(p_pre, p_post, M_260704, "260704 (Control)")

# 2. p50 Full Exp
M_p50 = np.float32([[ 9.9999946e-01,  1.0264802e-03,  1.4669377e+02], [-1.0264802e-03,  9.9999946e-01, -3.6355522e+00]]) # True optimal affine
p_pre = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-0.tif"
p_post = r"G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM\1-1-1.tif"
get_fair_correlation(p_pre, p_post, M_p50, "p50 Full Exp")
