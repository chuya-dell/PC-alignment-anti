import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

def get_pitch(img):
    rows, cols = img.shape
    f = np.fft.fft2(img - np.nanmean(img))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift)**2
    crow, ccol = rows//2, cols//2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = x**2 + y**2 > 100**2
    power_masked = power * mask
    max_idx = np.unravel_index(np.argmax(power_masked), power_masked.shape)
    dy, dx = max_idx[0] - crow, max_idx[1] - ccol
    r = np.sqrt(dx**2 + dy**2)
    return rows / r

def get_coarse_shift(img0, img1):
    rows, cols = img0.shape
    patch_size = 1000
    cy, cx = rows // 2, cols // 2
    patch = img0[cy - patch_size//2:cy + patch_size//2, cx - patch_size//2:cx + patch_size//2]
    search_margin = 15
    y1, y2 = max(0, cy - patch_size//2 - search_margin), min(rows, cy + patch_size//2 + search_margin)
    x1, x2 = max(0, cx - patch_size//2 - search_margin), min(cols, cx + patch_size//2 + search_margin)
    img1_search = img1[y1:y2, x1:x2]
    res = cv2.matchTemplate(img1_search, patch, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_loc[0] - search_margin, max_loc[1] - search_margin

def extract_defects(img):
    img_filled = np.nan_to_num(img, nan=np.nanmean(img))
    blurred = cv2.GaussianBlur(img_filled, (51, 51), 0)
    diff = img_filled - blurred
    std = np.std(diff[~np.isnan(img)])
    mask_bin = (np.abs(diff) > 3 * std).astype(np.uint8)
    return cv2.dilate(mask_bin, np.ones((7, 7), np.uint8)).astype(bool)

def find_grooves(img):
    col_means = np.nanmean(img, axis=0)
    row_means = np.nanmean(img, axis=1)
    c_smooth = ndi.gaussian_filter1d(col_means, 10)
    r_smooth = ndi.gaussian_filter1d(row_means, 10)
    c_thresh = np.percentile(c_smooth, 3)
    r_thresh = np.percentile(r_smooth, 6)
    g_mask = np.zeros_like(img, dtype=bool)
    g_mask[:, c_smooth < c_thresh] = True
    g_mask[r_smooth < r_thresh, :] = True
    return cv2.dilate(g_mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)

def process_fov(s, f):
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p_pre): return None
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    rows, cols = img0.shape
    total_area_px = rows * cols
    pitch_px = get_pitch(img0)
    theory_base = total_area_px / ((np.sqrt(3)/2) * (pitch_px**2))
    dx_c, dy_c = get_coarse_shift(img0, img1)
    warp = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
    try:
        _, warp = cv2.findTransformECC(np.clip(img0*255, 0, 255).astype(np.uint8), np.clip(img1*255, 0, 255).astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4), None, 1)
    except: pass
    img1_al = cv2.warpAffine(img1, warp, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    valid_overlap = ~np.isnan(img1_al)
    g_mask = find_grooves(img0)
    defects = extract_defects(img0) | extract_defects(img1_al)
    overlap_frac = np.sum(valid_overlap) / total_area_px
    groove_frac = np.sum(g_mask & valid_overlap) / np.sum(valid_overlap)
    defect_frac = np.sum(defects & valid_overlap & ~g_mask) / np.sum(valid_overlap)
    expected_count = theory_base * overlap_frac * (1 - groove_frac) * (1 - defect_frac)
    
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    freq = 1.0 / pitch_px
    cy, cx = rows//2, cols//2
    y, x = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    r = np.sqrt(x**2 + y**2)
    fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    img_filtered = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(f_fft) * fft_mask)).real
    local_max = ndi.maximum_filter(img_filtered, size=5) == img_filtered
    
    margin = 30
    valid_border = np.zeros_like(img0, dtype=bool)
    valid_border[margin:-margin, margin:-margin] = True
    gy, gx = np.where(local_max)
    st1 = len(gy)
    m_l3 = valid_border[gy, gx] & valid_overlap[gy, gx]
    gy_l3, gx_l3 = gy[m_l3], gx[m_l3]
    st2 = len(gy_l3)
    m_gr = ~g_mask[gy_l3, gx_l3]
    gy_g, gx_g = gy_l3[m_gr], gx_l3[m_gr]
    st3 = len(gy_g)
    m_def = ~defects[gy_g, gx_g]
    gy_fin, gx_fin = gy_g[m_def], gx_g[m_def]
    st4 = len(gy_fin)
    
    pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
    pad1 = np.pad(img1_al, 1, mode='constant', constant_values=np.nan)
    int0, int1 = np.zeros(st4, dtype=np.float32), np.zeros(st4, dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            int0 += pad0[gy_fin + 1 + dy_, gx_fin + 1 + dx_]
            int1 += pad1[gy_fin + 1 + dy_, gx_fin + 1 + dx_]
            
    corr = np.corrcoef(int0, int1)[0, 1] if st4 > 100 else 0
    delta_pct = (int0 - int1) / int0 * 100
    return {'s': s, 'f': f, 'Exp': expected_count, 'St1': st1, 'St2': st2, 'St3': st3, 'St4': st4, 'Rate': st4/expected_count*100 if expected_count>0 else 0, 'Corr': corr, 'Deltas': delta_pct}

results = []
for s in range(1, 9):
    for f in range(1, 9):
        res = process_fov(s, f)
        if res: results.append(res)

import pickle
with open(os.path.join(out_dir, 'p50_data.pkl'), 'wb') as f: pickle.dump(results, f)

