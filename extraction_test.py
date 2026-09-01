import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'

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
    pitch_px = rows / r
    return pitch_px

def process_fov(s, f):
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p_pre): return None
    img0_raw = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img1_raw = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    img0 = img0_raw.astype(np.float32) / 65535.0
    img1 = img1_raw.astype(np.float32) / 65535.0
    rows, cols = img0.shape
    total_area_px = rows * cols
    
    pitch_px = get_pitch(img0)
    unit_cell_area = (np.sqrt(3)/2) * (pitch_px**2)
    fov_um = (cols / pitch_px) * 0.46
    theory_base = total_area_px / unit_cell_area
    
    # 1. Grooves (Cross)
    col_means = np.nanmean(img0, axis=0)
    row_means = np.nanmean(img0, axis=1)
    c_smooth = ndi.gaussian_filter1d(col_means, 10)
    r_smooth = ndi.gaussian_filter1d(row_means, 10)
    c_thresh = np.percentile(c_smooth, 3)
    r_thresh = np.percentile(r_smooth, 6)
    g_mask = np.zeros_like(img0, dtype=bool)
    g_mask[:, c_smooth < c_thresh] = True
    g_mask[r_smooth < r_thresh, :] = True
    g_mask = cv2.dilate(g_mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    
    # 2. Alignment & Edge Loss
    patch_size = 1000
    cy, cx = rows//2, cols//2
    patch = img0[cy-500:cy+500, cx-500:cx+500]
    search_margin = 20
    img1_search = img1[max(0,cy-520):min(rows,cy+520), max(0,cx-520):min(cols,cx+520)]
    res = cv2.matchTemplate(img1_search, patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    dx_c = max_loc[0] - search_margin
    dy_c = max_loc[1] - search_margin
    
    warp = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
    img0_8u = np.clip(img0*255, 0, 255).astype(np.uint8)
    img1_8u = np.clip(img1*255, 0, 255).astype(np.uint8)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    try:
        _, warp = cv2.findTransformECC(img0_8u, img1_8u, warp, cv2.MOTION_EUCLIDEAN, crit, None, 1)
    except: pass
    img1_al = cv2.warpAffine(img1, warp, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    valid_overlap = ~np.isnan(img1_al)
    
    # 3. Defect Mask
    def get_def(img):
        d = np.nan_to_num(img, nan=np.nanmean(img))
        blur = cv2.GaussianBlur(d, (51, 51), 0)
        diff = d - blur
        std = np.std(diff[~np.isnan(img)])
        m = (np.abs(diff) > 3 * std).astype(np.uint8)
        return cv2.dilate(m, np.ones((7,7), np.uint8)).astype(bool)
    defects = get_def(img0) | get_def(img1_al)
    
    # Calculate specific areas
    overlap_frac = np.sum(valid_overlap) / total_area_px
    groove_frac = np.sum(g_mask & valid_overlap) / np.sum(valid_overlap)
    defect_frac = np.sum(defects & valid_overlap & ~g_mask) / np.sum(valid_overlap)
    
    expected_count = theory_base * overlap_frac * (1 - groove_frac) * (1 - defect_frac)
    
    # 4. Grid point method via FFT
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    fshift = np.fft.fftshift(f_fft)
    freq = 1.0 / pitch_px
    y, x = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    r = np.sqrt(x**2 + y**2)
    fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift * fft_mask)).real
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
    
    return {
        'Sample': s, 'FOV': f, 'Pitch': pitch_px, 'FOV_um': fov_um,
        'TheoryBase': theory_base, 'Expected': expected_count,
        'St1_Peak': st1, 'St2_L3': st2, 'St3_Groove': st3, 'St4_Final': st4,
        'ExtractionRate': st4 / expected_count * 100 if expected_count > 0 else 0
    }

results = []
print('Running extraction on S1 and S8 to assess counts...')
for s in [1, 8]:
    for f in range(1, 4):
        res = process_fov(s, f)
        if res: results.append(res)

df = pd.DataFrame(results)
print(df.to_string(index=False, float_format='%.2f'))

