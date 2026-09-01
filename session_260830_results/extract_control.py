import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import concurrent.futures
import pickle

data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260830_P50_条件a~d'
out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'

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
    _, _, _, max_loc = cv2.minMaxLoc(res)
    return max_loc[0] - search_margin, max_loc[1] - search_margin

def process_fov(s, f):
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p_pre): return None
    
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    rows, cols = img0.shape
    
    dx_c, dy_c = get_coarse_shift(img0, img1)
    warp = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
    try:
        _, warp = cv2.findTransformECC(np.clip(img0*255, 0, 255).astype(np.uint8), np.clip(img1*255, 0, 255).astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4), None, 1)
    except: pass
    img1_al = cv2.warpAffine(img1, warp, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    
    # Defect mask
    d0 = np.nan_to_num(img0, nan=np.nanmean(img0))
    d1 = np.nan_to_num(img1_al, nan=np.nanmean(img1_al))
    diff0 = d0 - cv2.GaussianBlur(d0, (51, 51), 0)
    diff1 = d1 - cv2.GaussianBlur(d1, (51, 51), 0)
    std0 = np.std(diff0[~np.isnan(img0)])
    std1 = np.std(diff1[~np.isnan(img1_al)])
    m0 = cv2.dilate((np.abs(diff0) > 3*std0).astype(np.uint8), np.ones((7,7), np.uint8))
    m1 = cv2.dilate((np.abs(diff1) > 3*std1).astype(np.uint8), np.ones((7,7), np.uint8))
    defects = (m0 | m1).astype(bool)
    
    # Grooves
    c_smooth = ndi.gaussian_filter1d(np.nanmean(img0, axis=0), 10)
    r_smooth = ndi.gaussian_filter1d(np.nanmean(img0, axis=1), 10)
    g_mask = np.zeros_like(img0, dtype=bool)
    g_mask[:, c_smooth < np.percentile(c_smooth, 3)] = True
    g_mask[r_smooth < np.percentile(r_smooth, 6), :] = True
    g_mask = cv2.dilate(g_mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    
    # FFT Grid point (Fixing pitch to 6.38)
    pitch = 6.38
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    freq = 1.0 / pitch
    cy, cx = rows//2, cols//2
    y, x = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    r = np.sqrt(x**2 + y**2)
    fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    img_filt = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(f_fft) * fft_mask)).real
    local_max = ndi.maximum_filter(img_filt, size=5) == img_filt
    
    gy, gx = np.where(local_max)
    margin = 30
    valid = (gy > margin) & (gy < rows-margin) & (gx > margin) & (gx < cols-margin) & ~np.isnan(img1_al[gy, gx]) & ~defects[gy, gx] & ~g_mask[gy, gx]
    gy_fin, gx_fin = gy[valid], gx[valid]
    
    pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
    pad1 = np.pad(img1_al, 1, mode='constant', constant_values=np.nan)
    int0, int1 = np.zeros(len(gy_fin), dtype=np.float32), np.zeros(len(gy_fin), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            int0 += pad0[gy_fin + 1 + dy_, gx_fin + 1 + dx_]
            int1 += pad1[gy_fin + 1 + dy_, gx_fin + 1 + dx_]
            
    corr = np.corrcoef(int0, int1)[0, 1] if len(gy_fin) > 100 else 0
    delta = (int0 - int1) / int0 * 100
    
    # 2D Heatmap (32x32)
    bins = 32
    heatmap = np.zeros((bins, bins), dtype=np.float32)
    counts = np.zeros((bins, bins), dtype=np.float32)
    y_bins = np.digitize(gy_fin, np.linspace(0, rows, bins+1)) - 1
    x_bins = np.digitize(gx_fin, np.linspace(0, cols, bins+1)) - 1
    
    # fix out of bounds bins safely
    y_bins = np.clip(y_bins, 0, bins-1)
    x_bins = np.clip(x_bins, 0, bins-1)
    
    np.add.at(heatmap, (y_bins, x_bins), delta)
    np.add.at(counts, (y_bins, x_bins), 1)
    with np.errstate(invalid='ignore'):
        heatmap = heatmap / counts
        
    return {
        's': s, 'f': f, 
        'mean_delta': np.mean(delta), 
        'corr': corr, 
        'heatmap': heatmap,
        'delta_array': delta
    }

if __name__ == '__main__':
    args = [(s, f) for s in range(1, 6) for f in range(1, 9)]
    results = []
    print("Processing FOVs for control experiment...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_fov, [a[0] for a in args], [a[1] for a in args]):
            if res: results.append(res)
            
    with open(os.path.join(out_dir, 'control_data.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print("Saved control_data.pkl")
