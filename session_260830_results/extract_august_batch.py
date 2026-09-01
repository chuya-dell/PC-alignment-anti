import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import concurrent.futures
import pickle
import pandas as pd

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'
base_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D'
datasets = ['260824_p50_SHC6OH', '260826-p50-sam', '260829-p50-sam']

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

def process_fov(args):
    dataset, s, f = args
    data_dir = os.path.join(base_dir, dataset)
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p_pre) or not os.path.exists(p_post): return None
    
    timestamp = os.path.getmtime(p_post)
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    rows, cols = img0.shape
    
    dx_c, dy_c = get_coarse_shift(img0, img1)
    warp = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
    try:
        _, warp = cv2.findTransformECC(np.clip(img0*255, 0, 255).astype(np.uint8), np.clip(img1*255, 0, 255).astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4), None, 1)
    except: pass
    img1_al = cv2.warpAffine(img1, warp, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    
    d0 = np.nan_to_num(img0, nan=np.nanmean(img0))
    diff0 = d0 - cv2.GaussianBlur(d0, (51, 51), 0)
    std0 = np.std(diff0[~np.isnan(img0)])
    m0 = cv2.dilate((np.abs(diff0) > 3*std0).astype(np.uint8), np.ones((7,7), np.uint8))
    g_mask = np.zeros_like(img0, dtype=bool)
    c_smooth = ndi.gaussian_filter1d(np.nanmean(img0, axis=0), 10)
    g_mask[:, c_smooth < np.percentile(c_smooth, 3)] = True
    g_mask = cv2.dilate(g_mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    invalid = (m0 > 0) | g_mask | np.isnan(img1_al)
    
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    freq = 1.0 / 6.38
    cy, cx = rows//2, cols//2
    y, x = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    r = np.sqrt(x**2 + y**2)
    fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    img_filt = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(f_fft) * fft_mask)).real
    
    local_max = ndi.maximum_filter(img_filt, size=5) == img_filt
    margin = 30
    gy, gx = np.where(local_max)
    valid = (gy > margin) & (gy < rows-margin) & (gx > margin) & (gx < cols-margin) & ~invalid[gy, gx]
    gy, gx = gy[valid], gx[valid]
    
    if len(gy) == 0: return None
    
    pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
    pad1 = np.pad(img1_al, 1, mode='constant', constant_values=np.nan)
    int0, int1 = np.zeros(len(gy), dtype=np.float32), np.zeros(len(gy), dtype=np.float32)
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            int0 += pad0[gy + 1 + dy_, gx + 1 + dx_]
            int1 += pad1[gy + 1 + dy_, gx + 1 + dx_]
            
    corr = np.corrcoef(int0, int1)[0, 1] if len(gy) > 100 else 0
    pillar_delta = np.mean((int0 - int1) / int0 * 100)
    
    return {'dataset': dataset, 's': s, 'f': f, 'timestamp': timestamp, 'pillar_delta': pillar_delta, 'corr': corr}

if __name__ == '__main__':
    args = [(d, s, f) for d in datasets for s in range(1, 9) for f in range(1, 9)]
    results = []
    print("Batch processing August datasets...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_fov, args):
            if res: results.append(res)
            
    df = pd.DataFrame(results)
    with open(os.path.join(out_dir, 'august_batch_results.pkl'), 'wb') as f:
        pickle.dump(df, f)
    print("Finished batch extraction.")
