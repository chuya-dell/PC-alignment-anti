import os
import cv2
import numpy as np
import scipy.ndimage as ndi
import concurrent.futures
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

out_dir = r'C:\Users\chuya\.gemini\antigravity\brain\9714e4b9-9c39-4048-aa79-f9fbd0ecb1a1\scratch'
data_dir = r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260828-p50-SAM'

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

def calculate_fwhm(patch):
    p = patch - np.min(patch)
    total = np.sum(p)
    if total == 0: return 0
    X, Y = np.meshgrid(np.arange(7), np.arange(7))
    x = np.sum(X * p) / total
    cov_xx = np.sum(p * (X - x)**2) / total
    return 2.355 * np.sqrt(cov_xx)

def process_fov(s, f):
    p_pre = os.path.join(data_dir, f'{s}-{f}-0.tif')
    p_post = os.path.join(data_dir, f'{s}-{f}-1.tif')
    if not os.path.exists(p_pre) or not os.path.exists(p_post): return None
    
    timestamp = os.path.getmtime(p_post)
    img0 = cv2.imdecode(np.fromfile(p_pre, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    img1 = cv2.imdecode(np.fromfile(p_post, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
    rows, cols = img0.shape
    
    # Align
    dx_c, dy_c = get_coarse_shift(img0, img1)
    warp = np.float32([[1, 0, dx_c], [0, 1, dy_c]])
    try:
        _, warp = cv2.findTransformECC(np.clip(img0*255, 0, 255).astype(np.uint8), np.clip(img1*255, 0, 255).astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4), None, 1)
    except: pass
    img1_al = cv2.warpAffine(img1, warp, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    
    # Masks
    d0 = np.nan_to_num(img0, nan=np.nanmean(img0))
    diff0 = d0 - cv2.GaussianBlur(d0, (51, 51), 0)
    std0 = np.std(diff0[~np.isnan(img0)])
    m0 = cv2.dilate((np.abs(diff0) > 3*std0).astype(np.uint8), np.ones((7,7), np.uint8))
    g_mask = np.zeros_like(img0, dtype=bool)
    c_smooth = ndi.gaussian_filter1d(np.nanmean(img0, axis=0), 10)
    g_mask[:, c_smooth < np.percentile(c_smooth, 3)] = True
    g_mask = cv2.dilate(g_mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    invalid = (m0 > 0) | g_mask | np.isnan(img1_al)
    
    # FFT
    f_fft = np.fft.fft2(img0 - np.nanmean(img0))
    freq = 1.0 / 6.38
    cy, cx = rows//2, cols//2
    y, x = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    r = np.sqrt(x**2 + y**2)
    fft_mask = (r >= rows*freq - 15) & (r <= rows*freq + 15)
    img_filt = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(f_fft) * fft_mask)).real
    
    # Pillars (Maxima) & Background (Minima)
    local_max = ndi.maximum_filter(img_filt, size=5) == img_filt
    local_min = ndi.minimum_filter(img_filt, size=5) == img_filt
    
    margin = 30
    def extract_points(mask):
        gy, gx = np.where(mask)
        valid = (gy > margin) & (gy < rows-margin) & (gx > margin) & (gx < cols-margin) & ~invalid[gy, gx]
        return gy[valid], gx[valid]
    
    py, px = extract_points(local_max)
    by, bx = extract_points(local_min)
    
    pad0 = np.pad(img0, 1, mode='constant', constant_values=np.nan)
    pad1 = np.pad(img1_al, 1, mode='constant', constant_values=np.nan)
    
    def get_delta_corr(gy_fin, gx_fin):
        if len(gy_fin) == 0: return np.nan, np.nan
        int0, int1 = np.zeros(len(gy_fin), dtype=np.float32), np.zeros(len(gy_fin), dtype=np.float32)
        for dy_ in [-1, 0, 1]:
            for dx_ in [-1, 0, 1]:
                int0 += pad0[gy_fin + 1 + dy_, gx_fin + 1 + dx_]
                int1 += pad1[gy_fin + 1 + dy_, gx_fin + 1 + dx_]
        corr = np.corrcoef(int0, int1)[0, 1] if len(gy_fin) > 100 else 0
        return np.mean((int0 - int1) / int0 * 100), corr

    pillar_delta, corr = get_delta_corr(py, px)
    bg_delta, _ = get_delta_corr(by, bx)
    
    # FWHM
    pad0_fwhm = np.pad(img0, 3, mode='constant', constant_values=np.nan)
    patches = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            patches.append(pad0_fwhm[py + 3 + dy, px + 3 + dx])
    patches = np.stack(patches, axis=-1).reshape(-1, 7, 7)
    mean_patch = np.nanmean(patches, axis=0)
    fwhm = calculate_fwhm(mean_patch)
    
    return {
        's': s, 'f': f,
        'timestamp': timestamp,
        'pillar_delta': pillar_delta,
        'bg_delta': bg_delta,
        'fwhm': fwhm,
        'corr': corr
    }

if __name__ == '__main__':
    args = [(s, f) for s in range(1, 9) for f in range(1, 9)]
    results = []
    print("Processing FOVs...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_fov, [a[0] for a in args], [a[1] for a in args]):
            if res: results.append(res)
            
    df = pd.DataFrame(results)
    df = df[df['corr'] >= 0.70].copy() # Filter out bad alignments
    df['time_rel'] = (df['timestamp'] - df['timestamp'].min()) / 60
    
    with open(os.path.join(out_dir, 'drift_results.pkl'), 'wb') as f:
        pickle.dump(df, f)
        
    print("Data extracted. Generating plots...")
    
    # 1. Global Time Drift (Pillar vs BG)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(df['time_rel'], df['pillar_delta'], c='blue', label='Pillars')
    ax1.scatter(df['time_rel'], df['bg_delta'], c='red', label='Background (Valleys)')
    
    sp, ip, _, pp, _ = stats.linregress(df['time_rel'], df['pillar_delta'])
    sb, ib, _, pb, _ = stats.linregress(df['time_rel'], df['bg_delta'])
    ax1.plot(df['time_rel'], ip + sp * df['time_rel'], 'b--', label=f'Pillars: {sp:.4f}/min (p={pp:.4f})')
    ax1.plot(df['time_rel'], ib + sb * df['time_rel'], 'r--', label=f'BG: {sb:.4f}/min (p={pb:.4f})')
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Mean Delta (%)')
    ax1.set_title('Global Time Drift: Pillars vs Background')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Global Time Drift (FWHM)
    ax2.scatter(df['time_rel'], df['fwhm'], c='green', label='FWHM')
    sf, inf, _, pf, _ = stats.linregress(df['time_rel'], df['fwhm'])
    ax2.plot(df['time_rel'], inf + sf * df['time_rel'], 'g--', label=f'FWHM: {sf:.4f}/min (p={pf:.4f})')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('FWHM (pixels)')
    ax2.set_title('Global Time Drift: FWHM (Focus)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'global_drift.png'))
    
    # 3. Intra-Sample (FOV) Drift
    intra_mean = df.groupby('f').agg({
        'pillar_delta': 'mean',
        'bg_delta': 'mean',
        'fwhm': 'mean'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(intra_mean['f'], intra_mean['pillar_delta'], 'b-o', label='Pillars')
    ax1.plot(intra_mean['f'], intra_mean['bg_delta'], 'r-o', label='Background')
    ax1.set_xlabel('FOV Index (1 to 8)')
    ax1.set_ylabel('Mean Delta (%)')
    ax1.set_title('Intra-Sample Drift (Averaged across S1-S8)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(intra_mean['f'], intra_mean['fwhm'], 'g-o', label='FWHM')
    ax2.set_xlabel('FOV Index (1 to 8)')
    ax2.set_ylabel('FWHM (pixels)')
    ax2.set_title('Intra-Sample FWHM Drift')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'intra_drift.png'))
    
    print("Global Drift Stats:")
    print(f"  Pillar Delta: slope={sp:.5f}, p={pp:.4f}")
    print(f"  BG Delta:     slope={sb:.5f}, p={pb:.4f}")
    print(f"  FWHM:         slope={sf:.5f}, p={pf:.4f}")
    
    print("\nIntra-Sample (FOV 1->8) Stats:")
    sp_i, _, _, pp_i, _ = stats.linregress(df['f'], df['pillar_delta'])
    sb_i, _, _, pb_i, _ = stats.linregress(df['f'], df['bg_delta'])
    sf_i, _, _, pf_i, _ = stats.linregress(df['f'], df['fwhm'])
    print(f"  Pillar Delta: slope={sp_i:.5f}/FOV, p={pp_i:.4f}")
    print(f"  BG Delta:     slope={sb_i:.5f}/FOV, p={pb_i:.4f}")
    print(f"  FWHM:         slope={sf_i:.5f}/FOV, p={pf_i:.4f}")
